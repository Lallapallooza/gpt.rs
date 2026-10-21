//! The x86 backend. Its pipeline lowers each module through `tile`, `loop` and `isa` to the buffer
//! form. It writes each function as C, a step function that calls its dispatches in order, with
//! AVX-512 intrinsics and OpenMP parallel loops. It compiles that C with clang, and runs a function
//! as one call of its step.
//!
//! It compiles for the target it is made with, which the program states where it starts
//! (`gpt_rs_backends::BackendOptions`): a preset, `host` or a target file, as `gpt-rs-cli target
//! list` shows.

mod compile;
mod emit;

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Result};
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs::profiling::{self, ScopeGuard, ScopeMeta};
use gpt_rs_ir::conversion::TensorToTile;
use gpt_rs_ir::dialects::arith::{ExpandMath, ExpandRandom};
use gpt_rs_ir::dialects::host::PlanMemory;
use gpt_rs_ir::dialects::tile::{
    FormGroups, InlineStaticMaps, NameTarget, Numerics, PadWithIdentity, PlaceEpilogues,
    PlanLayouts, Schedule, SealDispatches, SelectAtoms, SplitLaneFolds, WalkOrderLayouts,
    WidenReusedMaps, WinogradConv,
};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::ir::{DType, Module, TensorType};
use gpt_rs_ir::passes::{self as ir_passes, Canonicalize, Pipeline, WidenNarrowFloats};
use gpt_rs_ir::targets::{Feature, Target};
use libloading::Library;

pub use self::emit::{LineMap, LineOrigin};
use self::emit::{Memory, Produced, StepPlan};

/// How the x86 backend compiles, besides its target.
#[derive(Debug, Clone)]
pub struct X86Options {
    /// The rounding changes the compilation may make.
    pub numerics: Numerics,
}

impl Default for X86Options {
    /// Every rounding change of a dispatch, and no Winograd, which runs only when asked for: a
    /// product and the sum it enters round once where the target has FMA, a point whose padded
    /// reads give products of a zero pad into a sum is skipped, a sum may add its points in the
    /// lanes of a vector, a static scale of a dot rounds into its weight, and the math functions of
    /// f32 are polynomials.
    fn default() -> Self {
        Self {
            numerics: Numerics {
                contract: true,
                skip_pad: true,
                reassoc: true,
                winograd: false,
                scale_fold: true,
                expand_math: true,
            },
        }
    }
}

/// The x86 backend: its target, its options, and the parameters of the models it loaded.
pub struct X86Backend {
    params: ParamStore<HostTensor>,
    target: Target,
    options: X86Options,
    /// The target's reference and the options, which the programs it compiles depend on.
    identity: String,
}

impl X86Backend {
    /// The backend that compiles for `target`, which must have AVX-512, as `options` say.
    pub fn new(target: Target, options: X86Options) -> Result<Self> {
        ensure!(
            target.has_feature(Feature::Avx512f),
            "the x86 backend writes AVX-512, which the target `{}` lacks",
            target.name
        );
        ensure!(
            options.numerics.expand_math,
            "the x86 backend computes exp, log, tanh, sin, cos and erf of f32 as polynomials, \
             which its numerics must allow (`expand_math`)"
        );
        let identity = format!(
            "{} numerics={}",
            String::from(target.reference()),
            options.numerics
        );
        Ok(Self {
            params: ParamStore::default(),
            target,
            options,
            identity,
        })
    }
}

/// `gptrs_step`: runs a function on the addresses of its memory and of its arenas, calling
/// `mark` with `context` around each launch when `mark` is not null.
type StepFn = unsafe extern "C" fn(*const *mut c_void, *const *mut u8, Option<MarkFn>, *mut c_void);

/// What a step calls before (`begin` 1) and after (`begin` 0) its launch `launch`.
type MarkFn = unsafe extern "C" fn(*mut c_void, i64, i32);

/// A compiled function: its step, what the step takes, and the library that holds it.
pub struct Function {
    _library: Arc<Library>,
    step: StepFn,
    /// The types of its arguments, which each run checks.
    args: Vec<TensorType>,
    plan: StepPlan,
    /// The arenas of finished runs, which later runs take.
    arenas: Mutex<Vec<Vec<HostTensor>>>,
}

impl Function {
    /// Fails unless `args` have the types of the function's arguments, for which its step was
    /// compiled.
    fn check(&self, args: &[HostTensor]) -> Result<()> {
        ensure!(
            args.len() == self.args.len(),
            "the function takes {} arguments, got {}",
            self.args.len(),
            args.len()
        );
        for (index, (arg, ty)) in args.iter().zip(&self.args).enumerate() {
            let given = arg.tensor_type();
            ensure!(
                given == *ty,
                "argument {index} is {given}, and the function takes {ty}"
            );
        }
        Ok(())
    }

    /// The arenas of a run: those an earlier run finished with, or new ones, each with the room to
    /// start at a multiple of its alignment.
    fn take_arenas(&self) -> Result<Vec<HostTensor>> {
        if let Some(kept) = self.arenas.lock().expect("arenas poisoned").pop() {
            return Ok(kept);
        }
        let mut arenas = Vec::with_capacity(self.plan.arenas.len());
        for arena in &self.plan.arenas {
            let bytes = arena.bytes + arena.alignment - 1;
            arenas.push(HostTensor::zeros(&TensorType::new(DType::Ui8, [bytes]))?);
        }
        Ok(arenas)
    }
}

/// What the marks of a step hold while a profile collects: the dispatch of each launch, and the
/// scope of the launch that runs.
struct Marks<'p> {
    launches: &'p [String],
    open: Option<ScopeGuard>,
}

/// Opens the profile scope of launch `launch` of the step whose `Marks` `context` is, or closes it.
unsafe extern "C" fn mark(context: *mut c_void, launch: i64, begin: i32) {
    // SAFETY: the step passes back the context that `run` gave it: its `Marks`, which outlive the
    // call of the step.
    let marks = unsafe { &mut *context.cast::<Marks<'_>>() };
    marks.open = None;
    if begin == 0 {
        return;
    }
    let Some(name) = usize::try_from(launch)
        .ok()
        .and_then(|index| marks.launches.get(index))
    else {
        return;
    };
    marks.open = Some(profiling::backend_scope_with_meta(
        "x86_backend.dispatch",
        || {
            profiling::signature_id(name)
                .map(ScopeMeta::signature)
                .unwrap_or_default()
        },
    ));
}

impl PortableBackend for X86Backend {
    type TensorHandle = HostTensor;
    type Executable = Function;

    fn backend_name(&self) -> &str {
        "x86"
    }

    fn compile_identity(&self) -> &str {
        &self.identity
    }

    /// The passes, in order:
    /// 1. the default passes over `tensor`;
    /// 2. the module in blocks: padded windows read with the identity of their fold, the maps of
    ///    parameters larger than what they read computed in their readers, the maps whose readers
    ///    widen each element at several points widened once, the values computed before the run
    ///    laid out for their readers' lanes, ops on bf16 and f16 computed in f32, math functions
    ///    written as polynomials, uniform random draws as integer ops, and with `winograd`, 3x3
    ///    convolutions as Winograd's F(2x2, 3x3);
    /// 3. the blocks in dispatches, then the loops of each dispatch for the target: its atoms, its
    ///    schedule (the register tile, the loops, each worker's share and the pieces of the caches
    ///    that need one), the blocks at its register tile, and the values computed before the run
    ///    laid out in the order their reader walks them;
    /// 4. `outline-static`, which moves work over parameters to load time;
    /// 5. the lowering to the buffer form: each dispatch through `loop` to `isa`, and each
    ///    function's buffers in one arena.
    fn pipeline(&self) -> Pipeline {
        let mut pm = Pipeline::new();
        ir_passes::populate_simplification_passes(&mut pm);
        ir_passes::populate_tensor_passes(&mut pm, &self.options.numerics);
        let target = || Some(self.target.clone());
        pm.add(TensorToTile)
            .add(PadWithIdentity)
            .add(InlineStaticMaps)
            .add(Canonicalize::default())
            .add(WidenReusedMaps)
            .add(PlanLayouts)
            .add(Canonicalize::default())
            .add(WidenNarrowFloats)
            .add(ExpandMath {
                numerics: self.options.numerics,
            })
            .add(ExpandRandom);
        if self.options.numerics.winograd {
            pm.add(WinogradConv {
                numerics: self.options.numerics,
            })
            .add(Canonicalize::default());
        }
        pm.add(FormGroups)
            .add(SealDispatches)
            .add(NameTarget {
                target: target(),
                numerics: self.options.numerics,
            })
            .add(SelectAtoms { target: target() })
            .add(SplitLaneFolds { target: target() })
            .add(Schedule { target: target() })
            .add(PlaceEpilogues)
            .add(WalkOrderLayouts);
        ir_passes::populate_load_time_passes(&mut pm);
        ir_passes::populate_isa_lowering_passes(&mut pm, target());
        pm.add(PlanMemory { target: target() });
        pm
    }

    fn params(&self) -> &ParamStore<HostTensor> {
        &self.params
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Function> {
        let inst = module
            .func(func)
            .ok_or_else(|| anyhow!("the module has no function @{func}"))?
            .inst();
        let args = emit::arg_types(module, inst)?;
        let (source, plan) = {
            let _scope = profiling::compile_scope("x86_backend.emit");
            emit::program(module, func, &self.target)?
        };
        let map = serde_json::to_string_pretty(&source.map)?;
        gpt_rs::ops::trace::artifact(&format!("x86-{func}.c"), &source.text);
        gpt_rs::ops::trace::artifact(&format!("x86-{func}.map.json"), &map);
        let library = {
            let _scope = profiling::compile_scope("x86_backend.compile");
            compile::library(&source.text, &source.map, &map, &self.target)?
        };
        let step_symbol = format!("{}\0", emit::STEP_FUNCTION);
        // SAFETY: the library defines its step with this signature.
        let step = unsafe { *library.get::<StepFn>(step_symbol.as_bytes())? };
        Ok(Function {
            _library: library,
            step,
            args,
            plan,
            arenas: Mutex::default(),
        })
    }

    fn run(&self, exe: &Function, args: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        exe.check(&args)?;
        let plan = &exe.plan;
        let mut arg_slots: Vec<Option<HostTensor>> = args.into_iter().map(Some).collect();
        let mut held: Vec<Option<HostTensor>> = Vec::with_capacity(plan.memory.len());
        let mut addresses: Vec<*mut c_void> = Vec::with_capacity(plan.memory.len());
        for memory in &plan.memory {
            let (address, tensor) = match memory {
                Memory::Read(index) => {
                    let arg = arg_slots[*index]
                        .as_ref()
                        .ok_or_else(|| anyhow!("argument {index} is taken"))?;
                    (arg.to_le_bytes().as_ptr().cast_mut(), None)
                }
                Memory::Taken(index) => {
                    let mut tensor = arg_slots[*index]
                        .take()
                        .ok_or_else(|| anyhow!("argument {index} is taken twice"))?;
                    (tensor.bytes_mut().as_mut_ptr(), Some(tensor))
                }
                Memory::Alloc { ty, alignment } => {
                    let mut tensor = HostTensor::zeros(ty)?;
                    let address = tensor.bytes_mut().as_mut_ptr();
                    ensure!(
                        address.align_offset(*alignment) == 0,
                        "the memory of a new {ty} starts at {address:p}, not at a multiple of the \
                         {alignment} bytes of its host.alloc"
                    );
                    (address, Some(tensor))
                }
                Memory::Copy(ty) => {
                    let mut tensor = HostTensor::zeros(ty)?;
                    (tensor.bytes_mut().as_mut_ptr(), Some(tensor))
                }
                Memory::Constant(tensor) => (tensor.to_le_bytes().as_ptr().cast_mut(), None),
            };
            addresses.push(address.cast::<c_void>());
            held.push(tensor);
        }
        let mut arenas = exe.take_arenas()?;
        let mut arena_addresses = Vec::with_capacity(arenas.len());
        for (arena, planned) in arenas.iter_mut().zip(&plan.arenas) {
            let memory = arena.bytes_mut().as_mut_ptr();
            arena_addresses.push(memory.wrapping_add(memory.align_offset(planned.alignment)));
        }
        let mut marks = Marks {
            launches: &plan.launches,
            open: None,
        };
        let (marker, context): (Option<MarkFn>, *mut c_void) = match profiling::collecting() {
            true => (Some(mark), (&mut marks as *mut Marks<'_>).cast()),
            false => (None, std::ptr::null_mut()),
        };
        // SAFETY: each address is of memory of the type the step was compiled for: the arguments
        // have the function's types, and the rest was made for the plan. The memory outlives the
        // call, and the step writes only what it owns: taken arguments, allocations and arenas.
        unsafe {
            (exe.step)(
                addresses.as_ptr(),
                arena_addresses.as_ptr(),
                marker,
                context,
            )
        };
        let mut results = Vec::with_capacity(plan.results.len());
        for produced in &plan.results {
            let tensor = match *produced {
                Produced::Arg(index) => arg_slots[index].clone(),
                Produced::Memory(entry) => match &plan.memory[entry] {
                    Memory::Constant(tensor) => Some(tensor.clone()),
                    Memory::Read(index) => arg_slots[*index].clone(),
                    _ => held[entry].clone(),
                },
            };
            results.push(tensor.ok_or_else(|| anyhow!("a result of the function has no memory"))?);
        }
        exe.arenas.lock().expect("arenas poisoned").push(arenas);
        Ok(results)
    }

    fn upload(&self, host: HostTensor) -> Result<HostTensor> {
        Ok(host)
    }

    fn download(&self, tensor: &HostTensor) -> Result<HostTensor> {
        Ok(tensor.clone())
    }
}
