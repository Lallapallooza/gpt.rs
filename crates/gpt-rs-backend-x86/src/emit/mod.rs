//! C for a function of a lowered module: its step function, which [`step`] writes, and its
//! dispatches. Each `loop.func` becomes one function, and each `loop.forall` an OpenMP parallel
//! loop on the units of its execution level: one thread per iteration, at most one per unit. Where
//! the target lists the CPUs of each unit, each worker thread pins itself to its unit's, and the
//! thread that runs the step runs the first iteration where it is. A `loop.forall` inside another
//! one is an error. Scalar code is plain C, and `isa` code AVX-512 intrinsics, one statement per
//! register of a group.
//!
//! Each line of C that an op writes ends with its op's number among the program's ops, `// #812`.
//! The program's [`LineMap`] gives the op's number in the dump, its location and its trail.

mod scalar;
mod step;
mod vector;

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, ensure, Context, Result};
use gpt_rs_ir::dialects::loops::{Affine, AffineForms};
use gpt_rs_ir::dialects::{host, isa, loops, x86, Dialect};
use gpt_rs_ir::ir::{BufferType, DType, Dim, Inst, Level, Module, Region, Type, Value};
use gpt_rs_ir::targets::TargetDesc;
use gpt_rs_ir::traits::Terminator;
use serde::{Deserialize, Serialize};

pub(crate) use self::step::{arg_types, Memory, Produced, StepPlan};

/// What every program starts with: the intrinsics, and the C types of buffer elements, which may
/// sit at any byte address, since weights are read in place from mapped checkpoints. Its only
/// functions are static inline helpers: the bf16 and e4m3 conversions, and `gptrs_pin`.
const PREAMBLE: &str = "#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>

typedef float gptrs_f32 __attribute__((aligned(1)));
typedef double gptrs_f64 __attribute__((aligned(1)));
typedef int8_t gptrs_si8 __attribute__((aligned(1)));
typedef int16_t gptrs_si16 __attribute__((aligned(1)));
typedef int32_t gptrs_si32 __attribute__((aligned(1)));
typedef int64_t gptrs_si64 __attribute__((aligned(1)));
typedef uint8_t gptrs_ui8 __attribute__((aligned(1)));
typedef uint16_t gptrs_ui16 __attribute__((aligned(1)));
typedef uint16_t gptrs_bf16 __attribute__((aligned(1)));
typedef uint16_t gptrs_f16 __attribute__((aligned(1)));
typedef uint8_t gptrs_f8e4m3;
typedef uint8_t gptrs_f8e5m2;
typedef uint32_t gptrs_ui32 __attribute__((aligned(1)));
typedef uint64_t gptrs_ui64 __attribute__((aligned(1)));
typedef uint8_t gptrs_i1;

/* `value` rounded to the nearest bf16, ties to even, as `tensor.cast` rounds; a NaN stays a NaN. */
static inline uint16_t gptrs_bf16_of_f32(float value) {
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    if (value != value) {
        return (uint16_t)((bits >> 16) | 0x40);
    }
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

/* The value of the e4m3 code `code`, exactly: its fields moved to f32's bias when it is normal,
   its mantissa times 2^-9 when it is subnormal, and NaN at S.1111.111. */
static inline float gptrs_f32_of_f8e4m3(uint8_t code) {
    uint32_t magnitude = code & 0x7f;
    uint32_t sign = (uint32_t)(code & 0x80) << 24;
    if (magnitude == 0x7f) {
        return __builtin_bit_cast(float, 0x7fc00000u);
    }
    if (magnitude < 0x08) {
        return __builtin_bit_cast(float, sign | __builtin_bit_cast(uint32_t, (float)magnitude / 512.0f));
    }
    return __builtin_bit_cast(float, sign | ((magnitude << 20) + (120u << 23)));
}

/* The values of the 16 e4m3 codes of `codes`, as `gptrs_f32_of_f8e4m3` gives each. */
static inline __m512 gptrs_f32x16_of_f8e4m3(__m128i codes) {
    __m512i bits = _mm512_cvtepu8_epi32(codes);
    __m512i magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7f));
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(bits, _mm512_set1_epi32(0x80)), 24);
    __m512i normal = _mm512_add_epi32(_mm512_slli_epi32(magnitude, 20), _mm512_set1_epi32(120 << 23));
    __m512i subnormal = _mm512_castps_si512(
        _mm512_mul_ps(_mm512_cvtepi32_ps(magnitude), _mm512_set1_ps(1.0f / 512.0f)));
    __mmask16 small = _mm512_cmplt_epu32_mask(magnitude, _mm512_set1_epi32(0x08));
    __m512i value = _mm512_or_si512(sign, _mm512_mask_blend_epi32(small, normal, subnormal));
    __mmask16 nan = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x7f));
    return _mm512_mask_mov_ps(_mm512_castsi512_ps(value), nan, _mm512_set1_ps(__builtin_bit_cast(float, 0x7fc00000u)));
}

/* What a step calls before (`begin` 1) and after (`begin` 0) its launch `launch`, to profile. */
typedef void (*gptrs_mark)(void *context, int64_t launch, int32_t begin);

/* The CPUs that `gptrs_pin` last pinned the calling thread to. */
static __thread const int *gptrs_pinned;

/* Pins the calling thread to the CPUs `cpus`, a list that ends at -1, unless it is pinned to
   them: a worker thread of a parallel loop, which stays on its unit for the next loop. */
static inline void gptrs_pin(const int *cpus) {
    if (gptrs_pinned == cpus) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int *cpu = cpus; *cpu >= 0; ++cpu) {
        CPU_SET(*cpu, &set);
    }
    sched_setaffinity(0, sizeof set, &set);
    gptrs_pinned = cpus;
}
";

/// The C tables of the CPUs of the units of each execution level of `target` that lists them:
/// `gptrs_cpus_<level>`, one list of CPUs per unit, each ending at -1.
fn unit_cpus(target: &TargetDesc) -> Result<String> {
    let mut tables = String::new();
    for level in &target.levels {
        let units = level.unit_cpus()?;
        let Some(widest) = units.iter().map(Vec::len).max() else {
            continue;
        };
        let mut lists = Vec::with_capacity(units.len());
        for cpus in &units {
            let mut entries = Vec::with_capacity(cpus.len() + 1);
            for cpu in cpus {
                entries.push(cpu.to_string());
            }
            entries.push("-1".to_string());
            lists.push(format!("{{{}}}", entries.join(", ")));
        }
        tables.push_str(&format!(
            "\n/* The CPUs of each unit of level `{}`. */\nstatic const int gptrs_cpus_{}[{}][{}] = {{{}}};\n",
            level.name,
            symbol(&level.name),
            units.len(),
            widest + 1,
            lists.join(", ")
        ));
    }
    Ok(tables)
}

/// The C symbol stem of the dispatch `name`.
pub(crate) fn symbol(name: &str) -> String {
    name.replace(|character: char| !character.is_ascii_alphanumeric(), "_")
}

/// The name of the step function of every program.
pub(crate) const STEP_FUNCTION: &str = "gptrs_step";

/// The C of a program, and where each of its lines comes from.
pub(crate) struct Source {
    pub text: String,
    pub map: LineMap,
}

/// The C of the function `func` of `module`, in the buffer form, for `target`, with what the step
/// takes. The C holds:
/// - the CPUs of the units of each execution level of the target that lists them;
/// - the step function, `gptrs_step`;
/// - for each dispatch the step launches, `gptrs_run_<stem>`, which runs it on the buffers of its
///   parameters, in order.
///
/// The text depends on the code alone, so that two functions that compute alike, such as one
/// computation over the parameters of each layer, are one program. The step is named for no
/// function, and each line's comment numbers its op among the program's ops in the order they
/// first come, while the map keeps its place in the module.
pub(crate) fn program(
    module: &Module,
    func: &str,
    target: &TargetDesc,
) -> Result<(Source, StepPlan)> {
    let inst = module
        .func(func)
        .ok_or_else(|| anyhow!("the module has no function @{func}"))?
        .inst();
    let mut names = Vec::new();
    for op in module.insts(module.regions(inst)[0]) {
        if let Some(launch) = module.op(op).get::<host::Launch>() {
            if !names.contains(&launch.callee.0) {
                names.push(launch.callee.0.clone());
            }
        }
    }
    let numbers = module.op_numbers();
    let mut program_numbers: HashMap<Inst, u32> = HashMap::new();
    let mut text = String::from(PREAMBLE);
    text.push_str(&unit_cpus(target)?);
    let mut map = LineMap::default();
    let mut line = text.lines().count() as u32;
    let mut functions = Vec::new();
    for name in &names {
        let dispatch = module
            .symbol(name)
            .ok_or_else(|| anyhow!("the module has no dispatch @{name}"))?;
        ensure!(
            module.op(dispatch).get::<loops::Func>().is_some(),
            "@{name} is not a loop.func"
        );
        let stem = symbol(name);
        let function = format!("gptrs_run_{stem}");
        let lines = Printer::new(module, dispatch, target)?
            .function(&function)
            .with_context(|| format!("writing C for @{name}"))?;
        functions.push((function, lines));
    }
    let step_function = STEP_FUNCTION.to_string();
    let (step_lines, plan) = step::step(module, inst, &step_function)?;
    functions.push((step_function, step_lines));
    for (function, lines) in functions {
        for (statement, statement_origin) in lines {
            line += 1;
            text.push_str(&statement);
            let Some(origin) = statement_origin else {
                text.push('\n');
                continue;
            };
            let next = program_numbers.len() as u32;
            let program_number = *program_numbers.entry(origin).or_insert(next);
            text.push_str(&format!(" // #{program_number}\n"));
            map.lines.push(LineOrigin::new(
                module,
                origin,
                numbers[&origin],
                line,
                &function,
            ));
        }
        line += 1;
        text.push('\n');
    }
    Ok((Source { text, map }, plan))
}

/// A `reg` buffer held as vector variables: every access is a whole register at a constant
/// position.
#[derive(Debug, Clone, Copy)]
struct Registers {
    lanes: usize,
}

/// The C of one `loop.func`, written statement by statement, each line with the op it comes
/// from.
struct Printer<'m> {
    module: &'m Module,
    func: Inst,
    target: &'m TargetDesc,
    out: Vec<(String, Option<Inst>)>,
    /// The op whose C the printer writes.
    current: Option<Inst>,
    depth: usize,
    names: HashMap<Value, String>,
    /// How many register offsets the C names so far.
    offsets: usize,
    /// The affine form of each index of the function.
    affine: HashMap<Value, Affine>,
    forms: AffineForms<'m>,
    registers: HashMap<Value, Registers>,
    /// Whether the statements written now run inside a parallel loop.
    in_parallel: bool,
}

impl<'m> Printer<'m> {
    fn new(module: &'m Module, func: Inst, target: &'m TargetDesc) -> Result<Self> {
        let mut printer = Self {
            module,
            func,
            target,
            out: Vec::new(),
            current: Some(func),
            depth: 1,
            names: HashMap::new(),
            offsets: 0,
            affine: HashMap::new(),
            forms: AffineForms::new(module),
            registers: HashMap::new(),
            in_parallel: false,
        };
        printer.analyze(printer.body())?;
        Ok(printer)
    }

    fn body(&self) -> Region {
        self.module.regions(self.func)[0]
    }

    /// `function`: the dispatch on the buffers of its parameters. It stays a function of its own:
    /// inlined into the step that calls it, its registers would be allocated with the step's.
    fn function(mut self, function: &str) -> Result<Vec<(String, Option<Inst>)>> {
        self.header(format!(
            "__attribute__((noinline)) void {function}(void *const *buffers) {{"
        ));
        self.params()?;
        self.region(self.body())?;
        self.header("}".to_string());
        Ok(self.out)
    }

    /// The parameters, from the array of buffer addresses.
    fn params(&mut self) -> Result<()> {
        let params = self.module.args(self.body()).to_vec();
        for (position, param) in params.into_iter().enumerate() {
            let element = element_type(self.buffer_type(param)?.dtype)?;
            let name = self.define(param);
            self.line(format!(
                "{element} *{name} = ({element} *)buffers[{position}];"
            ));
        }
        Ok(())
    }

    /// The statements of `region`, up to its terminator. The code of each register tile is a run,
    /// which [`Printer::run`] writes.
    fn region(&mut self, region: Region) -> Result<()> {
        let runs = isa::vector_runs(self.module, region);
        let mut starts: HashMap<Inst, usize> = HashMap::new();
        let mut in_runs: HashSet<Inst> = HashSet::new();
        for (index, run) in runs.iter().enumerate() {
            starts.insert(run[0], index);
            in_runs.extend(run.iter().copied());
        }
        for inst in self.module.insts(region) {
            if self.module.op(inst).has_trait::<Terminator>() {
                break;
            }
            if let Some(&index) = starts.get(&inst) {
                self.run(&runs[index])?;
                continue;
            }
            if in_runs.contains(&inst) {
                continue;
            }
            self.inst(inst)
                .with_context(|| self.module.op(inst).name().into_owned())?;
        }
        Ok(())
    }

    /// The C of `run`, a run of [`isa::vector_runs`]: its scalar ops, then the group of each `isa`
    /// op, declared, then their registers, one statement each, in the order of the register model,
    /// [`isa::RegisterCode::order`], so that the C holds at once the registers that it counts.
    /// Each register that a read loads comes after an empty volatile `asm`, which the C compiler
    /// moves no memory access across: it starts no load ahead of the registers before it in the
    /// order, which would hold the loaded register while those are still live.
    fn run(&mut self, run: &[Inst]) -> Result<()> {
        let mut vector_insts = Vec::new();
        for &inst in run {
            if self.module.op(inst).dialect() == Dialect::Isa {
                vector_insts.push(inst);
                continue;
            }
            self.inst(inst)
                .with_context(|| self.module.op(inst).name().into_owned())?;
        }
        let mut statements: HashMap<Inst, Vec<String>> = HashMap::new();
        for &inst in &vector_insts {
            let outer = self.current.replace(inst);
            let vector_code = self.vector(inst);
            self.current = outer;
            let written = vector_code.with_context(|| self.module.op(inst).name().into_owned())?;
            statements.insert(inst, written);
        }
        let rules = self.target.isa_rules().ok_or_else(|| {
            anyhow!(
                "{} computes no vector code: gpt.rs has no rules for its instruction set",
                self.target.name
            )
        })?;
        let code = isa::RegisterCode::of_isa(self.module, &vector_insts, rules)?;
        for register in code.order() {
            let inst = vector_insts[register.op];
            let statement = statements[&inst][register.position].clone();
            let outer = self.current.replace(inst);
            let reads = isa::Access::of(self.module, inst).is_some()
                && !self.module.results(inst).is_empty();
            if reads {
                self.line("__asm__ volatile(\"\");".to_string());
            }
            self.line(statement);
            self.current = outer;
        }
        Ok(())
    }

    fn inst(&mut self, inst: Inst) -> Result<()> {
        let outer = self.current.replace(inst);
        let written = self.op(inst);
        self.current = outer;
        written
    }

    /// The C of the op of `inst`.
    fn op(&mut self, inst: Inst) -> Result<()> {
        let op = self.module.op(inst);
        if op.get::<loops::For>().is_some() || op.get::<loops::Forall>().is_some() {
            return self.for_loop(inst);
        }
        if op.get::<loops::If>().is_some() {
            let condition = self.name(self.module.operands(inst)[0])?.to_string();
            self.line(format!("if ({condition}) {{"));
            self.depth += 1;
            self.region(self.module.regions(inst)[0])?;
            self.depth -= 1;
            self.line("}".to_string());
            return Ok(());
        }
        if let Some(alloc) = op.get::<loops::Alloc>() {
            return self.alloc(inst, alloc);
        }
        if op.get::<loops::Pick>().is_some() {
            return self.pick(inst);
        }
        if op.get::<loops::Load>().is_some() || op.get::<loops::Store>().is_some() {
            return self.scalar_memory(inst);
        }
        if let Some(prefetch) = op.get::<loops::Prefetch>() {
            let operands = self.module.operands(inst);
            let (buffer, indices) = operands.split_first().expect("a prefetch reads a buffer");
            let address = self.element(*buffer, indices)?;
            let hint = self.prefetch_hint(&prefetch.level)?;
            self.line(format!("_mm_prefetch((const char *)({address}), {hint});"));
            return Ok(());
        }
        if op.get::<loops::Index>().is_some() || op.get::<loops::Predicate>().is_some() {
            return self.index(inst);
        }
        if op.dialect() == Dialect::Isa {
            return self.run(&[inst]);
        }
        self.scalar(inst)
    }

    /// `loop.for`, or a `loop.forall`, whose iterations its level's units share, one thread each,
    /// each worker thread pinned to the CPUs of its unit where the target lists them. Every
    /// parallel loop of a level runs on as many threads, so the threads of one stay for the next:
    /// those without an iteration wait at its end.
    fn for_loop(&mut self, inst: Inst) -> Result<()> {
        let operands = self.module.operands(inst);
        let (start, end) = (
            self.name(operands[0])?.to_string(),
            self.name(operands[1])?.to_string(),
        );
        let region = self.module.regions(inst)[0];
        let index = self.define(self.module.args(region)[0]);
        let Some(forall) = self.module.op(inst).get::<loops::Forall>() else {
            // The schedule unrolls what runs at once into the register tile's code; the C compiler
            // unrolls no loop of its own, which would hold more in registers than the tile planned.
            self.line("#pragma clang loop unroll(disable)".to_string());
            self.c_loop([&index, &start, &end], region, None)?;
            // What follows a loop runs after it: the C compiler moves no memory access across its
            // end, so the work the schedule runs one part after another holds no registers across
            // the parts.
            self.line("__asm__ volatile(\"\" : : : \"memory\");".to_string());
            return Ok(());
        };
        ensure!(
            !self.in_parallel,
            "a loop.forall runs in the iterations of another one, each of which the x86 backend \
             runs on one thread"
        );
        let level = self.target.level(&forall.level).ok_or_else(|| {
            anyhow!(
                "a loop.forall runs on level {}, which {} does not have",
                forall.level,
                self.target.name
            )
        })?;
        let units = level.units;
        let pinned = (!level.cpus.is_empty()).then(|| symbol(&level.name));
        self.line(format!(
            "#pragma omp parallel for schedule(static) num_threads({units})"
        ));
        self.in_parallel = true;
        // The thread that runs the step runs the first iteration where it is: pinning it would
        // move it, and the work around the loop with it.
        let pin = pinned.as_ref().map(|table| {
            format!(
                "if (omp_get_thread_num() > 0) gptrs_pin(gptrs_cpus_{table}[omp_get_thread_num()]);"
            )
        });
        self.c_loop([&index, &start, &end], region, pin)?;
        self.in_parallel = false;
        Ok(())
    }

    /// The hint of `_mm_prefetch` that asks the memory `level` of the target for a line.
    fn prefetch_hint(&self, level: &Level) -> Result<&'static str> {
        let hint = self
            .target
            .memory(level)
            .and_then(|memory| memory.prefetch)
            .ok_or_else(|| {
                anyhow!(
                    "a prefetch asks {level} for lines, which no prefetch of {} asks",
                    self.target.name
                )
            })?;
        Ok(x86::cache_hint(hint))
    }

    /// A C `for` loop of `index` from `start` up to `end`, whose body is `first`, when given, then
    /// `region`.
    fn c_loop(
        &mut self,
        [index, start, end]: [&str; 3],
        region: Region,
        first: Option<String>,
    ) -> Result<()> {
        self.line(format!(
            "for (int64_t {index} = {start}; {index} < {end}; ++{index}) {{"
        ));
        self.depth += 1;
        if let Some(statement) = first {
            self.line(statement);
        }
        self.region(region)?;
        self.depth -= 1;
        self.line("}".to_string());
        Ok(())
    }

    /// `loop.alloc`: a local array, or the vector variables of a `reg` buffer. An array with axes
    /// stays in memory: its address goes to an empty `asm` statement, so the C compiler does not
    /// keep its elements in registers, where they would compete with the register tile's. A cell,
    /// which has none, is a variable it may keep in a register.
    fn alloc(&mut self, inst: Inst, alloc: &loops::Alloc) -> Result<()> {
        let result = self.module.results(inst)[0];
        let count: usize = alloc.shape.iter().product();
        let name = self.define(result);
        if let Some(registers) = self.registers.get(&result) {
            let ty = x86::register_type(alloc.dtype, registers.lanes)?;
            let vectors = count / registers.lanes;
            self.line(format!("{ty} {name}[{vectors}] = {{0}};"));
        } else {
            ensure!(
                alloc.dtype.bits() >= 8,
                "the x86 backend writes no buffer of {} elements",
                alloc.dtype
            );
            let element = element_type(alloc.dtype)?;
            // Aligned to the widest vector, so that the vectors at its multiples are aligned.
            let alignment = self.target.max_vector_bits() / 8;
            self.line(format!(
                "{element} {name}[{count}] __attribute__((aligned({alignment})));"
            ));
            if !alloc.shape.is_empty() {
                self.line(format!("__asm__(\"\" : : \"r\"({name}));"));
            }
        }
        Ok(())
    }

    /// `loop.pick`: the pointer to the buffer at the picked position. Buffers that are
    /// consecutive parameters are read from the array of buffer addresses, which is their table,
    /// so the code holds none of them in a register; others are chosen one comparison at a time.
    fn pick(&mut self, inst: Inst) -> Result<()> {
        let operands = self.module.operands(inst);
        let (index_value, buffers) = operands.split_first().expect("a pick has an index");
        let index = self.name(*index_value)?.to_string();
        let result = self.module.results(inst)[0];
        let element = element_type(self.buffer_type(result)?.dtype)?;
        let params = self.module.args(self.body());
        let first_param = params.iter().position(|&param| param == buffers[0]);
        let consecutive = first_param.is_some_and(|first| params[first..].starts_with(buffers));
        let chosen = match (first_param, consecutive) {
            (Some(first), true) => format!("({element} *)buffers[{first} + {index}]"),
            _ => {
                let (last, earlier) = buffers.split_last().expect("a pick has a buffer");
                let mut chosen = self.name(*last)?.to_string();
                for (position, buffer) in earlier.iter().enumerate().rev() {
                    let name = self.name(*buffer)?;
                    chosen = format!("{index} == {position} ? {name} : {chosen}");
                }
                chosen
            }
        };
        let name = self.define(result);
        self.line(format!("{element} *{name} = {chosen};"));
        Ok(())
    }

    /// `loop.load` and `loop.store` of one element.
    fn scalar_memory(&mut self, inst: Inst) -> Result<()> {
        let op = self.module.op(inst);
        let operands = self.module.operands(inst);
        if let Some(load) = op.get::<loops::Load>() {
            let (buffer, indices) = operands.split_first().expect("a load reads a buffer");
            let ty = self.buffer_type(*buffer)?.clone();
            let element = scalar::c_type(ty.dtype)?;
            let read = self.element_value(*buffer, indices)?;
            let result = self.define(self.module.results(inst)[0]);
            let value = match load.pad {
                None => read,
                Some(pad) => {
                    let inside = self.inside(&ty, indices)?;
                    let pad_value = scalar::pad_literal(ty.dtype, pad.0)?;
                    format!("({inside}) ? {read} : {pad_value}")
                }
            };
            self.line(format!("{element} {result} = {value};"));
            return Ok(());
        }
        let (stored, rest) = operands.split_first().expect("a store writes a value");
        let (buffer, indices) = rest.split_first().expect("a store writes a buffer");
        let dtype = self.buffer_type(*buffer)?.dtype;
        ensure!(
            dtype.bits() >= 8,
            "the x86 backend stores no {dtype} element"
        );
        let address = self.element(*buffer, indices)?;
        let value = self.name(*stored)?.to_string();
        self.line(format!("*({address}) = {value};"));
        Ok(())
    }

    /// The value of the element of `buffer` at `indices`: of a 4-bit dtype, the half of its byte
    /// that its offset's parity picks, sign-extended for a signed one.
    fn element_value(&self, buffer: Value, indices: &[Value]) -> Result<String> {
        let dtype = self.buffer_type(buffer)?.dtype;
        if dtype.bits() >= 8 {
            return Ok(format!("*({})", self.element(buffer, indices)?));
        }
        let offset = self.element_offset(buffer, indices)?;
        let name = self.name(buffer)?;
        let nibble = format!("((({name})[({offset}) >> 1] >> ((({offset}) & 1) * 4)) & 15)");
        Ok(match dtype {
            DType::Ui4 => nibble,
            DType::Si4 => format!("((int8_t)({nibble} << 4) >> 4)"),
            other => bail!("the x86 backend reads no {other} element"),
        })
    }

    /// The address of the element of `buffer` at `indices`.
    fn element(&self, buffer: Value, indices: &[Value]) -> Result<String> {
        let offset = self.element_offset(buffer, indices)?;
        Ok(format!("{} + ({offset})", self.name(buffer)?))
    }

    /// The offset, in elements, of the element of `buffer` at `indices`.
    fn element_offset(&self, buffer: Value, indices: &[Value]) -> Result<String> {
        let ty = self.buffer_type(buffer)?;
        let strides = strides(ty)?;
        let mut terms = Vec::with_capacity(indices.len());
        for (&index, stride) in indices.iter().zip(strides) {
            let name = self.name(index)?;
            terms.push(match stride {
                1 => name.to_string(),
                _ => format!("{name} * {stride}"),
            });
        }
        Ok(match terms.is_empty() {
            true => "0".to_string(),
            false => terms.join(" + "),
        })
    }

    /// Whether `indices` lie inside a buffer of type `ty`.
    fn inside(&self, ty: &BufferType, indices: &[Value]) -> Result<String> {
        let mut checks = Vec::with_capacity(indices.len());
        for (&index, dim) in indices.iter().zip(&ty.dims) {
            let Dim::Static(extent) = dim else {
                bail!("the buffer {ty} has a dynamic extent");
            };
            let name = self.name(index)?;
            checks.push(format!("{name} >= 0 && {name} < {extent}"));
        }
        Ok(match checks.is_empty() {
            true => "1".to_string(),
            false => checks.join(" && "),
        })
    }

    /// Finds the affine form of every index, and the `reg` buffers held in registers.
    fn analyze(&mut self, region: Region) -> Result<()> {
        for inst in self.module.insts(region) {
            if self.module.op(inst).get::<loops::Index>().is_some() {
                let result = self.module.results(inst)[0];
                let form = self.forms.of(result);
                self.affine.insert(result, form);
            }
            for &inner in self.module.regions(inst) {
                self.analyze(inner)?;
            }
        }
        for inst in self.module.insts(region) {
            if self.module.op(inst).get::<loops::Alloc>().is_none() {
                continue;
            }
            let result = self.module.results(inst)[0];
            if !isa::in_registers(self.module, result) {
                continue;
            }
            if let Some(registers) = self.held_in_registers(result) {
                self.registers.insert(result, registers);
            }
        }
        Ok(())
    }

    fn affine_of(&self, value: Value) -> Affine {
        self.affine
            .get(&value)
            .cloned()
            .unwrap_or(Affine::opaque(value))
    }

    /// How the `reg` buffer `buffer` is held in registers: as the vector variables of the lanes of
    /// the registers that reach it ([`isa::reg_lanes`]), which `loop-to-isa` checks are whole
    /// registers at fixed positions; none when no register reaches it.
    fn held_in_registers(&self, buffer: Value) -> Option<Registers> {
        let lanes = isa::reg_lanes(self.module, buffer)?;
        Some(Registers { lanes })
    }

    fn buffer_type(&self, value: Value) -> Result<&BufferType> {
        match self.module.ty(value) {
            Type::Buffer(buffer) => Ok(buffer),
            other => bail!("{other} is not a buffer"),
        }
    }

    /// A new name for `value`.
    fn define(&mut self, value: Value) -> String {
        let name = format!("v{}", self.names.len());
        self.names.insert(value, name.clone());
        name
    }

    /// The literal `text` as the name of the constant `value`: the C writes a constant where it is
    /// used, so that a parallel region the compiler outlines reads it as a constant rather than as
    /// a variable it captures.
    fn define_constant(&mut self, value: Value, text: String) {
        self.names.insert(value, format!("({text})"));
    }

    fn name(&self, value: Value) -> Result<&str> {
        self.names
            .get(&value)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("a value is used before the C defines it"))
    }

    /// A statement of the current op, at the current depth.
    fn line(&mut self, text: String) {
        let indent = "    ".repeat(self.depth);
        self.out.push((format!("{indent}{text}"), self.current));
    }

    /// A line of the function itself, unindented.
    fn header(&mut self, text: String) {
        self.out.push((text, None));
    }
}

/// The row-major strides of a buffer of static extents.
fn strides(ty: &BufferType) -> Result<Vec<usize>> {
    let mut extents = Vec::with_capacity(ty.dims.len());
    for dim in &ty.dims {
        let Dim::Static(extent) = dim else {
            bail!("the buffer {ty} has a dynamic extent");
        };
        extents.push(*extent);
    }
    let mut strides = vec![1; extents.len()];
    for axis in (0..extents.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * extents[axis + 1];
    }
    Ok(strides)
}

/// The C type of a buffer element, at any byte address: for a 4-bit dtype, of the byte that holds
/// two elements, the lower one in its lower bits, at the element offset halved.
fn element_type(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::Si4 | DType::Ui4 => "gptrs_ui8",
        DType::F32 => "gptrs_f32",
        DType::F64 => "gptrs_f64",
        DType::Si8 => "gptrs_si8",
        DType::Si16 => "gptrs_si16",
        DType::Si32 => "gptrs_si32",
        DType::Si64 => "gptrs_si64",
        DType::Ui8 => "gptrs_ui8",
        DType::Ui16 => "gptrs_ui16",
        DType::Bf16 => "gptrs_bf16",
        DType::F16 => "gptrs_f16",
        DType::F8E4M3 => "gptrs_f8e4m3",
        DType::F8E5M2 => "gptrs_f8e5m2",
        DType::Ui32 => "gptrs_ui32",
        DType::Ui64 => "gptrs_ui64",
        DType::I1 => "gptrs_i1",
        other => bail!("the x86 backend has no C type for {other} elements"),
    })
}

/// Where each line of a program's C comes from: the lines that ops wrote.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LineMap {
    pub lines: Vec<LineOrigin>,
}

/// The op a line of C comes from: its name and number in the lowered module, as its dump prints
/// them, its location and its trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineOrigin {
    /// The line, from 1.
    pub line: u32,
    /// The C function the line is in.
    pub function: String,
    pub op: String,
    pub number: u32,
    /// The module path of the layer that recorded the op the line comes from.
    pub scope: Option<String>,
    /// The line of code that recorded it: `file:line:column`.
    pub source: Option<String>,
    /// The passes that made the op, the first first: `pass(op:number)`.
    pub trail: Vec<String>,
}

impl LineOrigin {
    fn new(module: &Module, inst: Inst, number: u32, line: u32, function: &str) -> Self {
        let recorded = module.location(inst);
        Self {
            line,
            function: function.to_string(),
            op: module.op(inst).name().into_owned(),
            number,
            scope: recorded.and_then(|location| location.scope.as_deref().map(str::to_string)),
            source: recorded.and_then(|location| {
                location
                    .source
                    .as_ref()
                    .map(|source| format!("{}:{}:{}", source.file, source.line, source.column))
            }),
            trail: recorded.map_or_else(Vec::new, |location| {
                location
                    .trail
                    .iter()
                    .map(|step| format!("{}({}:{})", step.pass, step.op, step.number))
                    .collect()
            }),
        }
    }
}
