mod hint_lowering;
mod validate;

use gpt_rs::backend::conversion::{BufferPlan, ConversionError, ConversionResult};
use gpt_rs::backend::spec::{Function, Program};

use crate::artifact::TritonArtifact;
use crate::kernels::{builtin_kernel_sources, builtin_kernel_specs};

pub fn lower_program_to_artifact(
    program: &Program,
    entrypoint_symbol: &str,
    buffer_plan: BufferPlan,
) -> ConversionResult<TritonArtifact> {
    // Touch all built-in assets so they are included and validated by the
    // compiler even before every kernel family is hooked into runtime dispatch.
    let _ = builtin_kernel_sources();

    let lowered_program = hint_lowering::lower_hint_regions_to_custom_calls(program)?;
    let function = entry_function(&lowered_program)?;
    for instruction in &function.body {
        validate::validate_instruction(function, instruction)?;
    }

    let kernels = builtin_kernel_specs();

    Ok(TritonArtifact::new(
        entrypoint_symbol.to_string(),
        lowered_program,
        buffer_plan,
        kernels,
    ))
}

fn entry_function(program: &Program) -> ConversionResult<&Function> {
    program
        .functions
        .iter()
        .find(|function| function.name == program.entry)
        .ok_or_else(|| ConversionError::new("entry function not found"))
}
