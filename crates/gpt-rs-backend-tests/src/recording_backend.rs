use std::sync::Mutex;

use gpt_rs::backend::spec::{
    BackendResult, Function, Instruction, PortableBackend, Program, TensorInit, TensorLiteral,
};

/// Test-only portable backend that records the latest emitted PTIR program.
#[derive(Default)]
pub struct RecordingBackend {
    last_program: Mutex<Option<Program>>,
}

impl RecordingBackend {
    pub fn recorded_program(&self) -> Option<Program> {
        self.last_program
            .lock()
            .expect("backend mutex poisoned")
            .clone()
    }

    pub fn recorded_program_or_panic(&self) -> Program {
        self.recorded_program()
            .expect("backend should record emitted program")
    }

    pub fn recorded_entry_function_or_panic(&self) -> Function {
        let program = self.recorded_program_or_panic();
        program
            .functions
            .iter()
            .find(|function| function.name == program.entry)
            .expect("captured function present")
            .clone()
    }
}

impl PortableBackend for RecordingBackend {
    type TensorHandle = ();

    fn backend_name(&self) -> &str {
        "recording"
    }

    fn materialize(&self, _init: TensorInit) -> BackendResult<Self::TensorHandle> {
        Ok(())
    }

    fn to_literal(&self, _tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        unreachable!("recording backend does not materialize to host")
    }

    fn execute_instruction(
        &self,
        _instruction: &Instruction,
        _inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        unreachable!("recording backend does not execute standalone instructions")
    }

    fn run_program(
        &self,
        program: &Program,
        _entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let output_count = program
            .functions
            .iter()
            .find(|function| function.name == program.entry)
            .map(|function| function.results.len())
            .unwrap_or(0);
        self.last_program
            .lock()
            .expect("backend mutex poisoned")
            .replace(program.clone());
        Ok(vec![(); output_count])
    }
}
