//! A graph gives a program run the handle of an input that nothing else holds, so the backend may
//! update its memory in place, and keeps every input that a tensor may still read.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs::ops::graph::context::push_default_arena;
use gpt_rs::ops::graph::{GraphArena, Traced};
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::recording_backend::RecordingBackend;

type Backend = RecordingBackend<CpuPortableBackend>;

fn recording() -> Arc<Backend> {
    Arc::new(RecordingBackend::new(CpuPortableBackend::new()))
}

fn input(backend: &Arc<Backend>) -> Result<DeviceTensor<Backend>> {
    DeviceTensor::from_host(
        Arc::clone(backend),
        HostTensor::f32([4], vec![1.0, 2.0, 3.0, 4.0])?,
    )
}

#[test]
fn a_run_takes_an_input_that_nothing_else_holds() -> Result<()> {
    let backend = recording();
    let state = input(&backend)?;
    let next = &state + 1.0;
    drop(state);
    assert_eq!(next.to_host()?.as_f32()?, [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(backend.runs(), [[true]]);
    Ok(())
}

#[test]
fn a_run_shares_an_input_that_a_tensor_still_holds() -> Result<()> {
    let backend = recording();
    let state = input(&backend)?;
    let next = &state + 1.0;
    assert_eq!(next.to_host()?.as_f32()?, [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(backend.runs(), [[false]]);
    assert_eq!(state.to_host()?.as_f32()?, [1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn a_run_shares_an_input_that_a_pending_tensor_still_reads() -> Result<()> {
    let backend = recording();
    let _arena = push_default_arena(GraphArena::new(Arc::clone(&backend)));
    let state = input(&backend)?;
    let next = &state + 1.0;
    let doubled = &state * 2.0;
    drop(state);
    assert_eq!(next.to_host()?.as_f32()?, [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(doubled.to_host()?.as_f32()?, [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(backend.runs(), [[false], [true]]);
    Ok(())
}

#[test]
fn a_traced_run_takes_an_input_that_nothing_else_holds() -> Result<()> {
    let backend = recording();
    let step = Traced::new(Arc::clone(&backend), |inputs| vec![&inputs[0] + 1.0]);
    let mut state = input(&backend)?;
    for _ in 0..2 {
        let [next] =
            <[DeviceTensor<Backend>; 1]>::try_from(step.call(vec![state])?).expect("one result");
        state = next;
    }
    assert_eq!(state.to_host()?.as_f32()?, [3.0, 4.0, 5.0, 6.0]);
    assert_eq!(backend.runs(), [[true], [true]]);
    Ok(())
}
