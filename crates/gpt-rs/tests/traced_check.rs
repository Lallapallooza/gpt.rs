//! With `GPTRS_TRACE_CHECK` set, a traced function records every call and fails when its graph is
//! not the one its trace replays. It is the only test in its binary, so the binary reads the
//! setting once, after the test sets it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::ops::graph::Traced;
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

type Tensor = DeviceTensor<CpuPortableBackend>;

#[test]
fn the_check_fails_a_function_whose_graph_holds_a_value_of_each_call() -> Result<()> {
    std::env::set_var("GPTRS_TRACE_CHECK", "1");
    let backend = Arc::new(CpuPortableBackend::new());
    // Adds the number of the call, a constant of its graph.
    let calls = AtomicUsize::new(0);
    let traced = Traced::new(Arc::clone(&backend), move |inputs: &[Tensor]| {
        let call = calls.fetch_add(1, Ordering::Relaxed) + 1;
        vec![&inputs[0] + call as f32]
    });
    let input = || {
        let host = HostTensor::f32([3], vec![1.0, 2.0, 3.0]).unwrap();
        DeviceTensor::from_host(Arc::clone(&backend), host).unwrap()
    };
    let results = traced.call(vec![input()])?;
    assert_eq!(results[0].to_host()?.as_f32()?, [2.0, 3.0, 4.0]);

    let err = traced
        .call(vec![input()])
        .expect_err("the second call records another graph");
    assert_eq!(
        err.to_string(),
        "a traced function recorded a graph other than the one its trace replays: its graph \
         depends on more than the dtypes and extents of its inputs"
    );
    Ok(())
}
