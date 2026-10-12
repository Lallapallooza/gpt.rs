//! The checks a run of a `loop.func` makes, which make its results independent of how its units
//! run: units of one `loop.forall` that race on an element, and reads of elements that no store
//! wrote.

use gpt_rs_ir::host::{HostTensor, HostValue};
use gpt_rs_ir::interp::run;
use gpt_rs_ir::module::Module;

/// The error of running `@main` of `text` on a tensor of two f32 elements, or its results.
fn launch(text: &str) -> Result<Vec<HostValue>, String> {
    let module = Module::from_text(text).expect("the module parses");
    let input = HostValue::Tensor(HostTensor::f32([2], vec![1.0, 2.0]).unwrap());
    run(&module, "main", vec![input]).map_err(|err| format!("{err:#}"))
}

/// A `loop.func` `@launch` with `body`, called by `@main` on a tensor of two f32 elements, which
/// gives two.
fn module(body: &str) -> String {
    format!(
        "loop.func @launch(%input: !loop.buf<2xf32, global>, %output: !loop.buf<2xf32, global>) attributes {{access = [in, out]}} {{
  %zero = loop.index() {{expr = 0}} : () -> index
  %one = loop.index() {{expr = 1}} : () -> index
  %two = loop.index() {{expr = 2}} : () -> index
{body}
  loop.return() : () -> ()
}}
func.func @main(%input: tensor<2xf32>) -> tensor<2xf32> {{
  %output = host.dispatch(%input) {{callee = @launch}} : (tensor<2xf32>) -> tensor<2xf32>
  func.return(%output) : (tensor<2xf32>) -> ()
}}"
    )
}

#[test]
fn two_units_that_write_one_element_race() {
    let text = module(
        "  loop.forall(%zero, %two) {level = core} ({
  ^body(%unit: index):
    %element = loop.load(%input, %unit) : (!loop.buf<2xf32, global>, index) -> f32
    loop.store(%element, %output, %zero) : (f32, !loop.buf<2xf32, global>, index) -> ()
    loop.store(%element, %output, %unit) : (f32, !loop.buf<2xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()",
    );
    assert_eq!(
        launch(&text).unwrap_err(),
        "host.dispatch: loop.forall: loop.store: units [0] and [1] of a loop.forall race on \
         element [0] of !loop.buf<2xf32, global>"
    );
}

#[test]
fn a_unit_that_reads_what_another_writes_races() {
    let text = module(
        "  loop.forall(%zero, %two) {level = core} ({
  ^body(%unit: index):
    %element = loop.load(%input, %unit) : (!loop.buf<2xf32, global>, index) -> f32
    loop.store(%element, %output, %unit) : (f32, !loop.buf<2xf32, global>, index) -> ()
    %first = loop.load(%output, %zero) : (!loop.buf<2xf32, global>, index) -> f32
    loop.yield() : () -> ()
  }) : (index, index) -> ()",
    );
    assert_eq!(
        launch(&text).unwrap_err(),
        "host.dispatch: loop.forall: loop.load: units [0] and [1] of a loop.forall race on \
         element [0] of !loop.buf<2xf32, global>"
    );
}

#[test]
fn units_of_two_foralls_run_one_after_the_other() {
    // Each unit of the second forall reads what the other unit of the first one wrote.
    let text = module(
        "  loop.forall(%zero, %two) {level = core} ({
  ^body(%unit: index):
    %element = loop.load(%input, %unit) : (!loop.buf<2xf32, global>, index) -> f32
    loop.store(%element, %output, %unit) : (f32, !loop.buf<2xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.forall(%zero, %two) {level = core} ({
  ^body(%unit: index):
    %other = loop.index(%unit) {expr = 1 - d0} : (index) -> index
    %written = loop.load(%output, %other) : (!loop.buf<2xf32, global>, index) -> f32
    %kept = loop.alloc() {shape = [], dtype = f32} : () -> !loop.buf<f32, private>
    loop.store(%written, %kept) : (f32, !loop.buf<f32, private>) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()",
    );
    let results = launch(&text).unwrap();
    assert_eq!(
        results[0].as_tensor().unwrap().as_f32().unwrap(),
        [1.0, 2.0]
    );
}

#[test]
fn a_read_of_an_element_no_store_wrote_fails() {
    let text = module(
        "  %scratch = loop.alloc() {shape = [2], dtype = f32} : () -> !loop.buf<2xf32, private>
  %element = loop.load(%input, %zero) : (!loop.buf<2xf32, global>, index) -> f32
  loop.store(%element, %scratch, %zero) : (f32, !loop.buf<2xf32, private>, index) -> ()
  %unwritten = loop.load(%scratch, %one) : (!loop.buf<2xf32, private>, index) -> f32",
    );
    assert_eq!(
        launch(&text).unwrap_err(),
        "host.dispatch: loop.load: no store wrote element [1] of !loop.buf<2xf32, private>"
    );
}

#[test]
fn an_out_element_no_store_wrote_fails() {
    let text = module(
        "  %element = loop.load(%input, %zero) : (!loop.buf<2xf32, global>, index) -> f32
  loop.store(%element, %output, %zero) : (f32, !loop.buf<2xf32, global>, index) -> ()",
    );
    assert_eq!(
        launch(&text).unwrap_err(),
        "host.dispatch: @launch result 0: no store wrote element 1 of !loop.buf<2xf32, global>"
    );
}
