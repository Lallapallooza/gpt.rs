//! The checks a run of the `host` dialect makes of its buffers. In the buffer form, a run fails
//! when it reads a value after another value wrote its bytes, and when a launch writes a buffer
//! that it gets for two parameters. In any call, a run fails when it reads an `in` parameter where
//! its `out<over = k>` parameter wrote already. A write into the elements of a tensor is a type
//! error: their buffer is `read_only`.

use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::interp::{run, HostValue};
use gpt_rs_ir::ir::{verify, Module};

/// `@double`, whose output may share the buffer of its input; `@accumulate`, which adds its second
/// parameter into its first in place; `@reverse`, which writes its input backwards into an output
/// of its own; and `@reverse_over`, which does so into an output that may share its input's
/// buffer, reading positions it wrote already: on buffers of four f32 elements.
const LAUNCHES: &str = "loop.func @double(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %element = loop.load(%input, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %doubled = arith.add(%element, %element) : (f32, f32) -> f32
    loop.store(%doubled, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @accumulate(%totals: !loop.buf<4xf32, global>, %addend: !loop.buf<4xf32, global>) attributes {access = [inout, in]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %term = loop.load(%addend, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %so_far = loop.load(%totals, %position) : (!loop.buf<4xf32, global>, index) -> f32
    %sum = arith.add(%so_far, %term) : (f32, f32) -> f32
    loop.store(%sum, %totals, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @reverse(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %mirrored = loop.index(%position) {expr = 3 - d0} : (index) -> index
    %element = loop.load(%input, %mirrored) : (!loop.buf<4xf32, global>, index) -> f32
    loop.store(%element, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}
loop.func @reverse_over(%input: !loop.buf<4xf32, global>, %output: !loop.buf<4xf32, global>) attributes {access = [in, out<over = 0>]} {
  %zero = loop.index() {expr = 0} : () -> index
  %count = loop.index() {expr = 4} : () -> index
  loop.for(%zero, %count) ({
  ^body(%position: index):
    %mirrored = loop.index(%position) {expr = 3 - d0} : (index) -> index
    %element = loop.load(%input, %mirrored) : (!loop.buf<4xf32, global>, index) -> f32
    loop.store(%element, %output, %position) : (f32, !loop.buf<4xf32, global>, index) -> ()
    loop.yield() : () -> ()
  }) : (index, index) -> ()
  loop.return() : () -> ()
}";

/// The result of running `@main`, whose body is `body`, on a tensor of four f32 elements, or the
/// error. The module verifies.
fn run_main(body: &str) -> Result<Vec<HostValue>, String> {
    let text = format!(
        "{LAUNCHES}
func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {{
{body}
}}"
    );
    let module = Module::from_text(&text).expect("the module parses");
    verify(&module).expect("the module verifies");
    let input = HostValue::Tensor(HostTensor::f32([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    run(&module, "main", vec![input]).map_err(|err| format!("{err:#}"))
}

#[test]
fn a_value_read_after_another_wrote_its_bytes_fails() {
    let error = run_main(
        "  %arena = host.arena() {bytes = 64, alignment = 8} : () -> !loop.buf<64xui8, arena<alignment = 8>>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %first = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<64xui8, arena<alignment = 8>>) -> !loop.buf<4xf32, global>
  %second = host.slice(%arena) {offset = 8, shape = [4], dtype = f32} : (!loop.buf<64xui8, arena<alignment = 8>>) -> !loop.buf<4xf32, global>
  %result = host.alloc() {shape = [4], dtype = f32, alignment = 8} : () -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %first) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.launch(%input_buffer, %second) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.launch(%first, %result) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%result) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output) : (tensor<4xf32>) -> ()",
    )
    .unwrap_err();
    assert_eq!(
        error,
        "host.launch: it reads !loop.buf<4xf32, global> at bytes 0..16 of an arena, which \
         another value wrote since"
    );
}

#[test]
fn a_launch_that_writes_its_input_bytes_through_another_slice_fails() {
    let error = run_main(
        "  %arena = host.arena() {bytes = 64, alignment = 8} : () -> !loop.buf<64xui8, arena<alignment = 8>>
  %input_buffer = host.to_buffer(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, read_only>
  %doubled = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<64xui8, arena<alignment = 8>>) -> !loop.buf<4xf32, global>
  %reversed = host.slice(%arena) {offset = 0, shape = [4], dtype = f32} : (!loop.buf<64xui8, arena<alignment = 8>>) -> !loop.buf<4xf32, global>
  host.launch(%input_buffer, %doubled) {callee = @double, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, read_only>, !loop.buf<4xf32, global>) -> ()
  host.launch(%doubled, %reversed) {callee = @reverse, operand_segments = [1, 1, 0]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%reversed) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output) : (tensor<4xf32>) -> ()",
    )
    .unwrap_err();
    assert_eq!(
        error,
        "host.launch: it reads !loop.buf<4xf32, global> at bytes 0..16 of an arena, which \
         another value wrote since"
    );
}

#[test]
fn one_buffer_for_an_update_and_another_parameter_fails() {
    let error = run_main(
        "  %totals = host.take(%input) : (tensor<4xf32>) -> !loop.buf<4xf32, global>
  host.launch(%totals, %totals) {callee = @accumulate, operand_segments = [1, 0, 1]} : (!loop.buf<4xf32, global>, !loop.buf<4xf32, global>) -> ()
  %output = host.to_tensor(%totals) : (!loop.buf<4xf32, global>) -> tensor<4xf32>
  func.return(%output) : (tensor<4xf32>) -> ()",
    )
    .unwrap_err();
    assert_eq!(
        error,
        "host.launch: parameters 0 and 1 of @accumulate get one buffer, which it writes"
    );
}

#[test]
fn a_call_that_reads_its_input_where_its_output_wrote_fails() {
    let error = run_main(
        "  %reversed = host.dispatch(%input) {callee = @reverse_over} : (tensor<4xf32>) -> tensor<4xf32>
  func.return(%reversed) : (tensor<4xf32>) -> ()",
    )
    .unwrap_err();
    assert_eq!(
        error,
        "host.dispatch: loop.for: loop.load: it reads element [1] of an `in` parameter that the \
         `out<over = k>` parameter wrote already, and a launch reads parameter k only at \
         positions that it has not written yet"
    );
}
