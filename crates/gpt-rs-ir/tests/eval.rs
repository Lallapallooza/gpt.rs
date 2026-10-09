//! The semantic rules of the ops on values computed by hand: rounding, saturation, wrapping, ties,
//! the random stream and the layouts of bits.

use gpt_rs_ir::host::{HostTensor, HostValue, TensorData};
use gpt_rs_ir::interp::run;
use gpt_rs_ir::module::Module;

fn f32s(dims: &[usize], values: &[f32]) -> HostValue {
    HostValue::Tensor(HostTensor::f32(dims, values.to_vec()).unwrap())
}

fn si32s(dims: &[usize], values: &[i32]) -> HostValue {
    HostValue::Tensor(HostTensor::si32(dims, values.to_vec()).unwrap())
}

fn i1(value: bool) -> HostValue {
    let data = TensorData::I1(vec![u8::from(value)].into());
    HostValue::Tensor(HostTensor::new(Vec::new(), data).unwrap())
}

/// The results of `@main` in `text` on `args`.
fn eval(text: &str, args: Vec<HostValue>) -> Vec<HostTensor> {
    let module = Module::from_text(text).expect("the module parses");
    run(&module, "main", args)
        .expect("the function runs")
        .into_iter()
        .map(|value| match value {
            HostValue::Tensor(tensor) => tensor,
            other => panic!("expected a tensor, got {other:?}"),
        })
        .collect()
}

fn as_f32(tensor: &HostTensor) -> Vec<f32> {
    tensor.as_f32().unwrap().to_vec()
}

#[test]
fn argmax_prefers_nan_then_the_first_maximum() {
    let out = eval(
        "func.func @main(%0: tensor<2x4xf32>) -> tensor<2xsi32> {
  %1 = tensor.argmax(%0) {axis = 1, out = si32} : (tensor<2x4xf32>) -> tensor<2xsi32>
  func.return(%1) : (tensor<2xsi32>) -> ()
}",
        vec![f32s(
            &[2, 4],
            &[1.0, 3.0, f32::NAN, 3.0, 2.0, 2.0, 1.0, 0.0],
        )],
    );
    assert_eq!(out[0].as_si32().unwrap(), [2, 0]);
}

#[test]
fn top_k_orders_ties_by_index() {
    let out = eval(
        "func.func @main(%0: tensor<5xf32>) -> (tensor<2xf32>, tensor<2xsi32>, tensor<2xf32>, tensor<2xsi32>) {
  %1, %2 = tensor.top_k(%0) {k = 2, axis = 0, largest = true, indices = si32} : (tensor<5xf32>) -> (tensor<2xf32>, tensor<2xsi32>)
  %3, %4 = tensor.top_k(%0) {k = 2, axis = 0, largest = false, indices = si32} : (tensor<5xf32>) -> (tensor<2xf32>, tensor<2xsi32>)
  func.return(%1, %2, %3, %4) : (tensor<2xf32>, tensor<2xsi32>, tensor<2xf32>, tensor<2xsi32>) -> ()
}",
        vec![f32s(&[5], &[5.0, 1.0, 5.0, 3.0, 1.0])],
    );
    assert_eq!(as_f32(&out[0]), [5.0, 5.0]);
    assert_eq!(out[1].as_si32().unwrap(), [0, 2]);
    assert_eq!(as_f32(&out[2]), [1.0, 1.0]);
    assert_eq!(out[3].as_si32().unwrap(), [1, 4]);
}

#[test]
fn gather_and_scatter_address_one_axis() {
    let out = eval(
        "func.func @main(%0: tensor<2x3xf32>, %1: tensor<2x2xsi32>, %2: tensor<3xf32>, %3: tensor<3xsi32>, %4: tensor<3xf32>) -> (tensor<2x2xf32>, tensor<3xf32>, tensor<3xf32>) {
  %5 = tensor.gather(%0, %1) {axis = 1} : (tensor<2x3xf32>, tensor<2x2xsi32>) -> tensor<2x2xf32>
  %6 = tensor.scatter_add(%2, %3, %4) {axis = 0} : (tensor<3xf32>, tensor<3xsi32>, tensor<3xf32>) -> tensor<3xf32>
  %7 = tensor.scatter_replace(%2, %3, %4) {axis = 0} : (tensor<3xf32>, tensor<3xsi32>, tensor<3xf32>) -> tensor<3xf32>
  func.return(%5, %6, %7) : (tensor<2x2xf32>, tensor<3xf32>, tensor<3xf32>) -> ()
}",
        vec![
            f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            si32s(&[2, 2], &[2, 0, 1, 1]),
            f32s(&[3], &[10.0, 20.0, 30.0]),
            si32s(&[3], &[0, 2, 0]),
            f32s(&[3], &[1.0, 2.0, 3.0]),
        ],
    );
    assert_eq!(as_f32(&out[0]), [3.0, 1.0, 5.0, 5.0]);
    assert_eq!(as_f32(&out[1]), [14.0, 20.0, 32.0]);
    // The last update to an element wins.
    assert_eq!(as_f32(&out[2]), [3.0, 20.0, 2.0]);
}

#[test]
fn segments_that_no_row_reaches_hold_the_identity() {
    let out = eval(
        "func.func @main(%0: tensor<4x1xf32>, %1: tensor<4xsi32>) -> (tensor<3x1xf32>, tensor<3x1xf32>) {
  %2 = tensor.segment_sum(%0, %1) {num_segments = 3, indices_are_sorted = true} : (tensor<4x1xf32>, tensor<4xsi32>) -> tensor<3x1xf32>
  %3 = tensor.segment_max(%0, %1) {num_segments = 3} : (tensor<4x1xf32>, tensor<4xsi32>) -> tensor<3x1xf32>
  func.return(%2, %3) : (tensor<3x1xf32>, tensor<3x1xf32>) -> ()
}",
        vec![
            f32s(&[4, 1], &[1.0, 2.0, 3.0, 4.0]),
            si32s(&[4], &[0, 0, 2, 2]),
        ],
    );
    assert_eq!(as_f32(&out[0]), [3.0, 0.0, 7.0]);
    assert_eq!(as_f32(&out[1]), [2.0, f32::NEG_INFINITY, 4.0]);
}

#[test]
fn cond_runs_one_branch() {
    let text = "func.func @main(%0: tensor<i1>, %1: tensor<2xf32>) -> tensor<2xf32> {
  %2 = tensor.cond(%0, %1) {results = [tensor<2xf32>]} ({
  ^then_branch(%3: tensor<2xf32>):
    %4 = tensor.neg(%3) : (tensor<2xf32>) -> tensor<2xf32>
    tensor.yield(%4) : (tensor<2xf32>) -> ()
  }, {
  ^else_branch(%5: tensor<2xf32>):
    %6 = tensor.add(%5, %5) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    tensor.yield(%6) : (tensor<2xf32>) -> ()
  }) : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>
  func.return(%2) : (tensor<2xf32>) -> ()
}";
    let x = || f32s(&[2], &[1.0, -2.0]);
    assert_eq!(as_f32(&eval(text, vec![i1(true), x()])[0]), [-1.0, 2.0]);
    assert_eq!(as_f32(&eval(text, vec![i1(false), x()])[0]), [2.0, -4.0]);
}

#[test]
fn while_loops_until_cond_fails() {
    let out = eval(
        "func.func @main(%0: tensor<si32>, %1: tensor<f32>) -> (tensor<si32>, tensor<f32>) {
  %2, %3 = tensor.while(%0, %1) ({
  ^cond(%4: tensor<si32>, %5: tensor<f32>):
    %6 = tensor.constant() {value = dense<[] x si32: 5>} : () -> tensor<si32>
    %7 = tensor.compare(%4, %6) {kind = less} : (tensor<si32>, tensor<si32>) -> tensor<i1>
    tensor.yield(%7, %4, %5) : (tensor<i1>, tensor<si32>, tensor<f32>) -> ()
  }, {
  ^body(%8: tensor<si32>, %9: tensor<f32>):
    %10 = tensor.constant() {value = dense<[] x si32: 1>} : () -> tensor<si32>
    %11 = tensor.add(%8, %10) : (tensor<si32>, tensor<si32>) -> tensor<si32>
    %12 = tensor.add(%9, %9) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tensor.yield(%11, %12) : (tensor<si32>, tensor<f32>) -> ()
  }) : (tensor<si32>, tensor<f32>) -> (tensor<si32>, tensor<f32>)
  func.return(%2, %3) : (tensor<si32>, tensor<f32>) -> ()
}",
        vec![si32s(&[], &[1]), f32s(&[], &[1.0])],
    );
    // Four steps take the counter from 1 to 5 and double the value four times.
    assert_eq!(out[0].as_si32().unwrap(), [5]);
    assert_eq!(as_f32(&out[1]), [16.0]);
}

#[test]
fn scan_carries_a_running_sum_and_stacks_each_step() {
    let out = eval(
        "func.func @main(%0: tensor<2xf32>, %1: tensor<3x2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>) {
  %2, %3 = tensor.scan(%0, %1) {carry_count = 1, outputs = [tensor<2xf32>]} ({
  ^body(%4: tensor<2xf32>, %5: tensor<2xf32>):
    %6 = tensor.add(%4, %5) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    tensor.yield(%6, %6) : (tensor<2xf32>, tensor<2xf32>) -> ()
  }) : (tensor<2xf32>, tensor<3x2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>)
  func.return(%2, %3) : (tensor<2xf32>, tensor<3x2xf32>) -> ()
}",
        vec![
            f32s(&[2], &[0.0, 100.0]),
            f32s(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
    );
    assert_eq!(as_f32(&out[0]), [9.0, 112.0]);
    assert_eq!(as_f32(&out[1]), [1.0, 102.0, 4.0, 106.0, 9.0, 112.0]);
}

#[test]
fn f4e2m1_rounds_ties_to_even_and_saturates() {
    // The f4e2m1 values are 0, 0.5, 1, 1.5, 2, 3, 4 and 6. Each input but the last two lies halfway
    // between two of them and goes to the one with the even code. 7 saturates, and NaN gives 0.
    let x = [0.25, 0.75, 1.25, 2.5, 5.0, -3.5, 7.0, f32::NAN];
    let out = eval(
        "func.func @main(%0: tensor<8xf32>, %1: tensor<1xf32>) -> tensor<8xf32> {
  %2 = tensor.quantize(%0, %1) {block = [8], to = f4e2m1} : (tensor<8xf32>, tensor<1xf32>) -> tensor<8xf4e2m1>
  %3 = tensor.dequantize(%2, %1) {block = [8], to = f32} : (tensor<8xf4e2m1>, tensor<1xf32>) -> tensor<8xf32>
  func.return(%3) : (tensor<8xf32>) -> ()
}",
        vec![f32s(&[8], &x), f32s(&[1], &[1.0])],
    );
    assert_eq!(as_f32(&out[0]), [0.0, 1.0, 1.0, 2.0, 4.0, -4.0, 6.0, 0.0]);
}

#[test]
fn mx_scales_map_the_block_maximum_to_the_top_power_of_two() {
    // amax 12 has exponent 3 and f4e2m1's largest power of two is 4 = 2^2, so the scale is 2^1:
    // e8m0 code 128. Every element divided by 2 is an f4e2m1 value.
    let out = eval(
        "func.func @main(%0: tensor<4xf32>) -> (tensor<4xf32>, tensor<1xe8m0>) {
  %1, %2 = tensor.quantize_dynamic(%0) {block = [4], to = f4e2m1, scale_dtype = e8m0, scale_rule = pow2_floor} : (tensor<4xf32>) -> (tensor<4xf4e2m1>, tensor<1xe8m0>)
  %3 = tensor.dequantize(%1, %2) {block = [4], to = f32} : (tensor<4xf4e2m1>, tensor<1xe8m0>) -> tensor<4xf32>
  func.return(%3, %2) : (tensor<4xf32>, tensor<1xe8m0>) -> ()
}",
        vec![f32s(&[4], &[1.0, -2.0, 3.0, 12.0])],
    );
    assert_eq!(as_f32(&out[0]), [1.0, -2.0, 3.0, 12.0]);
    assert_eq!(out[1].to_le_bytes(), [128]);
}

#[test]
fn f8e4m3_saturates_at_448() {
    let out = eval(
        "func.func @main(%0: tensor<3xf32>, %1: tensor<1xf32>) -> tensor<3xf32> {
  %2 = tensor.quantize(%0, %1) {block = [3], to = f8e4m3} : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf8e4m3>
  %3 = tensor.dequantize(%2, %1) {block = [3], to = f32} : (tensor<3xf8e4m3>, tensor<1xf32>) -> tensor<3xf32>
  func.return(%3) : (tensor<3xf32>) -> ()
}",
        vec![f32s(&[3], &[100.0, 300.0, -1000.0]), f32s(&[1], &[0.5])],
    );
    // 100 / 0.5 = 200 lies halfway between the e4m3 values 192 and 208 and rounds to 192, the even
    // code. 600 and -2000 saturate to +-448.
    assert_eq!(as_f32(&out[0]), [96.0, 224.0, -224.0]);
}

#[test]
fn casts_truncate_saturate_and_round_once() {
    let out = eval(
        "func.func @main(%0: tensor<4xf32>, %1: tensor<2xsi32>, %2: tensor<1xsi32>) -> (tensor<4xsi32>, tensor<2xf32>, tensor<1xf32>) {
  %3 = tensor.cast(%0) {to = si32} : (tensor<4xf32>) -> tensor<4xsi32>
  %4 = tensor.cast(%1) {to = si8} : (tensor<2xsi32>) -> tensor<2xsi8>
  %5 = tensor.cast(%4) {to = f32} : (tensor<2xsi8>) -> tensor<2xf32>
  %6 = tensor.cast(%2) {to = f32} : (tensor<1xsi32>) -> tensor<1xf32>
  func.return(%3, %5, %6) : (tensor<4xsi32>, tensor<2xf32>, tensor<1xf32>) -> ()
}",
        vec![
            f32s(&[4], &[-1.7, 2.5, f32::NAN, 3e9]),
            si32s(&[2], &[300, -300]),
            si32s(&[1], &[16_777_217]),
        ],
    );
    // Floats truncate toward zero, NaN gives 0, and 3e9 saturates.
    assert_eq!(out[0].as_si32().unwrap(), [-1, 2, 0, i32::MAX]);
    // si32 to si8 saturates.
    assert_eq!(as_f32(&out[1]), [127.0, -128.0]);
    // 2^24 + 1 lies halfway between two f32 values and rounds to the even one, 2^24.
    assert_eq!(as_f32(&out[2]), [16_777_216.0]);
}

#[test]
fn si64_indices_address_like_si32_ones() {
    let indices = HostValue::Tensor(
        HostTensor::new(vec![2], TensorData::Si64(vec![2i64, 0].into())).unwrap(),
    );
    let out = eval(
        "func.func @main(%0: tensor<3x2xf32>, %1: tensor<2xsi64>) -> tensor<2x2xf32> {
  %2 = tensor.take(%0, %1) : (tensor<3x2xf32>, tensor<2xsi64>) -> tensor<2x2xf32>
  func.return(%2) : (tensor<2x2xf32>) -> ()
}",
        vec![f32s(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), indices],
    );
    assert_eq!(as_f32(&out[0]), [5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn casts_into_narrow_floats_round_to_even_and_saturate() {
    let out = eval(
        "func.func @main(%0: tensor<5xf32>, %1: tensor<4xf32>, %2: tensor<3xf32>, %3: tensor<2xsi32>) -> (tensor<5xf32>, tensor<4xf32>, tensor<3xf32>, tensor<2xf32>) {
  %4 = tensor.cast(%0) {to = f8e4m3} : (tensor<5xf32>) -> tensor<5xf8e4m3>
  %5 = tensor.cast(%4) {to = f32} : (tensor<5xf8e4m3>) -> tensor<5xf32>
  %6 = tensor.cast(%1) {to = f8e5m2} : (tensor<4xf32>) -> tensor<4xf8e5m2>
  %7 = tensor.cast(%6) {to = f32} : (tensor<4xf8e5m2>) -> tensor<4xf32>
  %8 = tensor.cast(%2) {to = f4e2m1} : (tensor<3xf32>) -> tensor<3xf4e2m1>
  %9 = tensor.cast(%8) {to = f32} : (tensor<3xf4e2m1>) -> tensor<3xf32>
  %10 = tensor.cast(%3) {to = f8e4m3} : (tensor<2xsi32>) -> tensor<2xf8e4m3>
  %11 = tensor.cast(%10) {to = f32} : (tensor<2xf8e4m3>) -> tensor<2xf32>
  func.return(%5, %7, %9, %11) : (tensor<5xf32>, tensor<4xf32>, tensor<3xf32>, tensor<2xf32>) -> ()
}",
        vec![
            f32s(&[5], &[500.0, -1e9, f32::INFINITY, f32::NAN, 0.3]),
            f32s(&[4], &[1e6, f32::NEG_INFINITY, 3.0, 100.0]),
            f32s(&[3], &[7.0, f32::NAN, -0.2]),
            si32s(&[2], &[1000, 17]),
        ],
    );
    // e4m3 saturates at 448 and keeps NaN. 0.3 lies above the midpoint 0.296875 of its neighbors
    // 0.28125 and 0.3125.
    let e4m3 = as_f32(&out[0]);
    assert_eq!(e4m3[..3], [448.0, -448.0, 448.0]);
    assert!(e4m3[3].is_nan());
    assert_eq!(e4m3[4], 0.3125);
    // e5m2 saturates at 57344, infinities included. 100 lies below the midpoint 104 of 96 and 112.
    assert_eq!(as_f32(&out[1]), [57344.0, -57344.0, 3.0, 96.0]);
    // f4e2m1 saturates at 6 and has no NaN, so NaN gives 0.
    assert_eq!(as_f32(&out[2]), [6.0, 0.0, -0.0]);
    // 17 is halfway between the e4m3 values 16 and 18 and rounds to 16, the even code.
    assert_eq!(as_f32(&out[3]), [448.0, 16.0]);
}

fn tensor(dims: &[usize], data: TensorData) -> HostValue {
    HostValue::Tensor(HostTensor::new(dims, data).unwrap())
}

#[test]
fn integers_wrap_and_divide_toward_zero() {
    let out = eval(
        "func.func @main(%0: tensor<4xsi8>, %1: tensor<4xsi8>, %2: tensor<2xui8>, %3: tensor<2xui8>) -> (tensor<4xsi8>, tensor<4xsi8>, tensor<2xui8>) {
  %4 = tensor.add(%0, %1) : (tensor<4xsi8>, tensor<4xsi8>) -> tensor<4xsi8>
  %5 = tensor.div(%0, %1) : (tensor<4xsi8>, tensor<4xsi8>) -> tensor<4xsi8>
  %6 = tensor.sub(%2, %3) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
  func.return(%4, %5, %6) : (tensor<4xsi8>, tensor<4xsi8>, tensor<2xui8>) -> ()
}",
        vec![
            tensor(&[4], TensorData::Si8(vec![127, -128, -7, -128].into())),
            tensor(&[4], TensorData::Si8(vec![1, -1, 2, -1].into())),
            tensor(&[2], TensorData::Ui8(vec![0, 5].into())),
            tensor(&[2], TensorData::Ui8(vec![1, 3].into())),
        ],
    );
    // 127 + 1 and -128 - 1 wrap around 2^8.
    assert_eq!(
        out[0].data,
        TensorData::Si8(vec![-128, 127, -5, 127].into())
    );
    // -7 / 2 truncates to -3, and -128 / -1 wraps to -128.
    assert_eq!(
        out[1].data,
        TensorData::Si8(vec![127, -128, -3, -128].into())
    );
    assert_eq!(out[2].data, TensorData::Ui8(vec![255, 2].into()));
}

#[test]
fn shifts_read_the_width_of_their_dtype() {
    let out = eval(
        "func.func @main(%0: tensor<4xsi8>, %1: tensor<4xsi8>) -> (tensor<4xsi8>, tensor<4xsi8>, tensor<4xsi8>) {
  %2 = tensor.shift_right_arithmetic(%0, %1) : (tensor<4xsi8>, tensor<4xsi8>) -> tensor<4xsi8>
  %3 = tensor.shift_right_logical(%0, %1) : (tensor<4xsi8>, tensor<4xsi8>) -> tensor<4xsi8>
  %4 = tensor.shift_left(%0, %1) : (tensor<4xsi8>, tensor<4xsi8>) -> tensor<4xsi8>
  func.return(%2, %3, %4) : (tensor<4xsi8>, tensor<4xsi8>, tensor<4xsi8>) -> ()
}",
        vec![
            tensor(&[4], TensorData::Si8(vec![-16, -16, -16, 1].into())),
            // -1 reads as the unsigned amount 255, beyond the width like 9.
            tensor(&[4], TensorData::Si8(vec![2, -1, 9, 7].into())),
        ],
    );
    // An arithmetic shift by the width or more leaves the sign in every bit.
    assert_eq!(out[0].data, TensorData::Si8(vec![-4, -1, -1, 0].into()));
    // -16 is 0xf0, and 0xf0 >> 2 is 0x3c.
    assert_eq!(out[1].data, TensorData::Si8(vec![60, 0, 0, 0].into()));
    // 0xf0 << 2 is 0xc0, and 1 << 7 is 0x80.
    assert_eq!(out[2].data, TensorData::Si8(vec![-64, 0, 0, -128].into()));
}

#[test]
fn logic_on_i1_and_its_reductions() {
    let bits = |values: &[u8]| TensorData::I1(values.to_vec().into());
    let out = eval(
        "func.func @main(%0: tensor<2x2xi1>, %1: tensor<2x2xi1>) -> (tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2xi1>, tensor<2xi1>) {
  %2 = tensor.and(%0, %1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  %3 = tensor.or(%0, %1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  %4 = tensor.xor(%0, %1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  %5 = tensor.not(%0) : (tensor<2x2xi1>) -> tensor<2x2xi1>
  %6 = tensor.reduce_and(%0) {axes = [1]} : (tensor<2x2xi1>) -> tensor<2xi1>
  %7 = tensor.reduce_or(%1) {axes = [1]} : (tensor<2x2xi1>) -> tensor<2xi1>
  func.return(%2, %3, %4, %5, %6, %7) : (tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2xi1>, tensor<2xi1>) -> ()
}",
        vec![
            tensor(&[2, 2], bits(&[1, 1, 0, 1])),
            tensor(&[2, 2], bits(&[1, 0, 0, 0])),
        ],
    );
    assert_eq!(out[0].data, bits(&[1, 0, 0, 0]));
    assert_eq!(out[1].data, bits(&[1, 1, 0, 1]));
    assert_eq!(out[2].data, bits(&[0, 1, 0, 1]));
    assert_eq!(out[3].data, bits(&[0, 0, 1, 0]));
    // All of each row of %0, and any of each row of %1.
    assert_eq!(out[4].data, bits(&[1, 0]));
    assert_eq!(out[5].data, bits(&[1, 0]));
}

#[test]
fn unsigned_integers_order_as_unsigned() {
    let out = eval(
        "func.func @main(%0: tensor<2xui64>, %1: tensor<2xui64>) -> (tensor<2xi1>, tensor<2xui64>) {
  %2 = tensor.compare(%0, %1) {kind = greater} : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  %3 = tensor.maximum(%0, %1) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
  func.return(%2, %3) : (tensor<2xi1>, tensor<2xui64>) -> ()
}",
        vec![
            tensor(&[2], TensorData::Ui64(vec![u64::MAX, 1].into())),
            tensor(&[2], TensorData::Ui64(vec![1, 2].into())),
        ],
    );
    assert_eq!(out[0].data, TensorData::I1(vec![1, 0].into()));
    assert_eq!(out[1].data, TensorData::Ui64(vec![u64::MAX, 2].into()));
}

#[test]
fn f64_computes_in_f64() {
    let out = eval(
        "func.func @main(%0: tensor<1xf64>, %1: tensor<1xf64>) -> tensor<1xf64> {
  %2 = tensor.add(%0, %1) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
  func.return(%2) : (tensor<1xf64>) -> ()
}",
        vec![
            tensor(&[1], TensorData::F64(vec![0.1].into())),
            tensor(&[1], TensorData::F64(vec![0.2].into())),
        ],
    );
    // In f32 the sum would round to 0.3.
    assert_eq!(
        out[0].data,
        TensorData::F64(vec![0.30000000000000004].into())
    );
}

#[test]
fn sin_and_cos_keep_the_sign_of_zero() {
    let out = eval(
        "func.func @main(%0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %1 = tensor.sin(%0) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = tensor.cos(%0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return(%1, %2) : (tensor<2xf32>, tensor<2xf32>) -> ()
}",
        vec![f32s(&[2], &[0.0, -0.0])],
    );
    let sin = as_f32(&out[0]);
    assert_eq!(
        (sin[0].to_bits(), sin[1].to_bits()),
        (0, (-0.0f32).to_bits())
    );
    assert_eq!(as_f32(&out[1]), [1.0, 1.0]);
}

#[test]
fn bitcast_reads_the_bits_of_each_row() {
    let out = eval(
        "func.func @main(%0: tensor<1xsi32>, %1: tensor<2xui8>) -> (tensor<8xui4>, tensor<4xf32>) {
  %2 = tensor.bitcast(%0) {to = ui4} : (tensor<1xsi32>) -> tensor<8xui4>
  %3 = tensor.bitcast(%1) {to = f4e2m1} : (tensor<2xui8>) -> tensor<4xf4e2m1>
  %4 = tensor.cast(%3) {to = f32} : (tensor<4xf4e2m1>) -> tensor<4xf32>
  func.return(%2, %4) : (tensor<8xui4>, tensor<4xf32>) -> ()
}",
        vec![
            si32s(&[1], &[0x7654_3210]),
            tensor(&[2], TensorData::Ui8(vec![0x21, 0x43].into())),
        ],
    );
    // The nibbles of the word from its least significant one up, as GPTQ packs eight 4-bit codes.
    let TensorData::Packed(nibbles) = &out[0].data else {
        panic!("ui4 elements are packed");
    };
    assert_eq!(
        (0..8).map(|i| nibbles.get(i)).collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
    // The f4e2m1 codes 1, 2, 3 and 4.
    assert_eq!(as_f32(&out[1]), [0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn strided_slices_step_through_their_input() {
    let out = eval(
        "func.func @main(%0: tensor<9xf32>) -> tensor<3xf32> {
  %1 = tensor.slice(%0) {starts = [1], sizes = [3], strides = [3]} : (tensor<9xf32>) -> tensor<3xf32>
  func.return(%1) : (tensor<3xf32>) -> ()
}",
        vec![f32s(&[9], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
    );
    assert_eq!(as_f32(&out[0]), [1.0, 4.0, 7.0]);
}

#[test]
fn philox_gives_its_known_answer() {
    let out = eval(
        "func.func @main(%0: tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>, tensor<1xf32>) {
  %1, %2 = tensor.rng_uniform(%0) {shape = [4], dtype = ui32} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
  %3, %4 = tensor.rng_uniform(%0) {shape = [1], dtype = f32} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1xf32>)
  func.return(%1, %2, %4) : (tensor<2xui64>, tensor<4xui32>, tensor<1xf32>) -> ()
}",
        vec![tensor(&[2], TensorData::Ui64(vec![0, 0].into()))],
    );
    // Random123's known answer for Philox4x32-10 with counter and key 0.
    let words = [0x6627_e8d5u32, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8];
    assert_eq!(out[1].data, TensorData::Ui32(words.to_vec().into()));
    // One block read: the counter moves on by one.
    assert_eq!(out[0].data, TensorData::Ui64(vec![0, 1].into()));
    // A uniform f32 is the top 24 bits of its word over 2^24.
    assert_eq!(as_f32(&out[2]), [(words[0] >> 8) as f32 / 16_777_216.0]);
}

#[test]
fn stochastic_rounding_compares_the_draw_with_the_fraction() {
    let out = eval(
        "func.func @main(%0: tensor<4xf32>, %1: tensor<1xf32>, %2: tensor<4xui32>) -> tensor<4xsi8> {
  %3 = tensor.quantize(%0, %1, %2) {block = [4], to = si8, rounding = stochastic, operand_segments = [1, 1, 0, 0, 1]} : (tensor<4xf32>, tensor<1xf32>, tensor<4xui32>) -> tensor<4xsi8>
  func.return(%3) : (tensor<4xsi8>) -> ()
}",
        vec![
            f32s(&[4], &[0.25, 0.25, -0.25, 2.75]),
            f32s(&[1], &[1.0]),
            tensor(
                &[4],
                TensorData::Ui32(vec![0, 0x8000_0000, 0x3fff_ffff, 0xbfff_ffff].into()),
            ),
        ],
    );
    // A fraction of 0.25 rounds away from zero for draws below 2^30, and a fraction of 0.75 for
    // draws below 0xc0000000.
    assert_eq!(out[0].data, TensorData::Si8(vec![1, 0, -1, 3].into()));
}

#[test]
fn asymmetric_quantization_keeps_zero_exact() {
    let out = eval(
        "func.func @main(%0: tensor<4xf32>) -> (tensor<4xui8>, tensor<4xf32>) {
  %1, %2, %3 = tensor.quantize_dynamic(%0) {block = [4], to = ui8, scale_dtype = f32, asymmetric = true} : (tensor<4xf32>) -> (tensor<4xui8>, tensor<1xf32>, tensor<1xf32>)
  %4 = tensor.dequantize(%1, %2, %3) {block = [4], to = f32, operand_segments = [1, 1, 1, 0]} : (tensor<4xui8>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>
  func.return(%1, %4) : (tensor<4xui8>, tensor<4xf32>) -> ()
}",
        vec![f32s(&[4], &[-1.0, 0.0, 1.0, 3.0])],
    );
    // The scale is 4 / 255, and 0 sits at the code round(1 / scale) = round(63.75) = 64. The other
    // quotients are 0.25, 127.75 and 255.25, which saturates.
    assert_eq!(out[0].data, TensorData::Ui8(vec![0, 64, 128, 255].into()));
    assert_eq!(as_f32(&out[1])[1].to_bits(), 0);
}

#[test]
fn tables_map_codes_to_their_values() {
    let out = eval(
        "func.func @main(%0: tensor<4xf32>, %1: tensor<1xf32>, %2: tensor<5xf32>) -> (tensor<4xui8>, tensor<4xf32>) {
  %3 = tensor.quantize(%0, %1, %2) {block = [4], to = ui8, operand_segments = [1, 1, 0, 1, 0]} : (tensor<4xf32>, tensor<1xf32>, tensor<5xf32>) -> tensor<4xui8>
  %4 = tensor.dequantize(%3, %1, %2) {block = [4], to = f32, operand_segments = [1, 1, 0, 1]} : (tensor<4xui8>, tensor<1xf32>, tensor<5xf32>) -> tensor<4xf32>
  func.return(%3, %4) : (tensor<4xui8>, tensor<4xf32>) -> ()
}",
        vec![
            f32s(&[4], &[0.3, -0.75, 1.2, f32::NAN]),
            f32s(&[1], &[1.0]),
            f32s(&[5], &[-1.0, -0.5, 0.0, 0.5, 1.0]),
        ],
    );
    // -0.75 is as near -1 as -0.5 and takes the lowest index. NaN takes index 0.
    assert_eq!(out[0].data, TensorData::Ui8(vec![3, 0, 4, 0].into()));
    assert_eq!(as_f32(&out[1]), [0.5, -1.0, 1.0, -1.0]);
}

#[test]
fn power_of_two_scales_round_the_exponent_down_or_up() {
    let out = eval(
        "func.func @main(%0: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
  %1, %2 = tensor.quantize_dynamic(%0) {block = [1], to = f8e4m3, scale_dtype = e8m0, scale_rule = pow2_floor} : (tensor<1xf32>) -> (tensor<1xf8e4m3>, tensor<1xe8m0>)
  %3 = tensor.dequantize(%1, %2) {block = [1], to = f32} : (tensor<1xf8e4m3>, tensor<1xe8m0>) -> tensor<1xf32>
  %4, %5 = tensor.quantize_dynamic(%0) {block = [1], to = f8e4m3, scale_dtype = e8m0, scale_rule = pow2_ceil} : (tensor<1xf32>) -> (tensor<1xf8e4m3>, tensor<1xe8m0>)
  %6 = tensor.dequantize(%4, %5) {block = [1], to = f32} : (tensor<1xf8e4m3>, tensor<1xe8m0>) -> tensor<1xf32>
  func.return(%3, %6) : (tensor<1xf32>, tensor<1xf32>) -> ()
}",
        vec![f32s(&[1], &[500.0])],
    );
    // The MX scale 2^(8 - 8) = 1 leaves 500 above 448, which saturates. The ceiling scale
    // 2^ceil(log2(500 / 448)) = 2 maps it to 250, which rounds to the e4m3 value 256.
    assert_eq!(as_f32(&out[0]), [448.0]);
    assert_eq!(as_f32(&out[1]), [512.0]);
}

#[test]
fn mixed_dots_accumulate_in_accum() {
    let out = eval(
        "func.func @main(%0: tensor<1x2xf32>, %1: tensor<2x1xf32>, %2: tensor<1x2xsi8>) -> (tensor<1x1xf32>, tensor<1x1xsi32>) {
  %3 = tensor.cast(%0) {to = f8e4m3} : (tensor<1x2xf32>) -> tensor<1x2xf8e4m3>
  %4 = tensor.cast(%1) {to = f8e5m2} : (tensor<2x1xf32>) -> tensor<2x1xf8e5m2>
  %5 = tensor.dot_general(%3, %4) {contract_lhs = [1], contract_rhs = [0], accum = f32, out = f32} : (tensor<1x2xf8e4m3>, tensor<2x1xf8e5m2>) -> tensor<1x1xf32>
  %6 = tensor.transpose(%2) {perm = [1, 0]} : (tensor<1x2xsi8>) -> tensor<2x1xsi8>
  %7 = tensor.dot_general(%2, %6) {contract_lhs = [1], contract_rhs = [0], accum = si32, out = si32} : (tensor<1x2xsi8>, tensor<2x1xsi8>) -> tensor<1x1xsi32>
  func.return(%5, %7) : (tensor<1x1xf32>, tensor<1x1xsi32>) -> ()
}",
        vec![
            f32s(&[1, 2], &[1.5, 2.0]),
            f32s(&[2, 1], &[3.0, 0.5]),
            tensor(&[1, 2], TensorData::Si8(vec![100, 100].into())),
        ],
    );
    assert_eq!(as_f32(&out[0]), [5.5]);
    // 100 * 100 + 100 * 100 overflows si8 but not the si32 accumulator.
    assert_eq!(out[1].data, TensorData::Si32(vec![20_000].into()));
}

#[test]
fn fnuz_floats_have_no_negative_zero() {
    let out = eval(
        "func.func @main(%0: tensor<3xf32>) -> tensor<3xf32> {
  %1 = tensor.cast(%0) {to = f8e4m3fnuz} : (tensor<3xf32>) -> tensor<3xf8e4m3fnuz>
  %2 = tensor.cast(%1) {to = f32} : (tensor<3xf8e4m3fnuz>) -> tensor<3xf32>
  func.return(%2) : (tensor<3xf32>) -> ()
}",
        vec![f32s(&[3], &[300.0, -0.0, f32::NAN])],
    );
    let values = as_f32(&out[0]);
    // e4m3fnuz saturates at 240, and its only NaN code is the one -0.0 would have.
    assert_eq!(values[0], 240.0);
    assert_eq!(values[1].to_bits(), 0);
    assert!(values[2].is_nan());
}

#[test]
fn iota_converts_each_count_as_a_cast_does() {
    let iota = |dtype: &str, extent: usize| {
        let text = format!(
            "func.func @main() -> tensor<{extent}x{dtype}> {{
  %0 = tensor.iota() {{shape = [{extent}], dtype = {dtype}, axis = 0}} : () -> tensor<{extent}x{dtype}>
  func.return(%0) : (tensor<{extent}x{dtype}>) -> ()
}}"
        );
        eval(&text, Vec::new()).remove(0)
    };
    // bf16 rounds 257 and 259 to the even neighbors 256 and 260. Counting in bf16 would stop at
    // 256.
    let bf16 = iota("bf16", 260);
    let bits: Vec<u16> = bf16.to_le_bytes()[2 * 255..]
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(bits, [0x437F, 0x4380, 0x4380, 0x4381, 0x4382]);
    // ui8 saturates at 255, as an integer cast does. Counting in ui8 would wrap to 0.
    assert_eq!(iota("ui8", 258).to_le_bytes()[254..], [254, 255, 255, 255]);
    // Counting along axis 1 repeats along axis 0.
    let out = eval(
        "func.func @main() -> tensor<2x3xsi32> {
  %0 = tensor.iota() {shape = [2, 3], dtype = si32, axis = 1} : () -> tensor<2x3xsi32>
  func.return(%0) : (tensor<2x3xsi32>) -> ()
}",
        Vec::new(),
    );
    assert_eq!(out[0].as_si32().unwrap(), [0, 1, 2, 0, 1, 2]);
}
