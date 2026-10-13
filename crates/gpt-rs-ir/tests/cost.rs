//! The costs that `gpt_rs_ir::analysis` documents, on hand-computed cases.

use gpt_rs_ir::analysis::Cost;
use gpt_rs_ir::ir::Module;

/// The cost of each op of the one function of `text`, in order, the terminator left out.
fn costs(text: &str) -> Vec<Option<Cost>> {
    let module = Module::from_text(text).expect("the case parses");
    let func = module.funcs()[0];
    let body = module.regions(func)[0];
    let insts: Vec<_> = module.insts(body).collect();
    insts[..insts.len() - 1]
        .iter()
        .map(|&inst| module.cost(inst))
        .collect()
}

fn cost(flops: u64, bytes_read: u64, bytes_written: u64) -> Option<Cost> {
    Some(Cost {
        flops,
        bytes_read,
        bytes_written,
    })
}

#[test]
fn dot_counts_a_multiply_and_an_add_per_product() {
    let text = "func.func @f(%0: tensor<2x3xf32>, %1: tensor<3x4xbf16>) -> tensor<2x4xf32> {
  %2 = tensor.dot_general(%0, %1) {contract_lhs = [1], contract_rhs = [0], accum = f32, out = f32} : (tensor<2x3xf32>, tensor<3x4xbf16>) -> tensor<2x4xf32>
  func.return(%2) : (tensor<2x4xf32>) -> ()
}";
    // 2 x 4 results, each the sum of 3 products; 24 + 24 bytes of operands, 32 of result.
    assert_eq!(costs(text), [cost(2 * 2 * 4 * 3, 24 + 24, 32)]);
}

#[test]
fn gather_reads_the_rows_it_gathers() {
    let text = "func.func @f(%0: tensor<100x8xf32>, %1: tensor<3xsi32>) -> tensor<3x8xf32> {
  %2 = tensor.gather(%0, %1) {axis = 0} : (tensor<100x8xf32>, tensor<3xsi32>) -> tensor<3x8xf32>
  func.return(%2) : (tensor<3x8xf32>) -> ()
}";
    // 3 rows of 8 f32 and 3 si32 indices, not the 3200-byte table.
    assert_eq!(costs(text), [cost(0, 12 + 96, 96)]);
}

#[test]
fn movement_reads_the_smaller_side_and_elementwise_reads_every_operand() {
    let text = "func.func @f(%0: tensor<8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %2 = tensor.broadcast_to(%0) {shape = [4, 8]} : (tensor<8xf32>) -> tensor<4x8xf32>
  %3 = tensor.add(%2, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %4 = tensor.slice(%3) {starts = [0, 0], sizes = [1, 8]} : (tensor<4x8xf32>) -> tensor<1x8xf32>
  %5 = tensor.broadcast_to(%4) {shape = [4, 8]} : (tensor<1x8xf32>) -> tensor<4x8xf32>
  func.return(%5) : (tensor<4x8xf32>) -> ()
}";
    assert_eq!(
        costs(text),
        [
            // The broadcast reads its 8 elements and writes 32.
            cost(0, 32, 128),
            // The add computes 32 elements from two 32-element operands.
            cost(32, 256, 128),
            // The slice reads the 8 elements it keeps.
            cost(0, 32, 32),
            cost(0, 32, 128),
        ]
    );
}

#[test]
fn a_dynamic_extent_leaves_the_cost_unknown() {
    let text = "func.func @f(%0: tensor<{T}x8xf32>) -> tensor<{T}x8xf32> {
  %1 = tensor.exp(%0) : (tensor<{T}x8xf32>) -> tensor<{T}x8xf32>
  func.return(%1) : (tensor<{T}x8xf32>) -> ()
}";
    assert_eq!(costs(text), [None]);
}
