//! The costs that `gpt_rs_ir::analysis` documents, on hand-computed cases.

use gpt_rs_ir::analysis::Cost;
use gpt_rs_ir::ir::Module;

/// The cost of each op of the one function of `text`, in order, the terminator left out.
fn costs(text: &str) -> Vec<Option<Cost>> {
    let module = Module::from_text(text).expect("the case parses");
    let funcs = module.funcs();
    let insts: Vec<_> = funcs[0].body().insts().collect();
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
    // 2 x 4 results, each the sum of 3 products. 24 + 24 bytes of operands, 32 of result.
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

/// A product of a 2x3 and a 3x4 matrix as one block: 24 points, each a product and the sum it
/// enters.
const PRODUCT_BLOCK: &str = "%2 = tile.block(%0, %1) {iters = [par 2, par 4, red 3], reads = [$0[d0, d2], $1[d2, d1]], folds = [sum -> [d0, d1]]} ({
  ^body(%3: f32, %4: f32):
    %5 = arith.mul(%3, %4) : (f32, f32) -> f32
    tile.yield(%5) : (f32) -> ()
  }) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>";

#[test]
fn a_block_runs_its_body_at_each_point() {
    let text = format!(
        "func.func @f(%0: tensor<2x3xf32>, %1: tensor<3x4xf32>) -> tensor<2x4xf32> {{
  {PRODUCT_BLOCK}
  func.return(%2) : (tensor<2x4xf32>) -> ()
}}"
    );
    // 2 x 4 x 3 points of a product and a sum. 24 + 48 bytes of operands, 32 of result.
    assert_eq!(costs(&text), [cost(2 * 4 * 3 * 2, 24 + 48, 32)]);
}

#[test]
fn a_block_reads_the_rows_it_gathers() {
    let text = "func.func @f(%0: tensor<100x8xf32>, %1: tensor<3xsi32>) -> tensor<3x8xf32> {
  %2 = tile.block(%0, %1) {iters = [par 3, par 8], reads = [$0[$1[d0], d1]], folds = [none -> [d0, d1]]} ({
  ^body(%3: f32):
    tile.yield(%3) : (f32) -> ()
  }) : (tensor<100x8xf32>, tensor<3xsi32>) -> tensor<3x8xf32>
  func.return(%2) : (tensor<3x8xf32>) -> ()
}";
    // As `tensor.gather`: 3 rows of 8 f32 and 3 si32 indices, not the 3200-byte table.
    assert_eq!(costs(text), [cost(0, 12 + 96, 96)]);
}

#[test]
fn a_block_that_updates_an_operand_writes_the_elements_it_writes() {
    let text = "func.func @f(%0: tensor<16x8xf32>, %1: tensor<2x8xf32>, %2: tensor<1xsi32>) -> tensor<16x8xf32> {
  %3 = tile.block(%0, %1, %2) {iters = [par 2, par 8], reads = [$1[d0, d1]], folds = [none -> $0[$2[0] + d0, d1]]} ({
  ^body(%4: f32):
    tile.yield(%4) : (f32) -> ()
  }) : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<1xsi32>) -> tensor<16x8xf32>
  func.return(%3) : (tensor<16x8xf32>) -> ()
}";
    // As `tensor.dynamic_update_slice`: it reads the 2x8 f32 patch and its si32 start, and writes
    // the patch into the 512-byte cache, which it does not read.
    assert_eq!(costs(text), [cost(0, 4 + 64, 64)]);
}

#[test]
fn a_fold_takes_one_step_per_point() {
    let text = "func.func @f(%0: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xsi32>) {
  %1, %2 = tile.block(%0) {iters = [par 4, red 8], reads = [$0[d0, d1]], folds = [argmax -> ([d0], [d0])]} ({
  ^body(%3: f32):
    %4 = tile.index() {expr = d1, dtype = si32} : () -> si32
    tile.yield(%3, %4) : (f32, si32) -> ()
  }) : (tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xsi32>)
  func.return(%1, %2) : (tensor<4xf32>, tensor<4xsi32>) -> ()
}";
    // As `tensor.argmax`: one operation for each of the 32 elements, the index it yields computing
    // nothing. 128 bytes of scores, 16 of maxima and 16 of indices.
    assert_eq!(costs(text), [cost(32, 128, 16 + 16)]);
}

#[test]
fn a_call_costs_what_its_callee_does() {
    let text = format!(
        "tile.dispatch @product(%0: tensor<2x3xf32>, %1: tensor<3x4xf32>) -> tensor<2x4xf32> {{
  {PRODUCT_BLOCK}
  func.return(%2) : (tensor<2x4xf32>) -> ()
}}
func.func @f(%0: tensor<2x3xf32>, %1: tensor<3x4xf32>) -> tensor<2x4xf32> {{
  %2 = host.dispatch(%0, %1) {{callee = @product}} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  func.return(%2) : (tensor<2x4xf32>) -> ()
}}"
    );
    let module = Module::from_text(&text).expect("the case parses");
    let func = module.func("f").expect("the case has @f");
    let call = func.body().insts().next().expect("@f calls the dispatch");
    assert_eq!(module.cost(call), cost(2 * 4 * 3 * 2, 24 + 48, 32));
}
