//! Winograd's minimal filtering for 3x3 convolutions with stride 1.

use anyhow::Result;
use tinyvec::ArrayVec;

use crate::dialects::arith::{self, ScalarBuilder};
use crate::dialects::tile::{
    par_iters, push_iter, Block, BlockOp, CombineKind, Coords, Fold, Index, IterKind, Iters,
    Numerics, Read, Source, Write,
};
use crate::ir::{AccessExpr, DType, Dim, IndexExpr, Module, Value, MAX_ITERS};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{
    pattern, Build, Builder, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting,
};

/// A 3x3 convolution with stride 1, `y[p, q, k] = sum x[p + i - pad, q + j - pad, c] w[i, j, c,
/// k]` over the taps `i`, `j` and the channels `c`, as Winograd's F(2x2, 3x3) computes it. Each
/// 2x2 tile of the output comes from a 4x4 tile of the input. The weight transform `U = G w G^T`
/// and the input transform `V = B^T d B` give 16 matrices each, in one block per row of `G` or
/// `B^T`. One contraction over the input channels, `M[r, s, b, k] = sum V_b[r, s, c] U_b[c, k]`,
/// picks each pair of matrices from lists by `b`. The output transform `Y = A^T M A` gives each
/// output element at the convolution's coordinates and in its layout.
#[pattern]
struct ConvToWinograd {
    numerics: Numerics,
}

/// The output iterator and the tap iterator of an axis of a convolution, by position.
type Window = (usize, usize);

/// The coordinates of a convolution that a block states: which read is the padded input and which
/// the weight, the iterators of the output's rows and columns, their taps, the input and the
/// output channels, by position.
struct Conv {
    input: usize,
    weight: usize,
    rows: Window,
    columns: Window,
    channels: usize,
    outputs: usize,
}

impl Conv {
    /// The convolution that `block` computes, with the arguments of its body `args` and the
    /// value it yields `yielded`.
    fn of(block: &Block, module: &Module, args: &[Value], yielded: &[Value]) -> Option<Conv> {
        let (input, weight) = summed_product(block, module, args, yielded)?;
        let padded = &block.reads[input];
        if padded.pad.map(|pad| pad.0) != Some(0.0) || block.reads[weight].pad.is_some() {
            return None;
        }
        for read in &block.reads {
            if !matches!(read.source, Source::Operand(_)) {
                return None;
            }
        }
        let (rows, columns, channels) = padded_iters(block, padded)?;
        let outputs = output_channels(block, [rows.0, rows.1, columns.0, columns.1, channels])?;
        let conv = Conv {
            input,
            weight,
            rows,
            columns,
            channels,
            outputs,
        };
        conv.coords_fit(block).then_some(conv)
    }

    /// Whether the weight of `block` takes the taps, the channels and the output channels, and
    /// its output its rows, columns and output channels.
    fn coords_fit(&self, block: &Block) -> bool {
        for coord in &block.reads[self.weight].coords {
            for position in coord.iters() {
                if position == self.rows.0 || position == self.columns.0 {
                    return false;
                }
            }
        }
        let written = [self.rows.0, self.columns.0, self.outputs];
        for coord in &block.folds[0].writes[0].coords {
            let iters = coord.iters();
            let named = coord.as_iter();
            let zero = iters.is_empty() && constant(coord) == Some(0);
            if !zero && !named.is_some_and(|position| written.contains(&position)) {
                return false;
            }
        }
        true
    }

    /// Builds Winograd's blocks for the convolution of the block `attrs` on its input `input` and
    /// weight `weight` through `builder`: the weight and the input transforms, the products, and
    /// the output transform. Gives the output transform's results.
    fn build(
        &self,
        builder: &mut impl Build,
        attrs: &Block,
        (input, weight): (Value, Value),
    ) -> Vec<Value> {
        let extent = |position: usize| attrs.iters[position].extent.clone();
        let (channels, outputs) = (extent(self.channels), extent(self.outputs));
        let half = |position: usize| {
            let points = attrs.iters[position].extent.as_static().expect("static");
            Dim::Static(points.div_ceil(2))
        };
        let (tile_rows, tile_columns) = (half(self.rows.0), half(self.columns.0));
        let (pair, _) = par_iters(&[channels.clone(), outputs.clone()]);
        let transformed_weights = self.build_weight_transform(builder, attrs, weight, pair);
        let (tiles, _) = par_iters(&[tile_rows.clone(), tile_columns.clone(), channels.clone()]);
        let transformed_inputs = self.build_input_transform(builder, attrs, input, tiles);
        let (mut matrices, _) = par_iters(&[Dim::Static(16), tile_rows, tile_columns, outputs]);
        push_iter(&mut matrices, IterKind::Red, channels);
        let products = Self::build_products(
            builder,
            matrices,
            [transformed_inputs, transformed_weights].concat(),
        );
        self.build_output_transform(builder, attrs, products)
    }

    /// U_b[c, k]: the weight `weight` of the block `attrs` at each pair of taps, transformed, one
    /// row of `G` per block over `pair`, the input and the output channels.
    fn build_weight_transform(
        &self,
        builder: &mut impl Build,
        attrs: &Block,
        weight: Value,
        pair: Iters,
    ) -> Vec<Value> {
        let weight_coords = &attrs.reads[self.weight].coords;
        let mut weight_reads = Vec::with_capacity(9);
        for tap_row in 0..3 {
            for tap_column in 0..3 {
                let coords = substitute(weight_coords, &|position| match position {
                    _ if position == self.rows.1 => Some(IndexExpr::constant(tap_row)),
                    _ if position == self.columns.1 => Some(IndexExpr::constant(tap_column)),
                    _ if position == self.channels => Some(IndexExpr::iter(0)),
                    _ if position == self.outputs => Some(IndexExpr::iter(1)),
                    _ => None,
                });
                weight_reads.push(Read::new(0, coords));
            }
        }
        let mut transformed_weights = Vec::with_capacity(16);
        for row in 0..4 {
            let block = Block::new(pair.clone(), weight_reads.clone(), plain_folds(2, 4));
            transformed_weights.extend(block.build(builder, vec![weight], |body, taps| {
                weight_row(body, taps, row)
            }));
        }
        transformed_weights
    }

    /// V_b[r, s, c]: the input `input` of the block `attrs` at each 4x4 tile, transformed, one
    /// row of `B^T` per block over `tiles`, the rows and the columns of tiles and the input
    /// channels.
    fn build_input_transform(
        &self,
        builder: &mut impl Build,
        attrs: &Block,
        input: Value,
        tiles: Iters,
    ) -> Vec<Value> {
        let input_coords = &attrs.reads[self.input].coords;
        let pad = attrs.reads[self.input].pad;
        let mut input_reads = Vec::with_capacity(16);
        for row in 0..4 {
            for column in 0..4 {
                let coords = substitute(input_coords, &|position| match position {
                    _ if position == self.rows.0 => {
                        Some(IndexExpr::iter(0) * IndexExpr::constant(2))
                    }
                    _ if position == self.rows.1 => Some(IndexExpr::constant(row)),
                    _ if position == self.columns.0 => {
                        Some(IndexExpr::iter(1) * IndexExpr::constant(2))
                    }
                    _ if position == self.columns.1 => Some(IndexExpr::constant(column)),
                    _ if position == self.channels => Some(IndexExpr::iter(2)),
                    _ => None,
                });
                input_reads.push(Read {
                    pad,
                    ..Read::new(0, coords)
                });
            }
        }
        let mut transformed_inputs = Vec::with_capacity(16);
        for row in 0..4 {
            let block = Block::new(tiles.clone(), input_reads.clone(), plain_folds(3, 4));
            transformed_inputs.extend(block.build(builder, vec![input], |body, elements| {
                input_row(body, elements, row)
            }));
        }
        transformed_inputs
    }

    /// M[r, s, b, k] = sum over c of V_b[r, s, c] U_b[c, k], over `iters`, `b`, the rows and the
    /// columns of tiles, the output and the input channels, each pair of matrices picked from the
    /// lists `factors`, the transformed inputs then the transformed weights, by `b`.
    fn build_products(builder: &mut impl Build, iters: Iters, factors: Vec<Value>) -> Vec<Value> {
        let picked = |first: usize, coords: Vec<IndexExpr>| Read {
            source: Source::List {
                first,
                count: 16,
                pick: Box::new(IndexExpr::iter(0).into()),
            },
            ..Read::new(0, coords)
        };
        let reads = vec![
            picked(
                0,
                vec![IndexExpr::iter(1), IndexExpr::iter(2), IndexExpr::iter(4)],
            ),
            picked(16, vec![IndexExpr::iter(4), IndexExpr::iter(3)]),
        ];
        let folds = vec![Fold::new(
            CombineKind::Sum,
            vec![Write::new([
                IndexExpr::iter(1),
                IndexExpr::iter(2),
                IndexExpr::iter(0),
                IndexExpr::iter(3),
            ])],
        )];
        Block::new(iters, reads, folds).build(builder, factors, |body, elements| {
            vec![body.create(arith::Mul, (elements[0], elements[1]))]
        })
    }

    /// y[p, q, k] from the 16 matrices `products` of its tile, at the coordinates of the
    /// convolution of the block `attrs`.
    fn build_output_transform(
        &self,
        builder: &mut impl Build,
        attrs: &Block,
        products: Vec<Value>,
    ) -> Vec<Value> {
        let mut parallel = ArrayVec::<[usize; MAX_ITERS]>::new();
        for (position, iter) in attrs.iters.iter().enumerate() {
            if iter.kind == IterKind::Par {
                parallel.push(position);
            }
        }
        let renumber = |position: usize| parallel.iter().position(|&kept| kept == position);
        let (row, column, output) = (
            renumber(self.rows.0).expect("the rows are parallel"),
            renumber(self.columns.0).expect("the columns are parallel"),
            renumber(self.outputs).expect("the output channels are parallel"),
        );
        let mut product_reads = Vec::with_capacity(16);
        for matrix in 0..16 {
            let tile_row = IndexExpr::floordiv(IndexExpr::iter(row), IndexExpr::constant(2));
            let tile_column = IndexExpr::floordiv(IndexExpr::iter(column), IndexExpr::constant(2));
            product_reads.push(Read::new(
                0,
                [
                    tile_row,
                    tile_column,
                    IndexExpr::constant(matrix),
                    IndexExpr::iter(output),
                ],
            ));
        }
        let mut iters = Iters::new();
        for &position in &parallel {
            iters.push(attrs.iters[position].clone());
        }
        let write = Write::new(substitute(&attrs.folds[0].writes[0].coords, &|position| {
            renumber(position).map(IndexExpr::iter)
        }));
        Block::new(iters, product_reads, vec![Fold::plain(write)]).build(
            builder,
            products,
            |body, matrices| vec![output_transform(body, matrices, row, column)],
        )
    }
}

/// The positions of the padded read of `block` and of its other read, where its one fold sums,
/// from no init and into no operand, the product of the two f32 arguments `args` of its body,
/// which yields it as `yielded`, under no constraint.
fn summed_product(
    block: &Block,
    module: &Module,
    args: &[Value],
    yielded: &[Value],
) -> Option<(usize, usize)> {
    let [fold] = block.folds.as_slice() else {
        return None;
    };
    let [write] = fold.writes.as_slice() else {
        return None;
    };
    if fold.combine != CombineKind::Sum || write.into.is_some() || !fold.init.is_empty() {
        return None;
    }
    if !block.constraints.is_empty() || block.reads.len() != 2 || args.len() != 2 {
        return None;
    }
    let &[value] = yielded else {
        return None;
    };
    let product = module.value_ref(value).defining_op::<arith::Mul>()?;
    let factors = [product.lhs().id(), product.rhs().id()];
    if !factors.contains(&args[0]) || !factors.contains(&args[1]) {
        return None;
    }
    for &arg in args {
        if module.ty(arg).as_scalar() != Some(DType::F32) {
            return None;
        }
    }
    let input = block.reads.iter().position(|read| read.pad.is_some())?;
    Some((input, 1 - input))
}

/// The output and tap iterators of the rows and of the columns that the coordinates of the
/// padded read `padded` of `block` take, and its one reduced channel iterator, by position.
fn padded_iters(block: &Block, padded: &Read) -> Option<(Window, Window, usize)> {
    let mut spatial = Vec::new();
    let mut channel_position = None;
    for coord in &padded.coords {
        let iters: Vec<usize> = coord.iters().into_iter().collect();
        match iters.as_slice() {
            [] => {}
            &[channel] => {
                let kind = block.iters[channel].kind;
                let plain = coord.coefficient(channel) == Some(1) && constant(coord) == Some(0);
                if kind != IterKind::Red || !plain || channel_position.replace(channel).is_some() {
                    return None;
                }
            }
            &[first, second] => spatial.push(window(block, coord, first, second)?),
            _ => return None,
        }
    }
    let &[rows, columns] = spatial.as_slice() else {
        return None;
    };
    Some((rows, columns, channel_position?))
}

/// The output channels: the one iterator of `block` outside `taken`, by position, where it is
/// parallel and every iterator of `block` has a static extent.
fn output_channels(block: &Block, taken: [usize; 5]) -> Option<usize> {
    let mut output_position = None;
    for (position, iter) in block.iters.iter().enumerate() {
        iter.extent.as_static()?;
        if taken.contains(&position) {
            continue;
        }
        if iter.kind != IterKind::Par || output_position.replace(position).is_some() {
            return None;
        }
    }
    output_position
}

/// The output and the tap iterator, by position, of a coordinate `coord` that takes `first` and
/// `second`: one of them parallel, the other reduced over 3 points, both with step 1.
fn window(block: &Block, coord: &AccessExpr, first: usize, second: usize) -> Option<Window> {
    let (output, tap) = match block.iters[first].kind {
        IterKind::Par => (first, second),
        _ => (second, first),
    };
    let taps = block.iters[tap].extent.as_static()?;
    let steps = coord.coefficient(output) == Some(1) && coord.coefficient(tap) == Some(1);
    let kinds = block.iters[output].kind == IterKind::Par && block.iters[tap].kind == IterKind::Red;
    (steps && kinds && taps == 3).then_some((output, tap))
}

/// The constant term of `coord`.
fn constant(coord: &AccessExpr) -> Option<i64> {
    coord.bounds(&|_| Some((0, 0))).map(|(low, _)| low)
}

impl Pattern for ConvToWinograd {
    type Root = Block;

    fn rewrite(&self, block: BlockOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = rewriter.module();
        let body = block.body();
        let args: Vec<Value> = body.args().iter().map(|arg| arg.id()).collect();
        let yielded: Vec<Value> = body.yielded().iter().map(|value| value.id()).collect();
        require!(
            self.numerics.winograd,
            "the compilation does not allow Winograd's rounding"
        );
        require!(
            let Some(conv) = Conv::of(&block, module, &args, &yielded),
            "not a 3x3 convolution with stride 1 over a read padded with 0"
        );
        let attrs = block.attrs().clone();
        let operand = |read: usize| match attrs.reads[read].source {
            Source::Operand(operand) => block.ins()[operand].id(),
            Source::List { .. } => unreachable!("a convolution reads single operands"),
        };
        let (input, weight) = (operand(conv.input), operand(conv.weight));
        let result = conv.build(rewriter, &attrs, (input, weight));
        rewriter.replace_op(&block, result);
        Ok(())
    }
}

/// `count` folds that each write one yielded value at the first `rank` iterators.
fn plain_folds(rank: usize, count: usize) -> Vec<Fold> {
    let mut folds = Vec::with_capacity(count);
    for _ in 0..count {
        folds.push(Fold::plain(Write::new((0..rank).map(IndexExpr::iter))));
    }
    folds
}

/// `coords` with each iterator that `subst` maps replaced by its expression.
fn substitute(coords: &[AccessExpr], subst: &dyn Fn(usize) -> Option<IndexExpr>) -> Coords {
    let mut substituted = Coords::new();
    for coord in coords {
        let replaced = coord.substitute(&|position| subst(position).map(AccessExpr::from));
        substituted.push(replaced.canonical(&|_| None));
    }
    substituted
}

/// Row `row` of `G g G^T`, the 3x3 taps `g` row by row, with `G = [1 0 0; 1/2 1/2 1/2;
/// 1/2 -1/2 1/2; 0 0 1]`: 4 transformed weights.
fn weight_row(body: &mut ScalarBuilder<Builder>, taps: &[Value], row: usize) -> Vec<Value> {
    let half = body.float(DType::F32, 0.5);
    // Row `row` of `G` applied to each column of taps, then `G` to the 3 values.
    let combined = std::array::from_fn(|column| {
        let values = [taps[column], taps[3 + column], taps[6 + column]];
        spread_row(body, values, half, row)
    });
    spread(body, combined, half).to_vec()
}

/// Row `row` of `G [a b c]^T`.
fn spread_row(
    body: &mut ScalarBuilder<Builder>,
    [first, middle, last]: [Value; 3],
    half: Value,
    row: usize,
) -> Value {
    let combined = match row {
        0 => return first,
        3 => return last,
        1 => body.create(arith::Add, (first, middle)),
        _ => body.create(arith::Sub, (first, middle)),
    };
    let sum = body.create(arith::Add, (combined, last));
    body.create(arith::Mul, (sum, half))
}

/// `G [a b c]^T`: `[a, (a + b + c) / 2, (a - b + c) / 2, c]`.
fn spread(
    body: &mut ScalarBuilder<Builder>,
    [first, middle, last]: [Value; 3],
    half: Value,
) -> [Value; 4] {
    let ends = body.create(arith::Add, (first, last));
    let sum = body.create(arith::Add, (ends, middle));
    let difference = body.create(arith::Sub, (ends, middle));
    let half_sum = body.create(arith::Mul, (sum, half));
    let half_difference = body.create(arith::Mul, (difference, half));
    [first, half_sum, half_difference, last]
}

/// Row `row` of `B^T d B`, the 4x4 tile `d` row by row, with `B^T = [1 0 -1 0; 0 1 1 0;
/// 0 -1 1 0; 0 1 0 -1]`: 4 transformed inputs.
fn input_row(body: &mut ScalarBuilder<Builder>, tile: &[Value], row: usize) -> Vec<Value> {
    // Row `row` of `B^T` applied to each column of the tile, then `B^T` to the 4 values.
    let combined = std::array::from_fn(|column| {
        let values = [
            tile[column],
            tile[4 + column],
            tile[8 + column],
            tile[12 + column],
        ];
        difference_row(body, values, row)
    });
    differences(body, combined).to_vec()
}

/// Row `row` of `B^T [a b c d]^T`.
fn difference_row(
    body: &mut ScalarBuilder<Builder>,
    [first, second, third, fourth]: [Value; 4],
    row: usize,
) -> Value {
    match row {
        0 => body.create(arith::Sub, (first, third)),
        1 => body.create(arith::Add, (second, third)),
        2 => body.create(arith::Sub, (third, second)),
        _ => body.create(arith::Sub, (second, fourth)),
    }
}

/// `B^T [a b c d]^T`: `[a - c, b + c, c - b, b - d]`.
fn differences(
    body: &mut ScalarBuilder<Builder>,
    [first, second, third, fourth]: [Value; 4],
) -> [Value; 4] {
    [
        body.create(arith::Sub, (first, third)),
        body.create(arith::Add, (second, third)),
        body.create(arith::Sub, (third, second)),
        body.create(arith::Sub, (second, fourth)),
    ]
}

/// The element of `A^T M A` at the point, `M` the 16 matrices of its tile, row by row, with
/// `A^T = [1 1 1 0; 0 1 -1 -1]`: which of the tile's 2x2 elements it is comes from whether the
/// iterators `row` and `column` are even.
fn output_transform(
    body: &mut ScalarBuilder<Builder>,
    matrices: &[Value],
    row: usize,
    column: usize,
) -> Value {
    let zero = body.int(DType::Si64, 0);
    let [even_row, even_column] = [row, column].map(|position| {
        let parity = body.create(
            Index {
                expr: IndexExpr::modulo(IndexExpr::iter(position), IndexExpr::constant(2)),
                dtype: DType::Si64,
            },
            (),
        );
        body.create(arith::Equal, (parity, zero))
    });
    // `A^T` on each column of the matrices, as the row's parity picks it, then on that row.
    let picked = std::array::from_fn(|column_of| {
        let values = [
            matrices[column_of],
            matrices[4 + column_of],
            matrices[8 + column_of],
            matrices[12 + column_of],
        ];
        gather(body, values, even_row)
    });
    gather(body, picked, even_column)
}

/// The row of `A^T` that `even` picks applied to `[a b c d]`: `a + b + c` where it holds, else
/// `b - c - d`.
fn gather(
    body: &mut ScalarBuilder<Builder>,
    [first, second, third, fourth]: [Value; 4],
    even: Value,
) -> Value {
    let pair = body.create(arith::Add, (first, second));
    let first_row = body.create(arith::Add, (pair, third));
    let difference = body.create(arith::Sub, (second, third));
    let second_row = body.create(arith::Sub, (difference, fourth));
    body.create(arith::Select, (even, first_row, second_row))
}

/// Computes each 3x3 convolution with stride 1 by Winograd's F(2x2, 3x3) where the compilation
/// allows its rounding (`numerics = <winograd = true>`). It transforms the weight and the input
/// into 16 matrices each, runs one contraction over the input channels for all 16 pairs, and
/// transforms their products into the convolution's result. That takes 16 products per tile of
/// 2x2 outputs and channel pair instead of 36. A pipeline runs it only when asked to.
///
/// - A convolution here is a block whose one fold sums the products of a read padded with 0 and
///   a read of the weight, where two coordinates of the padded read are each an output iterator
///   plus a tap iterator of 3 points, with step 1.
/// - It convolves one image: its iterators of static extents are the rows, the columns, their
///   taps, the input channels and one more, the output channels, so a batch of several images,
///   whose iterator is one more, keeps its block.
/// - Its body multiplies two f32 elements. A convolution of another dtype keeps its block.
/// - The weight transform runs with the other blocks. It computes a value larger than the weight
///   it reads, so no pass moves it to load time.
/// - The result is the convolution's, at its coordinates and in its layout, so the blocks that run
///   on its elements still take it as their epilogue.
///
/// The sums are equal in real arithmetic, not bit for bit: the transforms round on their own.
#[pass(
    name = "winograd-conv",
    stats(rewritten = "convolutions computed by Winograd's F(2x2, 3x3)")
)]
#[derive(Default)]
pub struct WinogradConv {
    /// The rounding changes the compilation allows: the pass rewrites only where they name
    /// `winograd`.
    pub numerics: Numerics,
}

fn populate_winograd_conv_patterns(set: &mut PatternSet, numerics: Numerics) {
    set.add(ConvToWinograd { numerics });
}

impl Pass for WinogradConv {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_winograd_conv_patterns(&mut patterns, self.numerics);
        Rewrites {
            patterns,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::REWRITTEN)
    }
}
