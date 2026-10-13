//! Building `tile.block`s: their attributes, and their bodies through [`Body`].

use smallvec::SmallVec;

use crate::dialects::arith::{self, Literal};
use crate::dialects::tile::ir::{Block, Fold, Index, Yield};
use crate::dialects::tile::{Iter, IterKind, Read};
use crate::host::HostScalar;
use crate::ir::{
    AccessExpr, Constraint, Created, DType, Dim, Format, InOp, IndexExpr, Operands, Value, Values,
    MAX_ITERS,
};
use crate::rewrite::{Builder, Rewriter};

impl Block {
    /// A block over `iters` with these reads and folds, and no constraints.
    #[track_caller]
    pub fn new(iters: Vec<Iter>, reads: Vec<Read>, folds: Vec<Fold>) -> Self {
        assert!(
            iters.len() <= MAX_ITERS,
            "a block has at most {MAX_ITERS} iterators, not {}",
            iters.len()
        );
        Self {
            iters: iters.into_iter().collect(),
            reads: reads.into(),
            folds: folds.into(),
            constraints: SmallVec::new(),
        }
    }

    /// This block restricted to the points where `constraints` hold.
    pub fn restrict(self, constraints: Vec<Constraint>) -> Self {
        Self {
            constraints: constraints.into(),
            ..self
        }
    }

    /// Creates the block on `ins` through `rewriter`. `body(body, args)` builds the body on the
    /// read values `args` and returns the values to yield. Returns the block's results.
    pub fn build(
        self,
        rewriter: &mut Rewriter<'_>,
        ins: Vec<Value>,
        body: impl FnOnce(&mut Body<'_, '_>, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let mut body = Some(body);
        rewriter.create_with_regions(self, ins, |_, builder, args| {
            let body = body.take().expect("a block has one region");
            let yields = body(&mut Body { builder }, args);
            builder.create(Yield, yields);
        })
    }
}

/// The operands of a block and its reads of them, each value once among the operands.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub values: Vec<Value>,
    pub reads: Vec<Read>,
}

impl Inputs {
    /// Inputs over the operands `values`, with no reads yet.
    pub fn new(values: Vec<Value>) -> Self {
        Self {
            values,
            reads: Vec::new(),
        }
    }

    /// The operand that holds `value`, added at the end when it is new.
    pub fn operand(&mut self, value: Value) -> usize {
        match self.values.iter().position(|&v| v == value) {
            Some(i) => i,
            None => {
                self.values.push(value);
                self.values.len() - 1
            }
        }
    }

    /// Reads `value` at `coords`, and returns the index of the read.
    pub fn read(
        &mut self,
        value: Value,
        coords: impl IntoIterator<Item = impl Into<AccessExpr>>,
    ) -> usize {
        let operand = self.operand(value);
        self.push(Read::new(operand, coords))
    }

    /// Adds `read`, whose operands are already among these, and returns its index.
    pub fn push(&mut self, read: Read) -> usize {
        self.reads.push(read);
        self.reads.len() - 1
    }
}

/// One parallel iterator over each of `dims`, the first ones of a block, with the expressions
/// that name them.
pub fn par_iters(dims: &[Dim]) -> (Vec<Iter>, Vec<IndexExpr>) {
    let mut iters = Vec::with_capacity(dims.len());
    for dim in dims {
        push_iter(&mut iters, IterKind::Par, dim.clone());
    }
    let mut coords = Vec::with_capacity(dims.len());
    for position in 0..dims.len() {
        coords.push(IndexExpr::iter(position));
    }
    (iters, coords)
}

/// Adds an iterator of `kind` over `extent` after `iters`, and gives the expression that names
/// it.
pub fn push_iter(iters: &mut Vec<Iter>, kind: IterKind, extent: Dim) -> IndexExpr {
    iters.push(Iter::new(kind, extent));
    IndexExpr::iter(iters.len() - 1)
}

/// The body of a block while it is built: arith ops and `tile.index` values. `create` creates an
/// op, and the other methods compute a value with several; an op its type rule rejects is a bug
/// in the caller, and panics.
pub struct Body<'b, 'm> {
    pub builder: &'b mut Builder<'m>,
}

impl Body<'_, '_> {
    /// The dtype of the scalar `value`.
    pub fn dtype(&self, value: Value) -> DType {
        self.builder
            .module
            .ty(value)
            .as_scalar()
            .expect("a block body computes on scalars")
    }

    /// The constant `value`.
    #[track_caller]
    pub fn lit(&mut self, value: HostScalar) -> Value {
        let value = Literal::new(value).expect("a scalar is a literal");
        self.builder.create(arith::Constant { value }, ())
    }

    /// The float constant `value` of `dtype`, rounded to it.
    #[track_caller]
    pub fn float(&mut self, dtype: DType, value: f64) -> Value {
        let scalar = HostScalar::from_f64(dtype, value).expect("a float dtype");
        self.lit(scalar)
    }

    /// The integer constant `value` of `dtype`.
    #[track_caller]
    pub fn int(&mut self, dtype: DType, value: i128) -> Value {
        let scalar = HostScalar::int(dtype, value).expect("an integer dtype");
        self.lit(scalar)
    }

    /// Creates `op` on `operands`, and gives its results.
    #[track_caller]
    pub fn create<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
    ) -> Created<T> {
        self.builder.create(op, operands)
    }

    /// `value` converted to `to`, or `value` itself when it is a `to` already.
    #[track_caller]
    pub fn cast(&mut self, value: Value, to: DType) -> Value {
        if self.dtype(value) == to {
            return value;
        }
        self.builder.create(arith::Cast { to }, value)
    }

    /// The bits of `value` as a `to`, of equal width, or `value` itself when it is a `to`
    /// already.
    #[track_caller]
    pub fn bitcast(&mut self, value: Value, to: DType) -> Value {
        if self.dtype(value) == to {
            return value;
        }
        self.builder.create(arith::Bitcast { to }, value)
    }

    /// The value of `expr` at the current point, as a `dtype`.
    #[track_caller]
    pub fn index(&mut self, expr: IndexExpr, dtype: DType) -> Value {
        self.builder.create(Index { expr, dtype }, ())
    }

    /// Whether `value` is NaN: whether it differs from itself.
    pub fn is_nan(&mut self, value: Value) -> Value {
        self.create(arith::NotEqual, (value, value))
    }

    /// `value` rounded to an integer, ties to even, in its own float dtype: adding and subtracting
    /// `2^p`, `p` the dtype's fraction bits, rounds a magnitude below it, and one at or above it is
    /// whole already.
    pub fn round_even(&mut self, value: Value) -> Value {
        let dtype = self.dtype(value);
        let significand_bits = dtype
            .significand_bits()
            .unwrap_or_else(|| panic!("round_even rounds a float, not {dtype}"));
        let magnitude = self.create(arith::Abs, value);
        let first_whole = self.float(dtype, 2f64.powi(significand_bits as i32 - 1));
        let shifted = self.create(arith::Add, (magnitude, first_whole));
        let below_whole = self.create(arith::Sub, (shifted, first_whole));
        let whole = self.create(arith::GreaterEqual, (magnitude, first_whole));
        let rounded = self.create(arith::Select, (whole, magnitude, below_whole));
        let negative = self.sign_bit(value);
        let negated = self.create(arith::Neg, rounded);
        self.create(arith::Select, (negative, negated, rounded))
    }

    /// The largest integer at or below `value`, for `0 <= value < 2^p`, where `round_even` is
    /// exact.
    pub fn floor_nonnegative(&mut self, value: Value) -> Value {
        let dtype = self.dtype(value);
        let rounded = self.round_even(value);
        let above = self.create(arith::Greater, (rounded, value));
        let one = self.float(dtype, 1.0);
        let lower = self.create(arith::Sub, (rounded, one));
        self.create(arith::Select, (above, lower, rounded))
    }

    /// Whether the sign bit of the float `value` is set, -0.0 and negative NaNs included.
    pub fn sign_bit(&mut self, value: Value) -> Value {
        let width = self.dtype(value).bits();
        let unsigned = DType::unsigned(width).expect("an integer has the float's width");
        let bits = self.bitcast(value, unsigned);
        let shift = self.int(unsigned, width as i128 - 1);
        let top = self.create(arith::ShiftRightLogical, (bits, shift));
        let one = self.int(unsigned, 1);
        self.create(arith::Equal, (top, one))
    }

    /// The exponent field of the f64 `value` minus its bias: `floor(log2(value))` for a normal
    /// `value > 0`.
    pub fn exponent_f64(&mut self, value: Value) -> Value {
        let bits = self.bitcast(value, DType::Ui64);
        let shift = self.int(DType::Ui64, i128::from(Format::F64.man_bits));
        let shifted = self.create(arith::ShiftRightLogical, (bits, shift));
        let mask = self.int(DType::Ui64, (1 << Format::F64.exp_bits) - 1);
        let field = self.create(arith::And, (shifted, mask));
        let exponent = self.cast(field, DType::Si64);
        let bias = self.int(DType::Si64, i128::from(Format::F64.bias));
        self.create(arith::Sub, (exponent, bias))
    }

    /// `2^exponent` as an f64, for an si64 `exponent` of a normal f64.
    pub fn pow2_f64(&mut self, exponent: Value) -> Value {
        let bias = self.int(DType::Si64, i128::from(Format::F64.bias));
        let biased = self.create(arith::Add, (exponent, bias));
        let field = self.cast(biased, DType::Ui64);
        let shift = self.int(DType::Ui64, i128::from(Format::F64.man_bits));
        let bits = self.create(arith::ShiftLeft, (field, shift));
        self.bitcast(bits, DType::F64)
    }
}
