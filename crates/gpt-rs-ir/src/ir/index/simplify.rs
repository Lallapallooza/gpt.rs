//! Simplification: the linear normal form of the affine parts of an expression, and the bounds of
//! its values.

use super::expr::{pop, Node};
use super::{gcd, BinaryOp, Expr, ReadKind, Root};
use crate::ir::Dim;

impl<R: ReadKind> Expr<R> {
    /// Whether the expression is affine in the iterators: no product of two subexpressions that
    /// both use iterators, and no divisor that uses them.
    pub fn is_affine(&self) -> bool {
        // Per subexpression: whether it uses iterators, and whether it is affine.
        let mut stack: Vec<(bool, bool)> = Vec::new();
        for node in &self.nodes {
            let operands = stack.split_off(stack.len() - node.arity());
            let uses = matches!(node, Node::Iter(_)) || operands.iter().any(|&(uses, _)| uses);
            let operands_affine = operands.iter().all(|&(_, affine)| affine);
            let affine = operands_affine
                && match node {
                    Node::Binary(BinaryOp::Mul) => !(operands[0].0 && operands[1].0),
                    Node::Binary(BinaryOp::FloorDiv | BinaryOp::Mod) => !operands[1].0,
                    _ => true,
                };
            stack.push((uses, affine));
        }
        pop(&mut stack).1
    }

    /// How many `floordiv` and `mod` the expression computes.
    pub fn divisions(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, Node::Binary(BinaryOp::FloorDiv | BinaryOp::Mod)))
            .count()
    }

    /// How many operations the expression computes: its arithmetic, `min` and `max`.
    pub fn operations(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, Node::Binary(_) | Node::Neg))
            .count()
    }

    /// The constant divisors of the `floordiv` and `mod` whose dividends use the iterator at
    /// `position`.
    pub fn divisors_of(&self, position: usize) -> Vec<i64> {
        let mut divisors = Vec::new();
        self.map(&mut |expr| {
            divisors.extend(divisor_using(&expr, position));
            expr
        });
        divisors
    }

    /// Folds constants and drops the identities of `+`, `-` and `*`. Keeps the value.
    pub fn simplify(&self) -> Self {
        self.map(&mut Self::fold_root)
    }

    /// The expression with its root folded, its operands folded already.
    fn fold_root(self) -> Self {
        let pure = |expr: &Self| !expr.has_reads();
        match self.root() {
            Root::Binary(op, lhs, rhs) => match (op, lhs.as_const(), rhs.as_const()) {
                (BinaryOp::FloorDiv | BinaryOp::Mod, Some(_), Some(divisor)) if divisor <= 0 => {
                    self
                }
                (_, Some(left), Some(right)) => {
                    Self::constant(op.apply(left, right).expect("a positive divisor"))
                }
                (BinaryOp::Add, Some(0), None) | (BinaryOp::Mul, Some(1), None) => rhs,
                (BinaryOp::Add | BinaryOp::Sub, None, Some(0))
                | (BinaryOp::Mul | BinaryOp::FloorDiv, None, Some(1)) => lhs,
                (BinaryOp::Sub, ..) if lhs == rhs && pure(&lhs) => Self::constant(0),
                (BinaryOp::Mul, Some(0), None) | (BinaryOp::Mul, None, Some(0))
                    if pure(&lhs) && pure(&rhs) =>
                {
                    Self::constant(0)
                }
                (BinaryOp::Mod, None, Some(1)) if pure(&lhs) => Self::constant(0),
                (BinaryOp::Min | BinaryOp::Max, ..) if lhs == rhs => lhs,
                _ => self,
            },
            Root::Neg(operand) => match operand.as_const() {
                Some(value) => Self::constant(value.wrapping_neg()),
                None => self,
            },
            _ => self,
        }
    }

    /// An interval that holds every value, given the interval of each iterator, or `None` when
    /// the expression uses a symbol or reads an operand, or an interval is unknown.
    pub fn bounds(&self, iter_range: &dyn Fn(usize) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
        let mut stack: Vec<Option<(i64, i64)>> = Vec::new();
        for node in &self.nodes {
            let operands = stack.split_off(stack.len() - node.arity());
            let bounds = match node {
                Node::Const(value) => Some((*value, *value)),
                Node::Iter(position) => iter_range(usize::from(*position)),
                Node::Sym(_) | Node::Read(..) => None,
                Node::Binary(op) => match (operands[0], operands[1]) {
                    (Some(lhs), Some(rhs)) => binary_bounds(*op, lhs, rhs),
                    _ => None,
                },
                Node::Neg => operands[0]
                    .and_then(|(low, high)| Some((high.checked_neg()?, low.checked_neg()?))),
            };
            stack.push(bounds);
        }
        pop(&mut stack)
    }

    /// The expression in canonical form, with the same value at every point. `extent` gives the
    /// extent of the iterator at a position. The canonical form:
    /// - writes affine parts as sums of terms with merged coefficients;
    /// - reduces `floordiv` and `mod` by the multiples of their divisor, and by remainders that
    ///   `extent` shows to be in range;
    /// - decides `min` and `max` where the ranges decide them.
    pub fn canonical(&self, extent: &dyn Fn(usize) -> Option<Dim>) -> Self {
        linear(self, extent).expr()
    }
}

/// The constant divisor of `expr` when it is a `floordiv` or `mod` whose dividend uses the
/// iterator at `position`.
fn divisor_using<R: ReadKind>(expr: &Expr<R>, position: usize) -> Option<i64> {
    let Root::Binary(BinaryOp::FloorDiv | BinaryOp::Mod, dividend, divisor) = expr.root() else {
        return None;
    };
    divisor.as_const().filter(|_| dividend.uses_iter(position))
}

/// The interval of `lhs op rhs` for operands in the intervals `lhs` and `rhs`.
fn binary_bounds(op: BinaryOp, lhs: (i64, i64), rhs: (i64, i64)) -> Option<(i64, i64)> {
    let ((lhs_low, lhs_high), (rhs_low, rhs_high)) = (lhs, rhs);
    match op {
        BinaryOp::Add => Some((
            lhs_low.checked_add(rhs_low)?,
            lhs_high.checked_add(rhs_high)?,
        )),
        BinaryOp::Sub => Some((
            lhs_low.checked_sub(rhs_high)?,
            lhs_high.checked_sub(rhs_low)?,
        )),
        BinaryOp::Mul => {
            let corners = [
                lhs_low.checked_mul(rhs_low)?,
                lhs_low.checked_mul(rhs_high)?,
                lhs_high.checked_mul(rhs_low)?,
                lhs_high.checked_mul(rhs_high)?,
            ];
            Some((*corners.iter().min()?, *corners.iter().max()?))
        }
        BinaryOp::FloorDiv => {
            if rhs_low <= 0 {
                return None;
            }
            let corners = [
                lhs_low.div_euclid(rhs_low),
                lhs_low.div_euclid(rhs_high),
                lhs_high.div_euclid(rhs_low),
                lhs_high.div_euclid(rhs_high),
            ];
            Some((*corners.iter().min()?, *corners.iter().max()?))
        }
        BinaryOp::Mod => {
            if rhs_low <= 0 {
                return None;
            }
            if rhs_low == rhs_high && lhs_low >= 0 && lhs_high < rhs_low {
                return Some((lhs_low, lhs_high));
            }
            Some((0, rhs_high - 1))
        }
        BinaryOp::Min => Some((lhs_low.min(rhs_low), lhs_high.min(rhs_high))),
        BinaryOp::Max => Some((lhs_low.max(rhs_low), lhs_high.max(rhs_high))),
    }
}

/// A sum of terms with integer coefficients, plus a constant: the normal form of affine parts of
/// expressions.
#[derive(Clone)]
struct Linear<R> {
    terms: Vec<(Expr<R>, i64)>,
    constant: i64,
}

impl<R: ReadKind> Linear<R> {
    fn constant(value: i64) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }

    fn atom(expr: Expr<R>) -> Self {
        Self {
            terms: vec![(expr, 1)],
            constant: 0,
        }
    }

    fn as_constant(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.constant)
    }

    fn add(mut self, other: Linear<R>, sign: i64) -> Self {
        for (atom, coefficient) in other.terms {
            match self
                .terms
                .iter_mut()
                .find(|(existing_atom, _)| *existing_atom == atom)
            {
                Some((_, existing)) => {
                    *existing = existing.wrapping_add(coefficient.wrapping_mul(sign))
                }
                None => self.terms.push((atom, coefficient.wrapping_mul(sign))),
            }
        }
        self.terms.retain(|&(_, coefficient)| coefficient != 0);
        self.constant = self
            .constant
            .wrapping_add(other.constant.wrapping_mul(sign));
        self
    }

    fn scale(mut self, factor: i64) -> Self {
        if factor == 0 {
            return Self::constant(0);
        }
        for (_, coefficient) in &mut self.terms {
            *coefficient = coefficient.wrapping_mul(factor);
        }
        self.constant = self.constant.wrapping_mul(factor);
        self
    }

    /// The expression: the terms with positive coefficients in order, then the negative ones
    /// subtracted, then the constant.
    fn expr(&self) -> Expr<R> {
        let scaled = |atom: &Expr<R>, coefficient: i64| {
            if coefficient == 1 {
                atom.clone()
            } else {
                atom.clone() * Expr::constant(coefficient)
            }
        };
        let mut out: Option<Expr<R>> = None;
        for (atom, coefficient) in &self.terms {
            if *coefficient > 0 {
                let term = scaled(atom, *coefficient);
                out = Some(match out {
                    None => term,
                    Some(sum) => sum + term,
                });
            }
        }
        for (atom, coefficient) in &self.terms {
            if *coefficient < 0 {
                let term = scaled(atom, coefficient.wrapping_neg());
                out = Some(match out {
                    None => -term,
                    Some(sum) => sum - term,
                });
            }
        }
        match (out, self.constant) {
            (None, constant) => Expr::constant(constant),
            (Some(sum), 0) => sum,
            (Some(sum), constant) if constant > 0 => sum + Expr::constant(constant),
            (Some(sum), constant) => sum - Expr::constant(constant.wrapping_neg()),
        }
    }
}

/// The interval of `expr` when the iterators range over their extents.
fn range_of<R: ReadKind>(
    expr: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Option<(i64, i64)> {
    expr.bounds(&|position| match extent(position)? {
        Dim::Static(size) if size > 0 => Some((0, size as i64 - 1)),
        _ => None,
    })
}

/// Whether `0 <= remainder < divisor` at every point: by intervals, or because `remainder` is an
/// iterator whose extent is `divisor`.
fn below<R: ReadKind>(
    remainder: &Linear<R>,
    divisor: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> bool {
    if let (Some((low, high)), Some(constant_divisor)) =
        (range_of(&remainder.expr(), extent), divisor.as_const())
    {
        return low >= 0 && high < constant_divisor;
    }
    let [(atom, 1)] = remainder.terms.as_slice() else {
        return false;
    };
    let Some(position) = atom.as_iter().filter(|_| remainder.constant == 0) else {
        return false;
    };
    extent(position).is_some_and(|dim| Expr::dim(&dim) == *divisor)
}

/// `dividend` split by `divisor`: the quotient of the terms that are multiples of `divisor`, and
/// the rest.
fn split<R: ReadKind>(dividend: Linear<R>, divisor: &Expr<R>) -> (Linear<R>, Linear<R>) {
    let mut quotient = Linear::constant(0);
    let mut rest = Linear::constant(0);
    let constant_divisor = divisor.as_const();
    for (atom, coefficient) in dividend.terms {
        if let Some(constant) = constant_divisor {
            if coefficient % constant == 0 {
                quotient.terms.push((atom, coefficient / constant));
            } else {
                rest.terms.push((atom, coefficient));
            }
            continue;
        }
        if atom == *divisor {
            quotient.constant += coefficient;
            continue;
        }
        match atom.root() {
            Root::Binary(BinaryOp::Mul, lhs, rhs) if rhs == *divisor => {
                quotient.terms.push((lhs, coefficient))
            }
            Root::Binary(BinaryOp::Mul, lhs, rhs) if lhs == *divisor => {
                quotient.terms.push((rhs, coefficient))
            }
            _ => rest.terms.push((atom, coefficient)),
        }
    }
    match constant_divisor {
        Some(constant) => {
            quotient.constant += dividend.constant.div_euclid(constant);
            rest.constant = dividend.constant.rem_euclid(constant);
        }
        None => rest.constant = dividend.constant,
    }
    (quotient, rest)
}

/// The canonical linear form of `expr`.
fn linear<R: ReadKind>(expr: &Expr<R>, extent: &dyn Fn(usize) -> Option<Dim>) -> Linear<R> {
    match expr.root() {
        Root::Const(value) => Linear::constant(value),
        Root::Iter(_) | Root::Sym(_) => Linear::atom(expr.clone()),
        Root::Read(read, coords) => {
            let mut canonical = Vec::with_capacity(coords.len());
            for coord in &coords {
                canonical.push(coord.canonical(extent));
            }
            Linear::atom(Expr::on(Node::Read(read, coords.len() as u8), canonical))
        }
        Root::Neg(operand) => linear(&operand, extent).scale(-1),
        Root::Binary(BinaryOp::Add, lhs, rhs) => linear(&lhs, extent).add(linear(&rhs, extent), 1),
        Root::Binary(BinaryOp::Sub, lhs, rhs) => linear(&lhs, extent).add(linear(&rhs, extent), -1),
        Root::Binary(BinaryOp::Mul, lhs, rhs) => {
            let (left, right) = (linear(&lhs, extent), linear(&rhs, extent));
            match (left.as_constant(), right.as_constant()) {
                (_, Some(factor)) => left.scale(factor),
                (Some(factor), _) => right.scale(factor),
                _ => Linear::atom(left.expr() * right.expr()),
            }
        }
        Root::Binary(BinaryOp::FloorDiv, lhs, rhs) => linear_floordiv(&lhs, &rhs, extent),
        Root::Binary(BinaryOp::Mod, lhs, rhs) => {
            let divisor = rhs.canonical(extent);
            if divisor.as_const().is_some_and(|value| value <= 0) {
                return Linear::atom(Expr::modulo(lhs.canonical(extent), divisor));
            }
            let (_, rest) = split(linear(&lhs, extent), &divisor);
            if let (Some(remainder), Some(constant_divisor)) =
                (rest.as_constant(), divisor.as_const())
            {
                return Linear::constant(remainder.rem_euclid(constant_divisor));
            }
            if below(&rest, &divisor, extent) {
                return rest;
            }
            if let Some((factor, reduced, high, low)) =
                factor_out(&rest, divisor.as_const(), extent)
            {
                let inner = Expr::modulo(high.expr(), Expr::constant(reduced));
                return linear(&inner, extent).scale(factor).add(low, 1);
            }
            Linear::atom(Expr::modulo(rest.expr(), divisor))
        }
        Root::Binary(op @ (BinaryOp::Min | BinaryOp::Max), lhs, rhs) => {
            linear_extreme(op == BinaryOp::Min, &lhs, &rhs, extent)
        }
    }
}

/// The canonical linear form of `dividend floordiv divisor`.
fn linear_floordiv<R: ReadKind>(
    dividend: &Expr<R>,
    divisor: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Linear<R> {
    let canonical_divisor = divisor.canonical(extent);
    // `(x floordiv c1) floordiv c2` is `x floordiv (c1 * c2)`.
    if let (Root::Binary(BinaryOp::FloorDiv, inner_dividend, inner_divisor), Some(outer)) =
        (dividend.root(), canonical_divisor.as_const())
    {
        if let Some(inner) = inner_divisor
            .as_const()
            .filter(|&value| value > 0 && outer > 0)
        {
            let merged = Expr::floordiv(inner_dividend, Expr::constant(inner * outer));
            return linear(&merged, extent);
        }
    }
    if canonical_divisor.as_const().is_some_and(|value| value <= 0) {
        return Linear::atom(Expr::floordiv(
            dividend.canonical(extent),
            canonical_divisor,
        ));
    }
    let (quotient, rest) = split(linear(dividend, extent), &canonical_divisor);
    if let (Some(remainder), Some(constant_divisor)) =
        (rest.as_constant(), canonical_divisor.as_const())
    {
        return quotient.add(Linear::constant(remainder.div_euclid(constant_divisor)), 1);
    }
    if below(&rest, &canonical_divisor, extent) {
        return quotient;
    }
    if let Some((_, reduced, high, _)) = factor_out(&rest, canonical_divisor.as_const(), extent) {
        let inner = Expr::floordiv(high.expr(), Expr::constant(reduced));
        return quotient.add(linear(&inner, extent), 1);
    }
    quotient.add(
        Linear::atom(Expr::floordiv(rest.expr(), canonical_divisor)),
        1,
    )
}

/// `dividend` as `factor * high + low`, where `factor` divides `divisor` and `0 <= low < factor`
/// at every point, with the largest such factor above 1: then `dividend floordiv divisor` is
/// `high floordiv (divisor / factor)`, and `dividend mod divisor` is
/// `factor * (high mod (divisor / factor)) + low`. The factors tried are those `divisor` shares
/// with a coefficient of `dividend`. Returns `factor`, `divisor / factor`, `high` and `low`.
fn factor_out<R: ReadKind>(
    dividend: &Linear<R>,
    constant_divisor: Option<i64>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Option<(i64, i64, Linear<R>, Linear<R>)> {
    let divisor = constant_divisor.filter(|&value| value > 1)?;
    let mut factors = Vec::new();
    for &(_, coefficient) in &dividend.terms {
        let factor = gcd(divisor.unsigned_abs(), coefficient.unsigned_abs()) as i64;
        if factor > 1 && factor < divisor && !factors.contains(&factor) {
            factors.push(factor);
        }
    }
    factors.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
    for factor in factors {
        let mut high = Linear::constant(dividend.constant.div_euclid(factor));
        let mut low = Linear::constant(dividend.constant.rem_euclid(factor));
        for (atom, coefficient) in &dividend.terms {
            match coefficient % factor {
                0 => high.terms.push((atom.clone(), coefficient / factor)),
                _ => low.terms.push((atom.clone(), *coefficient)),
            }
        }
        let in_range =
            range_of(&low.expr(), extent).is_some_and(|(least, most)| least >= 0 && most < factor);
        if in_range {
            return Some((factor, divisor / factor, high, low));
        }
    }
    None
}

/// The canonical linear form of `min(lhs, rhs)` when `is_min`, else of `max(lhs, rhs)`.
fn linear_extreme<R: ReadKind>(
    is_min: bool,
    lhs: &Expr<R>,
    rhs: &Expr<R>,
    extent: &dyn Fn(usize) -> Option<Dim>,
) -> Linear<R> {
    let (left, right) = (lhs.canonical(extent), rhs.canonical(extent));
    if left == right {
        return linear(&left, extent);
    }
    if let (Some((left_low, left_high)), Some((right_low, right_high))) =
        (range_of(&left, extent), range_of(&right, extent))
    {
        let (first, second) = if is_min {
            (left_high <= right_low, right_high <= left_low)
        } else {
            (left_low >= right_high, right_low >= left_high)
        };
        if first {
            return linear(&left, extent);
        }
        if second {
            return linear(&right, extent);
        }
    }
    // `min(max(x, c), c)` and `max(min(x, c), c)` are `c`.
    let inner = match (left.root(), is_min) {
        (Root::Binary(BinaryOp::Max, _, bound), true)
        | (Root::Binary(BinaryOp::Min, _, bound), false) => Some(bound),
        _ => None,
    };
    if inner.is_some_and(|bound| bound == right && right.as_const().is_some()) {
        return linear(&right, extent);
    }
    let op = if is_min { BinaryOp::Min } else { BinaryOp::Max };
    Linear::atom(Expr::binary(op, left, right))
}
