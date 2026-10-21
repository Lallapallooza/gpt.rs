//! Scalar C: indices, `loop.predicate`, and the `arith` ops on one value.

use anyhow::{anyhow, bail, Result};
use gpt_rs_ir::dialects::{arith, loops};
use gpt_rs_ir::host::{HostScalar, ScalarValue};
use gpt_rs_ir::ir::{
    BinaryOp, Constraint, DType, Expr, InOp, IndexExpr, Inst, ReadKind, Relation, Root, Type,
};
use gpt_rs_ir::Op;

use super::Printer;

/// The C type of a scalar of `dtype`.
pub(super) fn c_type(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::Si8 => "int8_t",
        DType::Si16 => "int16_t",
        DType::Si32 => "int32_t",
        DType::Si64 => "int64_t",
        DType::Ui8 => "uint8_t",
        DType::Ui16 => "uint16_t",
        DType::Ui32 => "uint32_t",
        DType::Ui64 => "uint64_t",
        DType::I1 => "gptrs_i1",
        // Held as their bits; the x86 backend computes on them in f32.
        DType::Bf16 | DType::F16 => "uint16_t",
        DType::F8E4M3 | DType::F8E5M2 => "uint8_t",
        // A 4-bit integer in the lower bits of a byte.
        DType::Ui4 => "uint8_t",
        DType::Si4 => "int8_t",
        other => bail!("the x86 backend has no C type for {other} scalars"),
    })
}

/// The unsigned C type of the width of the integer `dtype`.
fn unsigned_type(dtype: DType) -> Result<&'static str> {
    let int = dtype
        .int_format()
        .ok_or_else(|| anyhow!("{dtype} is not an integer"))?;
    Ok(match int.bits {
        1 | 8 => "uint8_t",
        16 => "uint16_t",
        32 => "uint32_t",
        64 => "uint64_t",
        bits => bail!("the x86 backend has no C type for {bits}-bit integers"),
    })
}

/// The float `value`, exactly, as a C expression of `dtype`.
pub(super) fn float_literal(dtype: DType, value: f64) -> Result<String> {
    Ok(match dtype {
        DType::F32 => {
            let single = value as f32;
            match single.is_finite() {
                true => format!("{single:?}f"),
                false => format!("__builtin_bit_cast(float, 0x{:08x}u)", single.to_bits()),
            }
        }
        DType::F64 => match value.is_finite() {
            true => format!("{value:?}"),
            false => format!("__builtin_bit_cast(double, 0x{:016x}ull)", value.to_bits()),
        },
        other => bail!("{other} is not a float the x86 backend computes in"),
    })
}

/// The pad `value` of a load from a buffer of `dtype`, a value of that dtype.
pub(super) fn pad_literal(dtype: DType, value: f64) -> Result<String> {
    let scalar = HostScalar::from_f64(dtype, value)?;
    literal(&scalar)
}

/// The scalar `scalar` as a C expression of its dtype.
fn literal(scalar: &HostScalar) -> Result<String> {
    let dtype = scalar.dtype;
    if is_narrow(dtype) {
        let data = scalar.to_data()?;
        return Ok(match data.bytes() {
            [code] => format!("((uint8_t)0x{code:02x}u)"),
            [low, high] => format!("((uint16_t)0x{:04x}u)", u16::from_le_bytes([*low, *high])),
            _ => bail!("a {dtype} element is one or two bytes"),
        });
    }
    match scalar.value {
        ScalarValue::F32(value) => float_literal(dtype, f64::from(value)),
        ScalarValue::F64(value) => float_literal(dtype, value),
        ScalarValue::Int(value) => {
            let ty = c_type(dtype)?;
            Ok(match dtype {
                DType::I1 => format!("{value}"),
                DType::Si64 if value == i128::from(i64::MIN) => {
                    "(int64_t)(-9223372036854775807LL - 1)".to_string()
                }
                DType::Ui64 | DType::Ui32 => format!("(({ty}){value}ull)"),
                _ => format!("(({ty}){value}LL)"),
            })
        }
    }
}

/// `expr`, an expression of the index language, in C on the operands `operands`.
pub(super) fn index_expr<R: ReadKind>(expr: &Expr<R>, operands: &[String]) -> Result<String> {
    Ok(match expr.root() {
        Root::Const(value) => match value < 0 {
            true => format!("({value}LL)"),
            false => format!("{value}LL"),
        },
        Root::Iter(position) => operands
            .get(position)
            .cloned()
            .ok_or_else(|| anyhow!("the index names d{position}, which it has no operand for"))?,
        Root::Sym(sym) => bail!(
            "the x86 backend has no extent for the symbol {{{}}}",
            sym.as_str()
        ),
        Root::Read(..) => bail!("an index of a loop reads no operand"),
        Root::Neg(inner) => format!("(-{})", index_expr(&inner, operands)?),
        Root::Binary(op, lhs_expr, rhs_expr) => {
            let (lhs, rhs) = (
                index_expr(&lhs_expr, operands)?,
                index_expr(&rhs_expr, operands)?,
            );
            match op {
                BinaryOp::Add => format!("({lhs} + {rhs})"),
                BinaryOp::Sub => format!("({lhs} - {rhs})"),
                BinaryOp::Mul => format!("({lhs} * {rhs})"),
                // Floor division and a remainder of the divisor's sign, by a positive divisor:
                // C's division truncates, so a negative remainder steps down by one.
                BinaryOp::FloorDiv => format!("({lhs} / {rhs} - ({lhs} % {rhs} < 0))"),
                BinaryOp::Mod => format!("(({lhs} % {rhs} + {rhs}) % {rhs})"),
                BinaryOp::Min => format!("({lhs} < {rhs} ? {lhs} : {rhs})"),
                BinaryOp::Max => format!("({lhs} > {rhs} ? {lhs} : {rhs})"),
            }
        }
    })
}

/// Whether `constraint` holds: any of its comparisons.
fn constraint(constraint: &Constraint, operands: &[String]) -> Result<String> {
    let mut comparisons = Vec::with_capacity(constraint.0.len());
    for cmp in &constraint.0 {
        let lhs = index_expr(&cmp.lhs, operands)?;
        let rhs = index_expr(&cmp.rhs, operands)?;
        let symbol = match cmp.op {
            Relation::Less => "<",
            Relation::LessEqual => "<=",
            Relation::Equal => "==",
            Relation::NotEqual => "!=",
            Relation::GreaterEqual => ">=",
            Relation::Greater => ">",
        };
        comparisons.push(format!("{lhs} {symbol} {rhs}"));
    }
    Ok(format!("({})", comparisons.join(" || ")))
}

impl Printer<'_> {
    /// `loop.index` and `loop.predicate`.
    pub(super) fn index(&mut self, inst: Inst) -> Result<()> {
        let op = self.module.op(inst);
        let operands = self.operand_names(inst)?;
        let value = self.module.results(inst)[0];
        if let Some(constant) = op
            .get::<loops::Index>()
            .and_then(|index| index.expr.as_const())
        {
            self.define_constant(value, int_literal(i128::from(constant)));
            return Ok(());
        }
        let result = self.define(value);
        if let Some(index) = op.get::<loops::Index>() {
            let expr = index_expr::<_>(&index.expr as &IndexExpr, &operands)?;
            self.line(format!("int64_t {result} = {expr};"));
            return Ok(());
        }
        let Some(predicate) = op.get::<loops::Predicate>() else {
            bail!("{} is neither an index nor a predicate", op.name());
        };
        let mut conditions = Vec::with_capacity(predicate.constraints.len());
        for each in &predicate.constraints {
            conditions.push(constraint(each, &operands)?);
        }
        let all = match conditions.is_empty() {
            true => "1".to_string(),
            false => conditions.join(" && "),
        };
        self.line(format!("gptrs_i1 {result} = {all};"));
        Ok(())
    }

    /// An `arith` op on scalars.
    pub(super) fn scalar(&mut self, inst: Inst) -> Result<()> {
        let op = self.module.op(inst);
        let operands = self.operand_names(inst)?;
        let inputs: Vec<Option<DType>> = self
            .module
            .operands(inst)
            .iter()
            .map(|&value| self.module.ty(value).as_scalar())
            .collect();
        let [result] = self.module.results(inst) else {
            bail!("{} has no scalar form in the x86 backend", op.name());
        };
        let ty = match self.module.ty(*result) {
            Type::Index => "int64_t",
            Type::Scalar(dtype) => c_type(*dtype)?,
            other => bail!("{} gives a {other}, which is no scalar", op.name()),
        };
        if let Some(constant) = op.get::<arith::Constant>() {
            self.define_constant(*result, literal(&constant.value.scalar())?);
            return Ok(());
        }
        let output = self.module.ty(*result).as_scalar();
        let text = expression(op, &inputs, output, &operands)?;
        let name = self.define(*result);
        self.line(format!("{ty} {name} = {text};"));
        Ok(())
    }

    pub(super) fn operand_names(&self, inst: Inst) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for &value in self.module.operands(inst) {
            names.push(self.name(value)?.to_string());
        }
        Ok(names)
    }
}

/// The C of the `arith` op `op` on operands of dtypes `inputs`, named `args`, giving `output`.
fn expression(
    op: &Op,
    inputs: &[Option<DType>],
    output: Option<DType>,
    args: &[String],
) -> Result<String> {
    if op.get::<arith::ToIndex>().is_some() {
        return Ok(format!("(int64_t){}", args[0]));
    }
    if let Some(from_index) = op.get::<arith::FromIndex>() {
        return cast(DType::Si64, from_index.dtype, &args[0]);
    }
    if op.get::<arith::Select>().is_some() {
        return Ok(format!("({} ? {} : {})", args[0], args[1], args[2]));
    }
    let input = inputs
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| anyhow!("{} takes no scalar", op.name()))?;
    if let Some(cast_op) = op.get::<arith::Cast>() {
        return cast(input, cast_op.to, &args[0]);
    }
    if let Some(bitcast) = op.get::<arith::Bitcast>() {
        return bitcast_expr(input, bitcast.to, &args[0]);
    }
    let form = match input {
        DType::F32 | DType::F64 => float_form(op, input),
        DType::I1 => logic_form(op),
        _ if input.is_integer() => int_form(op, input)?,
        _ => None,
    };
    let Some(template) = form else {
        let math = op.get::<arith::Exp>().is_some()
            || op.get::<arith::Log>().is_some()
            || op.get::<arith::Tanh>().is_some()
            || op.get::<arith::Erf>().is_some()
            || op.get::<arith::Sin>().is_some()
            || op.get::<arith::Cos>().is_some();
        bail!(
            "{} of {input} has no C form in the x86 backend{}",
            op.name(),
            match (math, output) {
                (true, Some(DType::F32)) => {
                    "; expand-math writes the math functions of f32 as polynomials"
                }
                _ => "",
            }
        );
    };
    Ok(fill(&template, args))
}

/// `template` with `$0`, `$1`, ... replaced by the operands `args`.
pub(super) fn fill(template: &str, args: &[String]) -> String {
    let mut text = template.to_string();
    for (position, arg) in args.iter().enumerate().rev() {
        text = text.replace(&format!("${position}"), arg);
    }
    text
}

/// Whether `op` is a `T`.
fn is<T: InOp>(op: &Op) -> bool {
    op.get::<T>().is_some()
}

/// An op recognizer and the C form of the op it recognizes.
type Form = (fn(&Op) -> bool, &'static str);

/// An op recognizer and the C forms of its op on f32 and on f64.
type FloatForm = (fn(&Op) -> bool, &'static str, &'static str);

/// The C of the comparisons, on any scalars.
const COMPARISONS: &[Form] = &[
    (is::<arith::Less>, "($0 < $1)"),
    (is::<arith::LessEqual>, "($0 <= $1)"),
    (is::<arith::Equal>, "($0 == $1)"),
    (is::<arith::GreaterEqual>, "($0 >= $1)"),
    (is::<arith::Greater>, "($0 > $1)"),
    (is::<arith::NotEqual>, "($0 != $1)"),
];

/// The C of the `arith` ops on f32, then on f64.
const FLOAT_FORMS: &[FloatForm] = &[
    (is::<arith::Neg>, "(-$0)", "(-$0)"),
    (
        is::<arith::Abs>,
        "__builtin_fabsf($0)",
        "__builtin_fabs($0)",
    ),
    (
        is::<arith::Rsqrt>,
        "(1.0f / __builtin_sqrtf($0))",
        "(1.0 / __builtin_sqrt($0))",
    ),
    (is::<arith::Reciprocal>, "(1.0f / $0)", "(1.0 / $0)"),
    (is::<arith::Add>, "($0 + $1)", "($0 + $1)"),
    (is::<arith::Sub>, "($0 - $1)", "($0 - $1)"),
    (is::<arith::Mul>, "($0 * $1)", "($0 * $1)"),
    (is::<arith::Div>, "($0 / $1)", "($0 / $1)"),
    // One lane of the vector form: a NaN operand, else `vrangeps`, which orders the zeros as the
    // op does and which no C operator is.
    (
        is::<arith::Maximum>,
        "($0 != $0 ? $0 : $1 != $1 ? $1 : _mm_cvtss_f32(_mm_range_ss(_mm_set_ss($0), _mm_set_ss($1), 0x05)))",
        "($0 != $0 ? $0 : $1 != $1 ? $1 : _mm_cvtsd_f64(_mm_range_sd(_mm_set_sd($0), _mm_set_sd($1), 0x05)))",
    ),
    (
        is::<arith::Minimum>,
        "($0 != $0 ? $0 : $1 != $1 ? $1 : _mm_cvtss_f32(_mm_range_ss(_mm_set_ss($0), _mm_set_ss($1), 0x04)))",
        "($0 != $0 ? $0 : $1 != $1 ? $1 : _mm_cvtsd_f64(_mm_range_sd(_mm_set_sd($0), _mm_set_sd($1), 0x04)))",
    ),
    (
        is::<arith::Fma>,
        "__builtin_fmaf($0, $1, $2)",
        "__builtin_fma($0, $1, $2)",
    ),
];

fn float_form(op: &Op, dtype: DType) -> Option<String> {
    if let Some((_, form)) = COMPARISONS.iter().find(|(matches, _)| matches(op)) {
        return Some(form.to_string());
    }
    let (_, single, double) = FLOAT_FORMS.iter().find(|(matches, ..)| matches(op))?;
    Some(match dtype {
        DType::F32 => single.to_string(),
        _ => double.to_string(),
    })
}

/// The C of the `arith` ops on `i1`.
const LOGIC_FORMS: &[Form] = &[
    (is::<arith::Not>, "(!$0)"),
    (is::<arith::And>, "($0 & $1)"),
    (is::<arith::Or>, "($0 | $1)"),
    (is::<arith::Xor>, "($0 ^ $1)"),
];

fn logic_form(op: &Op) -> Option<String> {
    COMPARISONS
        .iter()
        .chain(LOGIC_FORMS)
        .find(|(matches, _)| matches(op))
        .map(|(_, form)| form.to_string())
}

/// The C of the `arith` ops on integers, which wrap: `$T` is the dtype's C type, `$U` the
/// unsigned type of its width and `$W` the width. A shift by the width or more gives 0, or the
/// sign for an arithmetic shift, and C shifts only by less.
const INT_FORMS: &[Form] = &[
    (is::<arith::Neg>, "(($T)(0 - ($U)$0))"),
    (is::<arith::Add>, "(($T)(($U)$0 + ($U)$1))"),
    (is::<arith::Sub>, "(($T)(($U)$0 - ($U)$1))"),
    (is::<arith::Mul>, "(($T)(($U)$0 * ($U)$1))"),
    (is::<arith::Maximum>, "($0 > $1 ? $0 : $1)"),
    (is::<arith::Minimum>, "($0 < $1 ? $0 : $1)"),
    (is::<arith::Not>, "(($T)~$0)"),
    (is::<arith::And>, "($0 & $1)"),
    (is::<arith::Or>, "($0 | $1)"),
    (is::<arith::Xor>, "($0 ^ $1)"),
    (
        is::<arith::ShiftLeft>,
        "(($T)(($U)$1 >= $W ? 0 : ($U)$0 << ($U)$1))",
    ),
    (
        is::<arith::ShiftRightLogical>,
        "(($T)(($U)$1 >= $W ? 0 : ($U)$0 >> ($U)$1))",
    ),
];

fn int_form(op: &Op, dtype: DType) -> Result<Option<String>> {
    let int = dtype.int_format().expect("an integer");
    let form = if let Some((_, form)) = COMPARISONS.iter().find(|(matches, _)| matches(op)) {
        form.to_string()
    } else if let Some((_, form)) = INT_FORMS.iter().find(|(matches, _)| matches(op)) {
        form.to_string()
    } else if is::<arith::Abs>(op) {
        match int.signed {
            true => "(($T)($0 < 0 ? 0 - ($U)$0 : ($U)$0))".to_string(),
            false => "$0".to_string(),
        }
    } else if is::<arith::Div>(op) {
        // Division truncates, a divisor of 0, which the IR leaves undefined, gives 0, and the
        // quotient of the least value by -1 wraps.
        match int.signed {
            true => "(($T)($1 == 0 ? 0 : $1 == -1 ? 0 - ($U)$0 : $0 / $1))".to_string(),
            false => "(($T)($1 == 0 ? 0 : $0 / $1))".to_string(),
        }
    } else if is::<arith::ShiftRightArithmetic>(op) {
        "(($T)(($U)$1 >= $W ? ($0 < 0 ? -1 : 0) : $0 >> ($U)$1))".to_string()
    } else {
        return Ok(None);
    };
    Ok(Some(
        form.replace("$T", c_type(dtype)?)
            .replace("$U", unsigned_type(dtype)?)
            .replace("$W", &int.bits.to_string()),
    ))
}

/// `arith.cast` of `value` from `from` to `to`.
fn cast(from: DType, to: DType, value: &str) -> Result<String> {
    if from == to {
        return Ok(value.to_string());
    }
    if let Some(text) = half_cast(from, to, value) {
        return Ok(text);
    }
    // Every narrow float is an f32, so a cast through f32 rounds once, as the cast itself does.
    if is_narrow(from) && to != DType::F32 {
        if let Some(wide) = half_cast(from, DType::F32, value) {
            return cast(DType::F32, to, &wide);
        }
    }
    if is_narrow(from) || is_narrow(to) {
        bail!("the x86 backend converts {from} to {to} only through f32");
    }
    let target = c_type(to)?;
    if to == DType::I1 {
        return Ok(format!("({value} != 0)"));
    }
    if to.is_float() {
        return Ok(format!("(({target}){value})"));
    }
    let int = to
        .int_format()
        .ok_or_else(|| anyhow!("the x86 backend casts to no {to}"))?;
    if !int.signed && int.bits == 64 {
        return Ok(to_ui64(from, value));
    }
    let (low, high) = (int.min(), int.max());
    let (low_int, high_int) = (int_literal(low), int_literal(high));
    if from == DType::Ui64 {
        // An unsigned value saturates at the largest value of the target alone.
        return Ok(format!(
            "(({target})({value} > (uint64_t){high_int} ? {high_int} : (int64_t){value}))"
        ));
    }
    if from.is_float() {
        // Truncated toward zero and saturated, and NaN gives 0: the conversion only runs on a
        // value in range.
        let (below, above) = (low as f64, high as f64 + 1.0);
        let wide = format!("((double){value})");
        return Ok(format!(
            "(({target})({wide} != {wide} ? 0 : {wide} < {below:?} ? {low_int} : \
             {wide} >= {above:?} ? {high_int} : (int64_t){wide}))"
        ));
    }
    let wide = format!("((int64_t){value})");
    Ok(format!(
        "(({target})({wide} < {low_int} ? {low_int} : {wide} > {high_int} ? {high_int} : {wide}))"
    ))
}

/// `value` of `from` cast to ui64, saturated as `tensor.cast` saturates: a negative value or NaN
/// gives 0, and a float at or above `2^64` the largest ui64.
fn to_ui64(from: DType, value: &str) -> String {
    if from.is_float() {
        let wide = format!("((double){value})");
        return format!(
            "((uint64_t)({wide} != {wide} || {wide} < 0.0 ? 0 : \
             {wide} >= 18446744073709551616.0 ? UINT64_MAX : (uint64_t){wide}))"
        );
    }
    match from.int_format().is_some_and(|int| int.signed) {
        true => format!("((uint64_t)((int64_t){value} < 0 ? 0 : (int64_t){value}))"),
        false => format!("((uint64_t){value})"),
    }
}

/// Whether the x86 backend holds `dtype` as its bits and computes on it in f32: a float narrower
/// than f32.
fn is_narrow(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Bf16 | DType::F16 | DType::F8E4M3 | DType::F8E5M2
    )
}

/// The cast of `value` between f32 and a narrower float, as `tensor.cast` converts: exactly to
/// f32, and to bf16 or f16 rounded to nearest-even. An e5m2 code is the upper byte of the f16 of
/// its value.
fn half_cast(from: DType, to: DType, value: &str) -> Option<String> {
    Some(match (from, to) {
        (DType::Bf16, DType::F32) => {
            format!("__builtin_bit_cast(float, (uint32_t)({value}) << 16)")
        }
        (DType::F32, DType::Bf16) => format!("gptrs_bf16_of_f32({value})"),
        (DType::F16, DType::F32) => format!("_cvtsh_ss({value})"),
        (DType::F32, DType::F16) => format!("_cvtss_sh({value}, _MM_FROUND_TO_NEAREST_INT)"),
        (DType::F8E4M3, DType::F32) => format!("gptrs_f32_of_f8e4m3({value})"),
        (DType::F8E5M2, DType::F32) => {
            format!("_cvtsh_ss((uint16_t)((uint16_t)({value}) << 8))")
        }
        _ => return None,
    })
}

/// The integer `value` of the range of an `int64_t` as a C expression of that type.
fn int_literal(value: i128) -> String {
    match value == i128::from(i64::MIN) {
        true => "(-9223372036854775807LL - 1)".to_string(),
        false => format!("{value}LL"),
    }
}

/// `arith.bitcast` of `value` from `from` to `to`, of one width.
fn bitcast_expr(from: DType, to: DType, value: &str) -> Result<String> {
    let target = c_type(to)?;
    Ok(match from.is_float() || to.is_float() {
        true => format!("__builtin_bit_cast({target}, {value})"),
        false => format!("(({target}){value})"),
    })
}
