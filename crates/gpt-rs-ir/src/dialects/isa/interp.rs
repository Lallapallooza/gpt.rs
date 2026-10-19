//! Semantics of the `isa` ops. The lanes of a group of registers are a tensor of its extents, and
//! the lane-wise ops compute them as the `tensor` ops of the same names compute elements.

use anyhow::{ensure, Result};

use crate::dialects::arith::math;
use crate::dialects::isa::ir::*;
use crate::dialects::isa::utils::{register_position, repeated_position, row_major};
use crate::dialects::tensor::interp::elementwise::{binary_data, compare_data, unary_data};
use crate::dialects::tensor::interp::kernels::{bitcast, cast};
use crate::host::{tensor, Decoded, HostScalar, HostTensor};
use crate::interp::{HostBuffer, Unit};
use crate::ir::{DType, Eval, EvalCx, Host, NoRegions, OwnedHost, Ranked, VectorType};

/// The semantics of every [`Unary`] op.
pub(crate) fn unary<T: Unary>(
    _op: &T,
    ins: UnaryOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<UnaryResults<OwnedHost>> {
    let data = unary_data::<T::Scalar>(&ins.input.data)?;
    Ok(UnaryResults {
        output: HostTensor::new(ins.input.dims, data)?,
    })
}

/// The semantics of every [`Binary`] op.
pub(crate) fn binary<T: Binary>(
    _op: &T,
    ins: BinaryOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<BinaryResults<OwnedHost>> {
    let data = binary_data::<T::Scalar>(&ins.lhs.data, &ins.rhs.data)?;
    Ok(BinaryResults {
        output: HostTensor::new(ins.lhs.dims, data)?,
    })
}

/// The semantics of every [`Compare`] op.
pub(crate) fn compare<T: Compare>(
    _op: &T,
    ins: CompareOperands<Host>,
    _cx: &mut EvalCx<NoRegions>,
) -> Result<CompareResults<OwnedHost>> {
    let flags = compare_data::<T::Scalar>(&ins.lhs.data, &ins.rhs.data)?;
    Ok(CompareResults {
        output: HostTensor::new(ins.lhs.dims, flags)?,
    })
}

impl Eval for Select {
    fn eval(
        &self,
        ins: SelectOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<SelectResults<OwnedHost>> {
        let flags = Decoded::new(&ins.pred.data);
        let on_true = Decoded::new(&ins.on_true.data);
        let on_false = Decoded::new(&ins.on_false.data);
        let mut lanes = Vec::with_capacity(lane_count(&ins.on_true.dims));
        for position in 0..lane_count(&ins.on_true.dims) {
            let branch = match is_set(&flags, position) {
                true => &on_true,
                false => &on_false,
            };
            lanes.push(branch.get(position));
        }
        Ok(SelectResults {
            output: lanes_of(ins.on_true.dtype(), &ins.on_true.dims, &lanes)?,
        })
    }
}

impl Eval for Fma {
    fn eval(
        &self,
        ins: FmaOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<FmaResults<OwnedHost>> {
        let lhs = Decoded::new(&ins.lhs.data);
        let rhs = Decoded::new(&ins.rhs.data);
        let acc = Decoded::new(&ins.acc.data);
        let mut lanes = Vec::with_capacity(lane_count(&ins.lhs.dims));
        for position in 0..lane_count(&ins.lhs.dims) {
            lanes.push(math::fma(
                lhs.get(position),
                rhs.get(position),
                acc.get(position),
            )?);
        }
        Ok(FmaResults {
            output: lanes_of(ins.lhs.dtype(), &ins.lhs.dims, &lanes)?,
        })
    }
}

impl Eval for Cast {
    fn eval(
        &self,
        ins: CastOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<CastResults<OwnedHost>> {
        Ok(CastResults {
            output: HostTensor::new(ins.input.dims, cast(&ins.input.data, self.to)?)?,
        })
    }
}

impl Eval for Bitcast {
    fn eval(
        &self,
        ins: BitcastOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<BitcastResults<OwnedHost>> {
        let data = bitcast(ins.input, self.to, ins.input.data.len())?;
        Ok(BitcastResults {
            output: HostTensor::new(ins.input.dims, data)?,
        })
    }
}

impl Eval for Splat {
    fn eval(
        &self,
        ins: SplatOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<SplatResults<OwnedHost>> {
        let lanes = vec![ins.input; lane_count(&self.dims)];
        Ok(SplatResults {
            output: lanes_of(ins.input.dtype(), &self.dims, &lanes)?,
        })
    }
}

impl Eval for Expand {
    fn eval(
        &self,
        ins: ExpandOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<ExpandResults<OwnedHost>> {
        let from = Decoded::new(&ins.input.data);
        let mut lanes = Vec::with_capacity(lane_count(&self.dims));
        for coords in row_major(&self.dims) {
            lanes.push(from.get(repeated_position(&coords, &ins.input.dims)));
        }
        Ok(ExpandResults {
            output: lanes_of(ins.input.dtype(), &self.dims, &lanes)?,
        })
    }
}

impl Eval for TileMask {
    fn eval(
        &self,
        ins: TileMaskOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<TileMaskResults<OwnedHost>> {
        let lane_extent = *self.dims.last().expect("a vector has lanes") as i64;
        let mut flags = Vec::with_capacity(lane_count(&self.dims));
        for coords in row_major(&self.dims) {
            let (lane, group) = coords.split_last().expect("a vector has lanes");
            let mut inside = true;
            match group.split_last() {
                None => inside &= (*lane as i64) < ins.extents[0],
                Some((last, outer)) => {
                    for (&coord, &extent) in outer.iter().zip(&ins.extents) {
                        inside &= (coord as i64) < extent;
                    }
                    let column = *last as i64 * lane_extent + *lane as i64;
                    inside &= column < ins.extents[outer.len()];
                }
            }
            flags.push(HostScalar::int(DType::I1, i128::from(inside))?);
        }
        Ok(TileMaskResults {
            output: lanes_of(DType::I1, &self.dims, &flags)?,
        })
    }
}

/// The semantics of every [`Read`] op.
pub(crate) fn read<T: Read>(
    op: &T,
    ins: ReadOperands<Host>,
    cx: &mut EvalCx<NoRegions>,
) -> Result<ReadResults<OwnedHost>> {
    let access = Access {
        buffer: ins.buffer,
        base: &ins.base,
        steps: &op.steps,
        mask: None,
        unit: cx.unit(),
    };
    Ok(ReadResults {
        output: access.read(op, T::ALONG)?,
    })
}

/// The semantics of every [`MaskedRead`] op.
pub(crate) fn masked_read<T: MaskedRead>(
    op: &T,
    ins: MaskedReadOperands<Host>,
    cx: &mut EvalCx<NoRegions>,
) -> Result<MaskedReadResults<OwnedHost>> {
    let access = Access {
        buffer: ins.buffer,
        base: &ins.base,
        steps: &op.steps,
        mask: Some(ins.mask),
        unit: cx.unit(),
    };
    Ok(MaskedReadResults {
        output: access.read(op, T::ALONG)?,
    })
}

/// A hint, which changes nothing.
impl Eval for Prefetch {
    fn eval(
        &self,
        _ins: PrefetchOperands<Host>,
        _cx: &mut EvalCx<NoRegions>,
    ) -> Result<PrefetchResults<OwnedHost>> {
        Ok(PrefetchResults::default())
    }
}

impl Eval for Store {
    fn eval(
        &self,
        ins: StoreOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<StoreResults<OwnedHost>> {
        let access = Access {
            buffer: ins.buffer,
            base: &ins.base,
            steps: &self.steps,
            mask: None,
            unit: cx.unit(),
        };
        access.write(ins.value)?;
        Ok(StoreResults::default())
    }
}

impl Eval for MaskedStore {
    fn eval(
        &self,
        ins: MaskedStoreOperands<Host>,
        cx: &mut EvalCx<NoRegions>,
    ) -> Result<MaskedStoreResults<OwnedHost>> {
        let access = Access {
            buffer: ins.buffer,
            base: &ins.base,
            steps: &self.steps,
            mask: Some(ins.mask),
            unit: cx.unit(),
        };
        access.write(ins.value)?;
        Ok(MaskedStoreResults::default())
    }
}

/// How a memory op reaches the elements of a buffer: from `base`, moving by `steps` from one
/// register to the next, and to the lanes a mask leaves set.
struct Access<'a> {
    buffer: &'a HostBuffer,
    base: &'a [i64],
    steps: &'a [Ranked<i64>],
    mask: Option<&'a HostTensor>,
    unit: &'a Unit,
}

impl Access<'_> {
    /// The position in the buffer of lane `lane` of the register at `coords` in the group: the
    /// register's position, and `lane` further along the last axis when `along` holds.
    fn position(&self, coords: &[usize], lane: usize, along: bool) -> Ranked<i64> {
        let mut position = register_position(self.base, self.steps, coords);
        if along {
            *position.last_mut().expect("a buffer has an axis") += lane as i64;
        }
        position
    }

    /// Whether the lane at `index`, in row-major order, is set in the mask.
    fn active(&self, flags: &Option<Decoded>, index: usize) -> bool {
        flags.as_ref().is_none_or(|decoded| is_set(decoded, index))
    }

    /// The registers that `read` reads from the buffer: each lane at its own position when
    /// `along` holds, else every lane at its register's position. With a pad, a lane may fall
    /// outside the buffer on the axes that `checked` names for its register, or on any axis when
    /// `checked` is empty. Such a lane is the pad. A lane outside on another axis fails the read.
    fn read(&self, read: &ReadAttrs, along: bool) -> Result<HostTensor> {
        let ty = self.buffer.ty();
        let flags = self.mask.map(|mask| Decoded::new(&mask.data));
        let lanes_per_register = *read.dims.last().expect("a vector has lanes");
        let mut lanes = Vec::with_capacity(lane_count(&read.dims));
        for (index, coords) in row_major(&read.dims).into_iter().enumerate() {
            let (lane, group) = coords.split_last().expect("a vector has lanes");
            if !self.active(&flags, index) {
                lanes.push(HostScalar::zero(ty.dtype)?);
                continue;
            }
            let position = self.position(group, *lane, along);
            let Some(pad_value) = read.pad.filter(|_| !self.buffer.contains(&position)) else {
                lanes.push(self.buffer.load(&position, self.unit)?);
                continue;
            };
            let register = index / lanes_per_register;
            for (axis, (&coord, dim)) in position.iter().zip(&ty.dims).enumerate() {
                let outside = dim
                    .as_static()
                    .is_none_or(|extent| coord < 0 || coord >= extent as i64);
                ensure!(
                    !outside || read.checked.is_empty() || read.checked[register].contains(&axis),
                    "lane {position:?} is outside {ty} on axis {axis}, which register {register} \
                     does not check"
                );
            }
            lanes.push(HostScalar::from_f64(ty.dtype, pad_value.0)?);
        }
        lanes_of(ty.dtype, &read.dims, &lanes)
    }

    /// Writes the lanes of `value` that the mask leaves set.
    fn write(&self, value: &HostTensor) -> Result<()> {
        let flags = self.mask.map(|mask| Decoded::new(&mask.data));
        let lanes = Decoded::new(&value.data);
        for (index, coords) in row_major(&value.dims).into_iter().enumerate() {
            if !self.active(&flags, index) {
                continue;
            }
            let (lane, group) = coords.split_last().expect("a vector has lanes");
            let position = self.position(group, *lane, true);
            self.buffer.store(&position, lanes.get(index), self.unit)?;
        }
        Ok(())
    }
}

/// The lanes of registers of the extents `dims`: their product.
fn lane_count(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Whether the flag at `position` of a mask is set.
fn is_set(flags: &Decoded, position: usize) -> bool {
    flags.get(position).as_int() != Some(0)
}

/// Registers of `dtype` and the extents `dims` that hold `lanes`, in row-major order.
fn lanes_of(dtype: DType, dims: &[usize], lanes: &[HostScalar]) -> Result<HostTensor> {
    ensure!(
        lanes.len() == lane_count(dims),
        "{} lanes for registers of the extents {dims:?}",
        lanes.len()
    );
    tensor(&VectorType::new(dtype, dims).tensor_type(), lanes)
}
