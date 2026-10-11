//! The geometry of the windows of `tensor.extract_patches` and the `tensor.reduce_window_*` ops.

use std::fmt;

use anyhow::{anyhow, ensure, Result};

use crate::dialects::tensor::ir::*;
use crate::types::TensorType;

impl ExtractPatches {
    /// The spatial axes of an `input` of type `[N, spatial..., C]`, and the result extents
    /// `[N, positions..., window elements * C]`.
    pub(crate) fn geometry(&self, input: &TensorType) -> Result<(Vec<WindowAxis>, Vec<usize>)> {
        let dims = input
            .static_dims()
            .ok_or_else(|| anyhow!("{input} has a dynamic dimension"))?;
        ensure!(
            dims.len() >= 3,
            "input must be [N, spatial..., C], got {input}"
        );
        let spatial = dims.len() - 2;
        let axes = format!("{spatial} spatial axes");
        ensure_per_axis("window", &self.window, spatial, &axes)?;
        ensure_per_axis("strides", &self.strides, spatial, &axes)?;
        ensure_per_axis("dilation", &self.dilation, spatial, &axes)?;
        ensure_per_axis("padding", &self.padding, spatial, &axes)?;
        ensure!(
            dims[1..].iter().all(|&n| n > 0),
            "input {input} has an empty spatial or channel axis"
        );
        ensure_positive("window", &self.window)?;
        ensure_positive("strides", &self.strides)?;
        ensure_positive("dilation", &self.dilation)?;

        let windowed: Vec<WindowAxis> = (0..spatial)
            .map(|i| WindowAxis {
                input: dims[1 + i],
                window: self.window[i],
                stride: self.strides[i],
                padding: self.padding[i],
                base_dilation: 1,
                window_dilation: self.dilation[i],
            })
            .collect();
        let mut out = Vec::with_capacity(dims.len());
        out.push(dims[0]);
        for (i, axis) in windowed.iter().enumerate() {
            out.push(axis.output(&format!("spatial axis {i}"))?);
        }
        let patch = self
            .window
            .iter()
            .try_fold(dims[dims.len() - 1], |acc, &n| acc.checked_mul(n))
            .ok_or_else(|| anyhow!("patch size overflows"))?;
        out.push(patch);
        Ok((windowed, out))
    }
}

impl ReduceWindowAttrs {
    /// The axes of an `input` of this type, and the result extents.
    pub(crate) fn geometry(&self, input: &TensorType) -> Result<(Vec<WindowAxis>, Vec<usize>)> {
        let dims = input
            .static_dims()
            .ok_or_else(|| anyhow!("{input} has a dynamic dimension"))?;
        let rank = dims.len();
        let axes = format!("rank {rank}");
        ensure_per_axis("window_dims", &self.window_dims, rank, &axes)?;
        ensure_per_axis("strides", &self.strides, rank, &axes)?;
        ensure_per_axis("padding", &self.padding, rank, &axes)?;
        ensure_per_axis("base_dilation", &self.base_dilation, rank, &axes)?;
        ensure_per_axis("window_dilation", &self.window_dilation, rank, &axes)?;
        ensure_positive("window_dims", &self.window_dims)?;
        ensure_positive("strides", &self.strides)?;
        ensure_positive("base_dilation", &self.base_dilation)?;
        ensure_positive("window_dilation", &self.window_dilation)?;

        let windowed: Vec<WindowAxis> = (0..rank)
            .map(|i| WindowAxis {
                input: dims[i],
                window: self.window_dims[i],
                stride: self.strides[i],
                padding: self.padding[i],
                base_dilation: self.base_dilation[i],
                window_dilation: self.window_dilation[i],
            })
            .collect();
        let out = windowed
            .iter()
            .enumerate()
            .map(|(i, axis)| axis.output(&format!("axis {i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok((windowed, out))
    }
}

/// One input axis that a window slides along.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowAxis {
    input: usize,
    window: usize,
    stride: usize,
    padding: (usize, usize),
    base_dilation: usize,
    window_dilation: usize,
}

impl WindowAxis {
    /// The number of window positions along the axis. Every field but `input` and `padding` must
    /// be positive. `at` names the axis in errors.
    fn output(&self, at: &str) -> Result<usize> {
        let dilated = self
            .input
            .checked_sub(1)
            .map_or(Some(0), |n| {
                n.checked_mul(self.base_dilation)?.checked_add(1)
            })
            .ok_or_else(|| anyhow!("the dilated input overflows on {at}"))?;
        let padded = dilated
            .checked_add(self.padding.0)
            .and_then(|n| n.checked_add(self.padding.1))
            .ok_or_else(|| anyhow!("the padded input overflows on {at}"))?;
        let span = (self.window - 1)
            .checked_mul(self.window_dilation)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow!("the dilated window overflows on {at}"))?;
        if self.input == 0 && span > padded {
            return Ok(0);
        }
        ensure!(
            span <= padded,
            "window extent {span} exceeds padded input extent {padded} on {at}"
        );
        Ok((padded - span) / self.stride + 1)
    }

    /// The input index that window element `offset` at output position `out` reads, or `None`
    /// when it falls in the padding or between two dilated input elements.
    pub(crate) fn source(&self, out: usize, offset: usize) -> Option<usize> {
        let position =
            (out * self.stride + offset * self.window_dilation).checked_sub(self.padding.0)?;
        let index = position / self.base_dilation;
        (position % self.base_dilation == 0 && index < self.input).then_some(index)
    }
}

/// Fails unless the per-axis attribute `name` has `len` entries. `axes` describes them in the
/// error.
fn ensure_per_axis<T: fmt::Debug>(name: &str, value: &[T], len: usize, axes: &str) -> Result<()> {
    ensure!(
        value.len() == len,
        "{name} {value:?} has rank {}, but the input has {axes}",
        value.len()
    );
    Ok(())
}

fn ensure_positive(name: &str, value: &[usize]) -> Result<()> {
    ensure!(
        value.iter().all(|&n| n > 0),
        "{name} {value:?} must all be positive"
    );
    Ok(())
}
