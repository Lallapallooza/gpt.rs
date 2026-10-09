//! 8-bit floating-point elements.

/// `float8_e4m3fn`: 4 exponent bits with bias 7, 3 mantissa bits, no infinities, and NaN at
/// `S.1111.111`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct F8E4M3(pub u8);

impl F8E4M3 {
    /// The value, exactly: every e4m3 value is an f32. It computes the normal and the subnormal
    /// form and selects one, so loops over weights vectorize.
    pub fn to_f32(self) -> f32 {
        let bits = u32::from(self.0);
        let sign = (bits & 0x80) << 24;
        let magnitude = bits & 0x7f;
        // A normal value keeps its fields, its exponent moved from bias 7 to f32's bias 127.
        let normal = f32::from_bits(sign | ((magnitude << 20) + (120 << 23)));
        // A subnormal is its mantissa times 2^-9.
        let subnormal = f32::from_bits(sign | (magnitude as f32 / 512.0).to_bits());
        let value = if magnitude < 0x08 { subnormal } else { normal };
        if magnitude == 0x7f {
            f32::NAN
        } else {
            value
        }
    }
}
