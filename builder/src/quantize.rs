use crate::sources::ReferenceEntry;
use shared::{QUANT_SCALE, VECTOR_DIM};

pub fn entry_to_f32(e: &ReferenceEntry) -> ([f32; VECTOR_DIM], bool) {
    debug_assert_eq!(e.vector.len(), VECTOR_DIM);
    let mut out = [0.0f32; VECTOR_DIM];
    for (i, &x) in e.vector.iter().enumerate().take(VECTOR_DIM) {
        out[i] = x;
    }
    (out, e.label == "fraud")
}

#[inline]
pub fn quantize_dim(v: f32) -> i16 {
    let scaled = (v * QUANT_SCALE).round();
    if scaled >= i16::MAX as f32 {
        i16::MAX
    } else if scaled <= i16::MIN as f32 {
        i16::MIN
    } else {
        scaled as i16
    }
}
