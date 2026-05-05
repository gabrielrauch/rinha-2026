use shared::{quantize_unit, VECTOR_DIM};

use crate::sources::ReferenceEntry;

pub fn quantize_entry(e: &ReferenceEntry) -> ([i8; VECTOR_DIM], bool) {
    debug_assert_eq!(e.vector.len(), VECTOR_DIM);
    let mut out = [0i8; VECTOR_DIM];
    for (i, &x) in e.vector.iter().enumerate().take(VECTOR_DIM) {
        out[i] = quantize_unit(x);
    }
    (out, e.label == "fraud")
}
