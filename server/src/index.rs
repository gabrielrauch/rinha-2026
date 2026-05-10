#![allow(clippy::needless_range_loop)]

use crate::blob::Blob;
use crate::hnsw::{search_top_k_ef, EF_FAST, EF_FULL, TOP_K};
use shared::VECTOR_DIM;

#[inline]
fn count_frauds(blob: &Blob, top: &[(u32, u32); TOP_K]) -> u8 {
    let mut count: u8 = 0;
    for i in 0..TOP_K {
        let (_, idx) = top[i];
        if idx == u32::MAX {
            continue;
        }
        if blob.is_fraud(idx) {
            count += 1;
        }
    }
    count
}

/// Adaptive HNSW search: fast (ef_FAST) by default; if the count lands in the
/// threshold-ambiguous zone (2 or 3 of 5), re-search with ef_FULL — analogous
/// to jairoblatt's IVF nprobe=8/24 fallback. Most queries finish on the fast
/// path, only ~3% retry.
pub fn fraud_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let top_fast = search_top_k_ef(blob, query, EF_FAST);
    let count_fast = count_frauds(blob, &top_fast);
    if count_fast == 2 || count_fast == 3 {
        let top_full = search_top_k_ef(blob, query, EF_FULL);
        return count_frauds(blob, &top_full);
    }
    count_fast
}
