#![allow(clippy::needless_range_loop)]

use crate::blob::Blob;
use crate::distance::l2_squared;
use crate::hnsw::{search_top_k_ef, EF_FAST, TOP_K};
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

/// Exact brute-force top-5: scans all 3M vectors with AVX2 L2 — guaranteed
/// perfect recall. Used only on threshold-ambiguous queries where HNSW's
/// 99% recall flips classification.
fn brute_force_count(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let vectors = blob.vectors();
    let mut top: [(i32, u32); TOP_K] = [(i32::MAX, u32::MAX); TOP_K];
    let mut worst_idx = 0usize;

    for (i, v) in vectors.iter().enumerate() {
        let d = l2_squared(query, v);
        if d < top[worst_idx].0 {
            top[worst_idx] = (d, i as u32);
            let mut wi = 0;
            let mut wd = top[0].0;
            for (j, &(d_j, _)) in top.iter().enumerate().skip(1) {
                if d_j > wd {
                    wd = d_j;
                    wi = j;
                }
            }
            worst_idx = wi;
        }
    }

    let mut count: u8 = 0;
    for &(_, idx) in &top {
        if idx == u32::MAX {
            continue;
        }
        if blob.is_fraud(idx) {
            count += 1;
        }
    }
    count
}

/// HNSW fast path; brute-force exact for the ~3% threshold-borderline queries
/// (count ∈ {2,3} of 5) where HNSW recall errors flip classification.
pub fn fraud_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let top_fast = search_top_k_ef(blob, query, EF_FAST);
    let count_fast = count_frauds(blob, &top_fast);
    if count_fast == 2 || count_fast == 3 {
        return brute_force_count(blob, query);
    }
    count_fast
}
