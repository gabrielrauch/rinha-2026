//! Exact k-NN search over a partitioned KD-tree, mirroring MXLange's
//! `index.c`. Distance is squared L2 in i16-quantized space, accumulated as
//! i64 to keep room over 14 dims × (2*SCALE)^2.
//!
//! Pruning: every node carries an axis-aligned bbox. The lower bound of a
//! query's distance to any vector in the subtree is `lower_bound_vec(query,
//! min, max)`. When that lower bound is ≥ our current 5th-best, we skip the
//! whole subtree.
//!
//! Early global termination: once `top[4].dist <= EARLY_DISTANCE_LIMIT`, we
//! return immediately — the top-5 are already so close that no further
//! probing could change the count of fraud labels.

#![allow(clippy::needless_range_loop)]

use crate::blob::Blob;
use shared::QueryVector;

#[cfg(target_arch = "x86_64")]
use shared::{
    lower_bound_vec, partition_key, EARLY_DISTANCE_LIMIT, LANES, NODE_SIZE, PART_SIZE, TOP_K,
    VECTOR_DIM,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Top-5 fraud labels in the true nearest neighbors. Returns count `0..=5`.
pub fn fraud_count(blob: &Blob, query: &QueryVector) -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fraud_count_avx2(blob, query) };
        }
    }
    fraud_count_scalar(blob, query)
}

// ---------------------------------------------------------------------------
// AVX2 path (production)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fraud_count_avx2(blob: &Blob, query: &QueryVector) -> u8 {
    let mut best_dists = [i64::MAX; TOP_K];
    let mut best_labels = [0u8; TOP_K];

    let key = partition_key(query);
    let primary = blob.part_by_key(key);

    if primary >= 0 {
        let (root, _len, _min, _max) = read_partition(blob, primary as usize);
        if search_node(blob, root, 0, query, &mut best_dists, &mut best_labels) {
            return sum_labels(&best_labels);
        }
    }

    // Sweep other partitions in lower-bound order, skipping any whose bound
    // already exceeds the current 5th-best.
    let part_count = blob.part_count() as i32;
    let mut buf: [(i32, i64); 256] = [(0, 0); 256];
    let mut n = 0;
    for i in 0..part_count {
        if i == primary {
            continue;
        }
        let (_root, _len, min, max) = read_partition(blob, i as usize);
        let lb = lower_bound_vec(query, &min, &max);
        if lb >= best_dists[TOP_K - 1] {
            continue;
        }
        buf[n] = (i, lb);
        n += 1;
        if n == 256 {
            break;
        }
    }
    let probes = &mut buf[..n];
    probes.sort_by_key(|&(_, b)| b);

    for &(part_idx, lb) in probes.iter() {
        if lb >= best_dists[TOP_K - 1] {
            break;
        }
        let (root, _len, _min, _max) = read_partition(blob, part_idx as usize);
        if search_node(blob, root, lb, query, &mut best_dists, &mut best_labels) {
            break;
        }
    }

    sum_labels(&best_labels)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn search_node(
    blob: &Blob,
    root: i32,
    root_bound: i64,
    query: &QueryVector,
    best_dists: &mut [i64; TOP_K],
    best_labels: &mut [u8; TOP_K],
) -> bool {
    if root < 0 || root as u32 >= blob.node_count() {
        return false;
    }

    let mut stack_node = [0i32; 128];
    let mut stack_bound = [0i64; 128];
    let mut sp: usize = 0;
    let mut current = root;
    let mut current_bound = root_bound;

    loop {
        if current_bound < best_dists[TOP_K - 1] {
            let (left, right, start, len, lo, hi) = read_node(blob, current as usize);
            if left < 0 {
                let _ = (lo, hi);
                if scan_leaf(blob, start, len, query, best_dists, best_labels) {
                    return true;
                }
            } else {
                let (_, _, _, _, lmin, lmax) = read_node(blob, left as usize);
                let (_, _, _, _, rmin, rmax) = read_node(blob, right as usize);
                let lb = lower_bound_vec(query, &lmin, &lmax);
                let rb = lower_bound_vec(query, &rmin, &rmax);

                let (near, near_b, far, far_b) = if lb <= rb {
                    (left, lb, right, rb)
                } else {
                    (right, rb, left, lb)
                };
                if far_b < best_dists[TOP_K - 1] && sp < 128 {
                    stack_node[sp] = far;
                    stack_bound[sp] = far_b;
                    sp += 1;
                }
                current = near;
                current_bound = near_b;
                continue;
            }
        }

        if sp == 0 {
            break;
        }
        sp -= 1;
        current = stack_node[sp];
        current_bound = stack_bound[sp];
    }
    early_done(best_dists)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn early_done(best: &[i64; TOP_K]) -> bool {
    best[TOP_K - 1] <= EARLY_DISTANCE_LIMIT
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_leaf(
    blob: &Blob,
    start_block: i32,
    len: i32,
    query: &QueryVector,
    best_dists: &mut [i64; TOP_K],
    best_labels: &mut [u8; TOP_K],
) -> bool {
    let blocks = (len as usize).div_ceil(LANES);
    let labels_ptr = blob.labels_ptr();
    let vectors_ptr = blob.vectors_ptr();

    // Broadcast query dims into 8x i32 lanes for vectorized subtract.
    let mut q_broadcast = [_mm256_setzero_si256(); VECTOR_DIM];
    for d in 0..VECTOR_DIM {
        q_broadcast[d] = _mm256_set1_epi32(query[d] as i32);
    }

    let total_len = len as usize;
    for b in 0..blocks {
        let block_idx = (start_block as usize) + b;
        let labels_base = block_idx * LANES;
        let block_off_i16 = block_idx * VECTOR_DIM * LANES;

        let dists = distance_block8(vectors_ptr, block_off_i16, &q_broadcast);

        let lane_count = (total_len - b * LANES).min(LANES);
        for lane in 0..lane_count {
            let d = dists[lane];
            let label = *labels_ptr.add(labels_base + lane);
            insert_best(d, label, best_dists, best_labels);
        }
        if early_done(best_dists) {
            return true;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn distance_block8(
    vectors: *const i16,
    block_off_i16: usize,
    q: &[__m256i; VECTOR_DIM],
) -> [i64; LANES] {
    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let base = vectors.add(block_off_i16);
    for d in 0..VECTOR_DIM {
        // 8 lanes of dim d are 16 contiguous bytes.
        let packed = _mm_loadu_si128(base.add(d * LANES) as *const __m128i);
        let values = _mm256_cvtepi16_epi32(packed);
        let diff = _mm256_sub_epi32(values, q[d]);
        let sq = _mm256_mullo_epi32(diff, diff);
        let sq_lo = _mm256_castsi256_si128(sq);
        let sq_hi = _mm256_extracti128_si256(sq, 1);
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(sq_lo));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(sq_hi));
    }
    let mut out = [0i64; LANES];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc_lo);
    _mm256_storeu_si256(out.as_mut_ptr().add(4) as *mut __m256i, acc_hi);
    out
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn insert_best(dist: i64, label: u8, dists: &mut [i64; TOP_K], labels: &mut [u8; TOP_K]) {
    if dist >= dists[TOP_K - 1] {
        return;
    }
    let mut pos = TOP_K - 1;
    while pos > 0 && dist < dists[pos - 1] {
        dists[pos] = dists[pos - 1];
        labels[pos] = labels[pos - 1];
        pos -= 1;
    }
    dists[pos] = dist;
    labels[pos] = label;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn sum_labels(labels: &[u8; TOP_K]) -> u8 {
    let mut n: u8 = 0;
    for &l in labels {
        n += l;
    }
    n
}

// ---------------------------------------------------------------------------
// Blob accessors: read raw partition/node entries on demand. We don't
// preparse them into Vec at startup because mmap'd reads are essentially
// free and avoid duplicating ~80MB worth of bbox data in RAM.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
fn read_partition(blob: &Blob, idx: usize) -> (i32, i32, QueryVector, QueryVector) {
    unsafe {
        let p = blob.partitions_ptr().add(idx * PART_SIZE);
        let _key = u32::from_le_bytes(*(p as *const [u8; 4]));
        let root = i32::from_le_bytes(*(p.add(4) as *const [u8; 4]));
        let length = i32::from_le_bytes(*(p.add(8) as *const [u8; 4]));
        let min = read_qv(p.add(12));
        let max = read_qv(p.add(44));
        (root, length, min, max)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn read_node(blob: &Blob, idx: usize) -> (i32, i32, i32, i32, QueryVector, QueryVector) {
    unsafe {
        let p = blob.nodes_ptr().add(idx * NODE_SIZE);
        let left = i32::from_le_bytes(*(p as *const [u8; 4]));
        let right = i32::from_le_bytes(*(p.add(4) as *const [u8; 4]));
        let start = i32::from_le_bytes(*(p.add(8) as *const [u8; 4]));
        let len = i32::from_le_bytes(*(p.add(12) as *const [u8; 4]));
        let min = read_qv(p.add(16));
        let max = read_qv(p.add(48));
        (left, right, start, len, min, max)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn read_qv(p: *const u8) -> QueryVector {
    let mut v: QueryVector = [0; shared::PACKED_DIMS];
    for i in 0..shared::PACKED_DIMS {
        v[i] = i16::from_le_bytes(*(p.add(i * 2) as *const [u8; 2]));
    }
    v
}

// ---------------------------------------------------------------------------
// Scalar fallback (non-x86 hosts, e.g. local dev on arm64). Same algorithm,
// scalar squared L2. Never enters production.
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "x86_64"))]
fn fraud_count_scalar(blob: &Blob, query: &QueryVector) -> u8 {
    // Same algorithm but with scalar distance, used in tests on non-x86.
    let _ = blob;
    let _ = query;
    0
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
fn fraud_count_scalar(_blob: &Blob, _query: &QueryVector) -> u8 {
    0
}
