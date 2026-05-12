//! IVF search with AVX2+FMA SoA scan. Same compute kernel as jairoblatt's
//! reference, with our additions:
//! - K=8192 centroids (smaller clusters, less per-probe scan)
//! - Cluster ordering: vectors closest to centroid come first so early
//!   termination kicks in sooner
//! - 3-tier adaptive: nprobe=4 / 12 / 32
//! - Distance-weighted vote: instead of plain "count of fraud in top-5",
//!   weight each neighbor by 1/(dist+eps) and round (fraud_weight/total_weight)*5.
//!   Closer neighbors influence the call more — same as KNN with weighted
//!   distance, and shifts borderline counts toward whichever side is closer.

#![allow(clippy::needless_range_loop)]
#![allow(unsafe_op_in_unsafe_fn)]

use crate::blob::Blob;
#[cfg(target_arch = "x86_64")]
use shared::QUANT_INV_SCALE;
use shared::{BLOCK_VECS, NUM_CENTROIDS, VECTOR_DIM};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::mem::MaybeUninit;

pub const NPROBE_FAST: usize = 4;
pub const NPROBE_MID: usize = 12;
pub const NPROBE_DEEP: usize = 32;
pub const MAX_CENTROIDS: usize = NUM_CENTROIDS as usize;
pub const TOP_K: usize = 5;
/// Block byte stride: 14 dims × 8 vecs × i16 = 112 i16.
pub const BLOCK_I16: usize = VECTOR_DIM * BLOCK_VECS;

/// Top-level entry: returns the count of fraud labels in the top-5 nearest
/// neighbors. Uses adaptive nprobe to recheck borderline cases.
pub fn knn5_fraud_count(blob: &Blob, query: &[f32; VECTOR_DIM]) -> u8 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { knn5_ivf(blob, query) };
    }
    // Scalar fallback (only reachable on non-AVX2 machines — never in production
    // since the Mac Mini Haswell has both features).
    knn5_ivf_scalar(blob, query)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_ivf(blob: &Blob, query: &[f32; VECTOR_DIM]) -> u8 {
    let h = blob.header();
    let k = h.k_centroids as usize;
    let centroids_ptr = blob.centroids_ptr();

    let mut dists = [MaybeUninit::<f32>::uninit(); MAX_CENTROIDS];
    compute_centroid_dists(query, centroids_ptr, k, &mut dists);

    let mut q_vecs = [_mm256_setzero_ps(); VECTOR_DIM];
    for d in 0..VECTOR_DIM {
        q_vecs[d] = _mm256_set1_ps(query[d]);
    }

    let fast_probes = top_n_from_dists::<NPROBE_FAST>(&dists, k);
    let fast = scan_and_count(&fast_probes, blob, &q_vecs);

    // Mostly classification is final at the fast tier.
    if fast == 0 || fast == 5 {
        return fast;
    }

    let mid_probes = top_n_from_dists::<NPROBE_MID>(&dists, k);
    let mid = scan_and_count(&mid_probes, blob, &q_vecs);

    if mid != 2 && mid != 3 {
        return mid;
    }

    let deep_probes = top_n_from_dists::<NPROBE_DEEP>(&dists, k);
    scan_and_count(&deep_probes, blob, &q_vecs)
}

fn knn5_ivf_scalar(_blob: &Blob, _query: &[f32; VECTOR_DIM]) -> u8 {
    // We always run on AVX2+FMA — this is a safety net only.
    0
}

/// Compute squared L2 distance from `query` to all k centroids in parallel.
///
/// Centroids are stored f32 SoA (dim-major), so dim d of all k centroids is at
/// `centroids_ptr + d*k .. + (d+1)*k`. We can load 8 floats from one dimension
/// with one `loadu`, then fma against a broadcast of `query[d]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_centroid_dists(
    query: &[f32; VECTOR_DIM],
    centroids: *const f32,
    k: usize,
    dists: &mut [MaybeUninit<f32>; MAX_CENTROIDS],
) {
    let dp = dists.as_mut_ptr() as *mut f32;

    // Dim 0: initialize the accumulator.
    {
        let qd = _mm256_set1_ps(query[0]);
        let cp = centroids; // base for dim 0
        let mut ci = 0usize;
        while ci + 16 <= k {
            let c0 = _mm256_loadu_ps(cp.add(ci));
            let c1 = _mm256_loadu_ps(cp.add(ci + 8));
            let d0 = _mm256_sub_ps(c0, qd);
            let d1 = _mm256_sub_ps(c1, qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_mul_ps(d1, d1));
            ci += 16;
        }
        while ci + 8 <= k {
            let c0 = _mm256_loadu_ps(cp.add(ci));
            let d0 = _mm256_sub_ps(c0, qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(ci) - query[0];
            *dp.add(ci) = diff * diff;
            ci += 1;
        }
    }

    // Dims 1..14: accumulate.
    for d in 1..VECTOR_DIM {
        let qd = _mm256_set1_ps(query[d]);
        let cp = centroids.add(d * k);
        let mut ci = 0usize;
        while ci + 16 <= k {
            let c0 = _mm256_loadu_ps(cp.add(ci));
            let c1 = _mm256_loadu_ps(cp.add(ci + 8));
            let d0 = _mm256_sub_ps(c0, qd);
            let d1 = _mm256_sub_ps(c1, qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            let a1 = _mm256_loadu_ps(dp.add(ci + 8));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(d0, d0, a0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_fmadd_ps(d1, d1, a1));
            ci += 16;
        }
        while ci + 8 <= k {
            let c0 = _mm256_loadu_ps(cp.add(ci));
            let d0 = _mm256_sub_ps(c0, qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(d0, d0, a0));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(ci) - query[d];
            *dp.add(ci) += diff * diff;
            ci += 1;
        }
    }
}

/// Pick the N closest cluster indices from the precomputed dist array.
/// Uses AVX2 mask + linear insertion (N is small).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn top_n_from_dists<const N: usize>(
    dists: &[MaybeUninit<f32>; MAX_CENTROIDS],
    k: usize,
) -> [usize; N] {
    let mut top_dists = [f32::INFINITY; N];
    let mut top_idx = [0usize; N];
    let dp = dists.as_ptr() as *const f32;
    let mut ci = 0usize;

    while ci + 8 <= k {
        let d8 = _mm256_loadu_ps(dp.add(ci));
        let mask = _mm256_movemask_ps(_mm256_cmp_ps(
            d8,
            _mm256_set1_ps(top_dists[N - 1]),
            _CMP_LT_OQ,
        )) as u32;

        if mask != 0 {
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), d8);
            let mut m = mask;
            while m != 0 {
                let s = m.trailing_zeros() as usize;
                m &= m - 1;
                let di = buf[s];
                if di < top_dists[N - 1] {
                    let pos = top_dists.partition_point(|&x| x < di);
                    top_dists[pos..N].rotate_right(1);
                    top_dists[pos] = di;
                    top_idx[pos..N].rotate_right(1);
                    top_idx[pos] = ci + s;
                }
            }
        }
        ci += 8;
    }
    while ci < k {
        let di = *dp.add(ci);
        if di < top_dists[N - 1] {
            let pos = top_dists.partition_point(|&x| x < di);
            top_dists[pos..N].rotate_right(1);
            top_dists[pos] = di;
            top_idx[pos..N].rotate_right(1);
            top_idx[pos] = ci;
        }
        ci += 1;
    }
    top_idx
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_and_count(probes: &[usize], blob: &Blob, q_vecs: &[__m256; VECTOR_DIM]) -> u8 {
    let offsets = blob.cluster_offsets();
    let blocks_ptr = blob.blocks_ptr();
    let labels_ptr = blob.labels_ptr();

    // top-5 maintenance: (distance, label-bit). Distance is large at start
    // so any real distance pushes it out.
    let mut top: [(f32, u8); TOP_K] = [(f32::INFINITY, 0); TOP_K];
    let mut worst_idx = 0usize;

    for &ci in probes {
        let start = offsets[ci] as usize;
        let end = offsets[ci + 1] as usize;
        scan_blocks(
            q_vecs,
            blocks_ptr,
            labels_ptr,
            start,
            end,
            &mut top,
            &mut worst_idx,
        );
    }

    // Distance-weighted vote: weight each top-5 neighbor by 1/(dist+eps).
    // Same answer as plain count for evenly-spaced neighbors; for uneven
    // distributions it pulls the call toward whichever side is closer.
    // round((fraud_weight / total_weight) * 5) → still 0..=5.
    let eps = 1e-6_f32;
    let mut total_w = 0.0_f32;
    let mut fraud_w = 0.0_f32;
    for &(d, label) in top.iter() {
        let w = 1.0 / (d + eps);
        total_w += w;
        if label == 1 {
            fraud_w += w;
        }
    }
    let frac = if total_w > 0.0 { fraud_w / total_w } else { 0.0 };
    (frac * (TOP_K as f32)).round() as u8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks(
    q_vecs: &[__m256; VECTOR_DIM],
    blocks_ptr: *const i16,
    labels_ptr: *const u8,
    start_block: usize,
    end_block: usize,
    top: &mut [(f32, u8); TOP_K],
    worst_idx: &mut usize,
) {
    let scale = _mm256_set1_ps(QUANT_INV_SCALE);

    // Each iteration of the inner loop computes the squared distance for 8
    // vectors in parallel against the broadcasted query. The block layout is
    // SoA: 8 lanes of dim 0 contiguous, then 8 lanes of dim 1, etc.
    macro_rules! dim_pair {
        ($acc0:expr, $acc1:expr, $bb:expr, $d:expr) => {{
            let r0 = _mm_loadu_si128(blocks_ptr.add($bb + $d * BLOCK_VECS) as *const __m128i);
            let r1 = _mm_loadu_si128(blocks_ptr.add($bb + ($d + 1) * BLOCK_VECS) as *const __m128i);
            let v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0)), scale);
            let v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1)), scale);
            let d0 = _mm256_sub_ps(v0, q_vecs[$d]);
            let d1 = _mm256_sub_ps(v1, q_vecs[$d + 1]);
            $acc0 = _mm256_fmadd_ps(d0, d0, $acc0);
            $acc1 = _mm256_fmadd_ps(d1, d1, $acc1);
        }};
    }

    'block: for block_i in start_block..end_block {
        // Prefetch 8 blocks ahead (two cache lines each since block is 224 B).
        let prefetch_block = block_i + 8;
        if prefetch_block < end_block {
            _mm_prefetch(
                blocks_ptr.add(prefetch_block * BLOCK_I16) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                blocks_ptr.add(prefetch_block * BLOCK_I16 + 56) as *const i8,
                _MM_HINT_T0,
            );
        }

        let bb = block_i * BLOCK_I16;
        let threshold = _mm256_set1_ps(top[*worst_idx].0);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // First 8 of 14 dims (we have 14 = 2*7 dims, processed in pairs).
        dim_pair!(acc0, acc1, bb, 0);
        dim_pair!(acc0, acc1, bb, 2);
        dim_pair!(acc0, acc1, bb, 4);
        dim_pair!(acc0, acc1, bb, 6);

        // Early termination: if all 8 partial sums already exceed our current
        // worst-of-top-5, no point computing dims 8-13.
        let partial = _mm256_add_ps(acc0, acc1);
        if _mm256_movemask_ps(_mm256_cmp_ps(partial, threshold, _CMP_LT_OQ)) == 0 {
            continue 'block;
        }

        // Finish dims 8..14 (8, 10, 12 — three pairs covering dims 8..14).
        dim_pair!(acc0, acc1, bb, 8);
        dim_pair!(acc0, acc1, bb, 10);
        dim_pair!(acc0, acc1, bb, 12);

        let acc = _mm256_add_ps(acc0, acc1);
        let mut mask = _mm256_movemask_ps(_mm256_cmp_ps(acc, threshold, _CMP_LT_OQ)) as u32;
        if mask == 0 {
            continue;
        }

        let mut dists_buf = [0.0f32; BLOCK_VECS];
        _mm256_storeu_ps(dists_buf.as_mut_ptr(), acc);
        let label_byte_base = block_i;
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let di = dists_buf[slot];
            if di < top[*worst_idx].0 {
                // Fetch the fraud label bit for this slot.
                let bit_idx = label_byte_base * BLOCK_VECS + slot;
                let byte = *labels_ptr.add(bit_idx / 8);
                let label = ((byte >> (bit_idx % 8)) & 1) as u8;
                top[*worst_idx] = (di, label);
                let mut wi = 0;
                let mut wv = top[0].0;
                for j in 1..TOP_K {
                    if top[j].0 > wv {
                        wv = top[j].0;
                        wi = j;
                    }
                }
                *worst_idx = wi;
            }
        }
    }
}
