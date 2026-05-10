//! Build the IVF SoA blob. Layout v3:
//!  - header
//!  - centroids: f32 SoA, dim-major (dim 0 of all K, then dim 1, ...)
//!  - cluster_offsets: u32, K+1 entries, indexed by cluster id → starting block index
//!  - blocks: i16 SoA, each block is 14 dims × 8 vecs (dim 0 slot 0..7, dim 1 slot 0..7, ...)
//!  - labels: 1 bit per (block_idx*8 + slot_idx), 1 = fraud, padding slots = 0
//!  - mcc table: 1024 × i8

use crate::quantize::quantize_dim;
use shared::*;
use std::collections::HashMap;

pub struct BuildInputs<'a> {
    pub centroids: &'a [[f32; VECTOR_DIM]],
    pub assignments: &'a [u32],
    pub vectors_f32: &'a [[f32; VECTOR_DIM]],
    pub is_fraud: &'a [bool],
    pub mcc_risk: &'a HashMap<u32, f32>,
}

pub fn build_blob(inp: &BuildInputs) -> Vec<u8> {
    let n = inp.vectors_f32.len();
    let k = inp.centroids.len();
    assert_eq!(inp.assignments.len(), n);
    assert_eq!(inp.is_fraud.len(), n);

    // 1. Group vector indices by cluster.
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, &a) in inp.assignments.iter().enumerate() {
        buckets[a as usize].push(i as u32);
    }

    // 2. Within each cluster, sort by distance to centroid ascending.
    //    Vectors closest to centroid (most "typical") come first → early termination
    //    in the search scan filters more aggressively on far candidates.
    for (ci, bucket) in buckets.iter_mut().enumerate() {
        let centroid = &inp.centroids[ci];
        bucket.sort_unstable_by(|&a, &b| {
            let da = dist_sq_f32(&inp.vectors_f32[a as usize], centroid);
            let db = dist_sq_f32(&inp.vectors_f32[b as usize], centroid);
            da.total_cmp(&db)
        });
    }

    // 3. Compute cluster_offsets: each cluster occupies ceil(size / BLOCK_VECS) blocks.
    let mut cluster_offsets: Vec<u32> = Vec::with_capacity(k + 1);
    cluster_offsets.push(0);
    let mut acc: u32 = 0;
    for bucket in &buckets {
        let n_blocks = (bucket.len() + BLOCK_VECS - 1) / BLOCK_VECS;
        acc += n_blocks as u32;
        cluster_offsets.push(acc);
    }
    let total_blocks = acc as usize;
    let padded_n = total_blocks * BLOCK_VECS;

    // 4. Allocate output buffers.
    let mut out_blocks: Vec<i16> = vec![0i16; total_blocks * VECTOR_DIM * BLOCK_VECS];
    let mut labels = vec![0u8; (padded_n + 7) / 8];

    // 5. Pack each cluster's vectors into SoA blocks.
    for (ci, bucket) in buckets.iter().enumerate() {
        let block_start = cluster_offsets[ci] as usize;
        for (slot_global, &vi) in bucket.iter().enumerate() {
            let block_offset = block_start + slot_global / BLOCK_VECS;
            let slot = slot_global % BLOCK_VECS;
            let base = block_offset * VECTOR_DIM * BLOCK_VECS;
            let v = &inp.vectors_f32[vi as usize];
            for d in 0..VECTOR_DIM {
                out_blocks[base + d * BLOCK_VECS + slot] = quantize_dim(v[d]);
            }
            if inp.is_fraud[vi as usize] {
                let bit = block_offset * BLOCK_VECS + slot;
                labels[bit / 8] |= 1 << (bit % 8);
            }
        }
        // Pad unused slots with i16::MAX so their distance computes to "infinity".
        let used = bucket.len();
        let last_block_used = used % BLOCK_VECS;
        if last_block_used != 0 && !bucket.is_empty() {
            let block_offset = block_start + used / BLOCK_VECS;
            let base = block_offset * VECTOR_DIM * BLOCK_VECS;
            for slot in last_block_used..BLOCK_VECS {
                for d in 0..VECTOR_DIM {
                    out_blocks[base + d * BLOCK_VECS + slot] = i16::MAX;
                }
            }
        }
    }

    // 6. Centroides as f32 dim-major SoA.
    let mut centroids_soa: Vec<f32> = Vec::with_capacity(VECTOR_DIM * k);
    for d in 0..VECTOR_DIM {
        for ci in 0..k {
            centroids_soa.push(inp.centroids[ci][d]);
        }
    }

    // 7. MCC table (i8, scale -127..127).
    let mut mcc_table = [(0.5f32 * 127.0).round() as i8; MCC_TABLE_SIZE];
    for (&mcc, &risk) in inp.mcc_risk {
        let q = (risk.clamp(0.0, 1.0) * 127.0).round() as i8;
        mcc_table[(mcc as usize) % MCC_TABLE_SIZE] = q;
    }

    // 8. Layout offsets.
    let header_size = std::mem::size_of::<BlobHeader>() as u32;
    let centroids_bytes = (VECTOR_DIM * k * 4) as u32;
    let cluster_offsets_bytes = ((k + 1) * 4) as u32;
    let blocks_bytes = (total_blocks * VECTOR_DIM * BLOCK_VECS * 2) as u32;
    let labels_bytes = ((padded_n + 7) / 8) as u32;
    let mcc_bytes = MCC_TABLE_SIZE as u32;

    let centroids_offset = header_size;
    let cluster_offsets_offset = centroids_offset + centroids_bytes;
    let blocks_offset = cluster_offsets_offset + cluster_offsets_bytes;
    let labels_offset = blocks_offset + blocks_bytes;
    let mcc_table_offset = labels_offset + labels_bytes;
    let blob_size = mcc_table_offset + mcc_bytes;

    let header = BlobHeader {
        magic: MAGIC,
        version: VERSION,
        total_vectors: n as u32,
        padded_n: padded_n as u32,
        total_blocks: total_blocks as u32,
        k_centroids: k as u32,
        centroids_offset,
        cluster_offsets_offset,
        blocks_offset,
        labels_offset,
        mcc_table_offset,
        blob_size,
        _padding: [0; 204],
    };

    // 9. Write out.
    let mut out = Vec::with_capacity(blob_size as usize);
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const BlobHeader as *const u8,
            std::mem::size_of::<BlobHeader>(),
        )
    };
    out.extend_from_slice(header_bytes);
    let centroids_b = unsafe {
        std::slice::from_raw_parts(centroids_soa.as_ptr() as *const u8, centroids_bytes as usize)
    };
    out.extend_from_slice(centroids_b);
    for &o in &cluster_offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    let blocks_b = unsafe {
        std::slice::from_raw_parts(out_blocks.as_ptr() as *const u8, blocks_bytes as usize)
    };
    out.extend_from_slice(blocks_b);
    out.extend_from_slice(&labels);
    let mcc_b =
        unsafe { std::slice::from_raw_parts(mcc_table.as_ptr() as *const u8, MCC_TABLE_SIZE) };
    out.extend_from_slice(mcc_b);

    debug_assert_eq!(out.len(), blob_size as usize, "blob size mismatch");
    out
}

#[inline]
fn dist_sq_f32(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..VECTOR_DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
