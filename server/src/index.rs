use crate::blob::Blob;
use crate::distance::l2_squared;
use shared::VECTOR_DIM;

pub fn fraud_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let centroids = blob.centroids();
    let mut best_centroid: u32 = 0;
    let mut best_d: i32 = i32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = l2_squared(query, c);
        if d < best_d {
            best_d = d;
            best_centroid = i as u32;
        }
    }

    let offsets = blob.cluster_offsets();
    let start = offsets[best_centroid as usize] as usize;
    let end = offsets[best_centroid as usize + 1] as usize;
    let vectors = blob.vectors();

    // top-5 nearest in this cluster — fixed array of (distance, vector_index)
    let mut top: [(i32, u32); 5] = [(i32::MAX, u32::MAX); 5];
    let mut worst_idx: usize = 0;
    for (relative_idx, vector) in vectors[start..end].iter().enumerate() {
        let idx = start + relative_idx;
        let d = l2_squared(query, vector);
        if d < top[worst_idx].0 {
            top[worst_idx] = (d, idx as u32);
            // recompute worst
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

    // count frauds among the 5
    let mut count: u8 = 0;
    for &(_, idx) in &top {
        if idx == u32::MAX {
            continue; // cluster smaller than 5 (rare)
        }
        if blob.is_fraud(idx) {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn returns_value_in_range() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
        let blob = Blob::open(&path).unwrap();
        let q = [0i8; VECTOR_DIM];
        let score = fraud_score(&blob, &q);
        assert!(score <= 5);
    }
}
