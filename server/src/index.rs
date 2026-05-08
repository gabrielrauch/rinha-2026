use crate::blob::Blob;
use crate::distance::l2_squared;
use shared::VECTOR_DIM;

// nprobe=2: scan the two clusters whose centroids are closest to the query, then
// pick top-5 across both. All state lives on the stack — no heap allocation per
// request. Single pass over centroids tracks the two smallest distances.

pub fn fraud_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let centroids = blob.centroids();

    let mut best_d: i32 = i32::MAX;
    let mut best_i: u32 = 0;
    let mut second_d: i32 = i32::MAX;
    let mut second_i: u32 = 0;
    for (i, c) in centroids.iter().enumerate() {
        let d = l2_squared(query, c);
        if d < best_d {
            second_d = best_d;
            second_i = best_i;
            best_d = d;
            best_i = i as u32;
        } else if d < second_d {
            second_d = d;
            second_i = i as u32;
        }
    }

    let offsets = blob.cluster_offsets();
    let vectors = blob.vectors();

    let mut top: [(i32, u32); 5] = [(i32::MAX, u32::MAX); 5];
    let mut worst_idx: usize = 0;

    let probes: [u32; 2] = [best_i, second_i];
    for &ci in &probes {
        let start = offsets[ci as usize] as usize;
        let end = offsets[ci as usize + 1] as usize;
        for (rel, vector) in vectors[start..end].iter().enumerate() {
            let d = l2_squared(query, vector);
            if d < top[worst_idx].0 {
                top[worst_idx] = (d, (start + rel) as u32);
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
