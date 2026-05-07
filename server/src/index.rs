use crate::blob::Blob;
use crate::distance::l2_squared;
use shared::VECTOR_DIM;

const FAST_NPROBE: usize = 8;
const FULL_NPROBE: usize = 24;
const TOP_K: usize = 5;

pub fn fraud_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let centroids = blob.centroids();
    let mut by_dist: Vec<(i32, u32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (l2_squared(query, c), i as u32))
        .collect();
    by_dist.sort_unstable_by_key(|&(d, _)| d);

    let fast = scan_clusters(blob, &by_dist[..FAST_NPROBE.min(by_dist.len())], query);

    // The threshold (3/5 = 0.6) is ambiguous when count ∈ {2,3}: a single
    // misplaced neighbor flips the decision. Refine to a wider probe.
    if fast == 2 || fast == 3 {
        scan_clusters(blob, &by_dist[..FULL_NPROBE.min(by_dist.len())], query)
    } else {
        fast
    }
}

fn scan_clusters(blob: &Blob, probes: &[(i32, u32)], query: &[i8; VECTOR_DIM]) -> u8 {
    let offsets = blob.cluster_offsets();
    let vectors = blob.vectors();

    let mut top: [(i32, u32); TOP_K] = [(i32::MAX, u32::MAX); TOP_K];
    let mut worst_idx: usize = 0;

    for &(_, ci) in probes {
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
