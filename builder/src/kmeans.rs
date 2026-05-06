use rayon::prelude::*;
use shared::VECTOR_DIM;

/// Simple xorshift64 PRNG; deterministic given a seed.
#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

#[inline]
fn dist_sq(a: &[i8; VECTOR_DIM], b: &[i8; VECTOR_DIM]) -> i32 {
    let mut s: i32 = 0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = *av as i32 - *bv as i32;
        s += d * d;
    }
    s
}

pub fn kmeans(
    vectors: &[[i8; VECTOR_DIM]],
    k: usize,
    iterations: usize,
    seed: u64,
) -> (Vec<[i8; VECTOR_DIM]>, Vec<u32>) {
    assert!(!vectors.is_empty());
    assert!(k <= vectors.len());

    let mut rng = Rng(seed | 1);

    // k-means++ init
    let mut centroids: Vec<[i8; VECTOR_DIM]> = Vec::with_capacity(k);
    centroids.push(vectors[rng.range(vectors.len())]);
    while centroids.len() < k {
        // PARALLEL: for each vector, find squared distance to nearest centroid so far
        let best_d: Vec<i32> = vectors
            .par_iter()
            .map(|v| centroids.iter().map(|c| dist_sq(v, c)).min().unwrap_or(0))
            .collect();
        let total: u64 = best_d.par_iter().map(|&x| x as u64).sum();
        if total == 0 {
            centroids.push(vectors[rng.range(vectors.len())]);
            continue;
        }
        let pick = rng.next() % total;
        let mut acc: u64 = 0;
        for (i, &d) in best_d.iter().enumerate() {
            acc += d as u64;
            if acc >= pick {
                centroids.push(vectors[i]);
                break;
            }
        }
    }

    let mut assignments = vec![0u32; vectors.len()];
    for _ in 0..iterations {
        // PARALLEL assign: each vector picks its nearest centroid independently
        assignments
            .par_iter_mut()
            .zip(vectors.par_iter())
            .for_each(|(a, v)| {
                let mut best = 0u32;
                let mut best_d = i32::MAX;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = dist_sq(v, c);
                    if d < best_d {
                        best_d = d;
                        best = ci as u32;
                    }
                }
                *a = best;
            });

        // PARALLEL update via fold/reduce: per-thread (sums, counts), then merge.
        let (sums, counts): (Vec<[i64; VECTOR_DIM]>, Vec<u64>) = vectors
            .par_iter()
            .zip(assignments.par_iter())
            .fold(
                || (vec![[0i64; VECTOR_DIM]; k], vec![0u64; k]),
                |(mut s, mut c), (v, &a)| {
                    let ci = a as usize;
                    c[ci] += 1;
                    for (d, &val) in v.iter().enumerate() {
                        s[ci][d] += val as i64;
                    }
                    (s, c)
                },
            )
            .reduce(
                || (vec![[0i64; VECTOR_DIM]; k], vec![0u64; k]),
                |(mut sa, mut ca), (sb, cb)| {
                    for ci in 0..k {
                        ca[ci] += cb[ci];
                        for d in 0..VECTOR_DIM {
                            sa[ci][d] += sb[ci][d];
                        }
                    }
                    (sa, ca)
                },
            );

        for c in 0..k {
            if counts[c] == 0 {
                centroids[c] = vectors[rng.range(vectors.len())];
            } else {
                for (d, sum) in sums[c].iter().enumerate() {
                    centroids[c][d] = (sum / counts[c] as i64) as i8;
                }
            }
        }
    }

    (centroids, assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_two_obvious_clusters() {
        let group_a = [10i8; VECTOR_DIM];
        let group_b = [-10i8; VECTOR_DIM];
        let mut data = vec![group_a; 50];
        data.extend(vec![group_b; 50]);

        let (centroids, assignments) = kmeans(&data, 2, 20, 42);
        assert_eq!(centroids.len(), 2);
        assert_eq!(assignments.len(), 100);
        let first_50: std::collections::HashSet<u32> = assignments[..50].iter().copied().collect();
        let last_50: std::collections::HashSet<u32> = assignments[50..].iter().copied().collect();
        assert_eq!(first_50.len(), 1);
        assert_eq!(last_50.len(), 1);
        assert_ne!(first_50, last_50);
    }
}
