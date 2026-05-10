//! K-means clustering in f32 space. Used to partition the dataset into
//! IVF cells. We work in float (not the i8-quantized space) because the
//! reference vectors come in as f32 and we'd lose precision if we
//! re-quantized during clustering.

use rayon::prelude::*;
use shared::VECTOR_DIM;

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
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
fn dist_sq(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..VECTOR_DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Returns `(centroids, assignments)`.
pub fn kmeans(
    vectors: &[[f32; VECTOR_DIM]],
    k: usize,
    iterations: usize,
    seed: u64,
) -> (Vec<[f32; VECTOR_DIM]>, Vec<u32>) {
    assert!(!vectors.is_empty());
    let k = k.min(vectors.len());

    let mut rng = Rng::new(seed);

    // kmeans++ init with rolling best_d (parallel).
    let mut centroids: Vec<[f32; VECTOR_DIM]> = Vec::with_capacity(k);
    centroids.push(vectors[rng.range(vectors.len())]);
    let first = centroids[0];
    let mut best_d: Vec<f32> = vectors.par_iter().map(|v| dist_sq(v, &first)).collect();

    while centroids.len() < k {
        let total: f64 = best_d.par_iter().map(|&x| x as f64).sum();
        let new_centroid = if total == 0.0 {
            vectors[rng.range(vectors.len())]
        } else {
            let pick = rng.next() as f64 / u64::MAX as f64 * total;
            let mut acc = 0.0f64;
            let mut chosen = vectors[rng.range(vectors.len())];
            for (i, &d) in best_d.iter().enumerate() {
                acc += d as f64;
                if acc >= pick {
                    chosen = vectors[i];
                    break;
                }
            }
            chosen
        };
        let new_c = new_centroid;
        centroids.push(new_c);
        best_d
            .par_iter_mut()
            .zip(vectors.par_iter())
            .for_each(|(d, v)| {
                let nd = dist_sq(v, &new_c);
                if nd < *d {
                    *d = nd;
                }
            });
    }

    let mut assignments = vec![0u32; vectors.len()];
    for it in 0..iterations {
        assignments
            .par_iter_mut()
            .zip(vectors.par_iter())
            .for_each(|(a, v)| {
                let mut best = 0u32;
                let mut best_d = f32::INFINITY;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = dist_sq(v, c);
                    if d < best_d {
                        best_d = d;
                        best = ci as u32;
                    }
                }
                *a = best;
            });

        let (sums, counts): (Vec<[f64; VECTOR_DIM]>, Vec<u64>) = vectors
            .par_iter()
            .zip(assignments.par_iter())
            .fold(
                || (vec![[0.0f64; VECTOR_DIM]; k], vec![0u64; k]),
                |(mut s, mut c), (v, &a)| {
                    let ci = a as usize;
                    c[ci] += 1;
                    for (d, &val) in v.iter().enumerate() {
                        s[ci][d] += val as f64;
                    }
                    (s, c)
                },
            )
            .reduce(
                || (vec![[0.0f64; VECTOR_DIM]; k], vec![0u64; k]),
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
                for d in 0..VECTOR_DIM {
                    centroids[c][d] = (sums[c][d] / counts[c] as f64) as f32;
                }
            }
        }
        eprintln!("  k-means iter {}/{}", it + 1, iterations);
    }

    (centroids, assignments)
}
