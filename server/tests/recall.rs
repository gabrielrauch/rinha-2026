// Offline recall validation: compares the IVF `fraud_score` against an exact
// brute-force scan over all vectors in the blob.
//
// Brute-force is ~30ms per query on this blob, so this test runs `#[ignore]`
// by default. Run with:
//
//   cargo test -p server --release --test recall -- --ignored --nocapture
//
// Requires `tmp/blob.bin` to exist (build via the builder crate).

use server::blob::Blob;
use server::distance::l2_squared;
use server::index::fraud_score;
use shared::VECTOR_DIM;
use std::path::PathBuf;
use std::time::Instant;

fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

fn random_query(rng: &mut u64) -> [i8; VECTOR_DIM] {
    let mut q = [0i8; VECTOR_DIM];
    for d in q.iter_mut() {
        // Bias toward [0, 127] (real queries are clamped to unit then quantized);
        // 1/8 of values get the -127 null sentinel.
        let r = xorshift64(rng);
        *d = if (r & 0b111) == 0 {
            -127
        } else {
            ((r >> 8) & 0x7F) as i8
        };
    }
    q
}

fn brute_force_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let vectors = blob.vectors();
    let mut top: [(i32, u32); 5] = [(i32::MAX, u32::MAX); 5];
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

#[test]
#[ignore]
fn recall_matches_brute_force() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).expect("tmp/blob.bin not found");

    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let n_queries = 200usize;

    let mut mismatches = 0usize;
    let mut score_diffs = [0usize; 6]; // |exact - approx| histogram, 0..=5
    let t0 = Instant::now();

    for q_idx in 0..n_queries {
        let q = random_query(&mut rng);
        let exact = brute_force_score(&blob, &q);
        let approx = fraud_score(&blob, &q);
        let diff = exact.abs_diff(approx) as usize;
        score_diffs[diff.min(5)] += 1;
        if exact != approx {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!("query {q_idx}: exact={exact} approx={approx}");
            }
        }
    }

    let elapsed = t0.elapsed();
    let rate = mismatches as f64 / n_queries as f64;
    eprintln!("recall test: {n_queries} queries in {:?}", elapsed);
    eprintln!("  mismatches: {mismatches} ({:.2}%)", rate * 100.0);
    eprintln!("  diff histogram (|exact-approx|): {:?}", score_diffs);

    // The /fraud-score endpoint maps count to {approved} via threshold 3 (≥3 → denied).
    // What ultimately matters is the boolean approved/denied; raw count diffs of 1 around
    // {2,3} flip the decision. Allow up to 1% raw-count mismatch — that's the slack the
    // adaptive 24-probe fallback should comfortably hit.
    assert!(rate < 0.01, "mismatch rate {rate} ≥ 1%");
}
