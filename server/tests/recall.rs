//! Offline recall validation. Compares HNSW search against an exact
//! brute-force scan over all 3M vectors.
//!
//! Run with:
//!   cargo test -p server --release --test recall -- --ignored --nocapture
//!
//! Requires `tmp/blob.bin` to exist (build via the builder crate).

use server::blob::Blob;
use server::distance::l2_squared;
use server::hnsw::{search_top_k, TOP_K};
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
        let r = xorshift64(rng);
        *d = if (r & 0b111) == 0 {
            -127
        } else {
            ((r >> 8) & 0x7F) as i8
        };
    }
    q
}

fn brute_force_top_k(blob: &Blob, query: &[i8; VECTOR_DIM]) -> [(u32, u32); 5] {
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

    // Sort ascending by distance, return as (u32, u32)
    top.sort_unstable_by_key(|&(d, _)| d);
    let mut out = [(u32::MAX, u32::MAX); 5];
    for (i, &(d, n)) in top.iter().enumerate() {
        out[i] = (d as u32, n);
    }
    out
}

fn brute_force_fraud_count(blob: &Blob, top: &[(u32, u32); 5]) -> u8 {
    let mut count = 0u8;
    for &(_, idx) in top {
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
fn hnsw_check_back_edges() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).expect("blob");
    let m0 = blob.header().hnsw_m0 as usize;

    // node 0's outgoing neighbors at layer 0
    let mut node0_out: Vec<u32> = Vec::new();
    for slot in 0..m0 {
        let nb = blob.hnsw_neighbor(0, 0 * m0 + slot);
        if nb == shared::HNSW_SENTINEL {
            break;
        }
        node0_out.push(nb);
    }
    eprintln!("node 0 outgoing: {:?}", node0_out);

    // For each of those neighbors, check if 0 is in their list
    for &n in &node0_out {
        let mut neigh: Vec<u32> = Vec::new();
        for slot in 0..m0 {
            let v = blob.hnsw_neighbor(0, n as usize * m0 + slot);
            if v == shared::HNSW_SENTINEL {
                break;
            }
            neigh.push(v);
        }
        let has0 = neigh.contains(&0);
        eprintln!(
            "  node {} neighbors: {:?}  has_0={}",
            n, neigh, has0
        );
    }
}

#[test]
#[ignore]
fn hnsw_dump_graph_stats() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).expect("tmp/blob.bin not found");

    eprintln!("=== blob ===");
    eprintln!("total_vectors: {}", blob.header().total_vectors);
    eprintln!("hnsw_num_layers: {}", blob.hnsw_num_layers());
    eprintln!("hnsw_entry_point: {}", blob.hnsw_entry_point());
    eprintln!("hnsw_m0: {}", blob.header().hnsw_m0);
    eprintln!("hnsw_m: {}", blob.header().hnsw_m);

    // Layer 0 neighbor stats
    let n = blob.header().total_vectors as usize;
    let m0 = blob.header().hnsw_m0 as usize;
    let mut empty_nodes = 0;
    let mut total_neighbors = 0;
    let mut min_neighbors = m0;
    let mut max_neighbors = 0;
    let sample = [0u32, 1, 100, 1000, n as u32 / 2, n as u32 - 1];
    for i in 0..n {
        let mut count = 0;
        for slot in 0..m0 {
            let nb = blob.hnsw_neighbor(0, i * m0 + slot);
            if nb == shared::HNSW_SENTINEL {
                break;
            }
            count += 1;
        }
        total_neighbors += count;
        if count == 0 {
            empty_nodes += 1;
        }
        if count < min_neighbors {
            min_neighbors = count;
        }
        if count > max_neighbors {
            max_neighbors = count;
        }
        if sample.contains(&(i as u32)) {
            let mut neighbors = vec![];
            for slot in 0..m0 {
                let nb = blob.hnsw_neighbor(0, i * m0 + slot);
                if nb == shared::HNSW_SENTINEL {
                    break;
                }
                neighbors.push(nb);
            }
            eprintln!("  node {} layer-0 neighbors: {:?}", i, neighbors);
        }
    }
    eprintln!(
        "Layer 0: avg degree {:.1}, min {}, max {}, empty {}",
        total_neighbors as f64 / n as f64, min_neighbors, max_neighbors, empty_nodes
    );

    // Layer 1 stats
    if blob.hnsw_num_layers() > 1 {
        let l1_nodes = blob.hnsw_layer_nodes(1);
        eprintln!("Layer 1: {} nodes", l1_nodes.len());
        eprintln!("  first 10 zero-ids: {:?}", &l1_nodes[..10.min(l1_nodes.len())]);
    }
}

/// Probe sanity: query MUST be in the index (we use vector i as query).
/// HNSW search should return that vector at distance 0 at position 0.
/// If this fails, the build/search is fundamentally broken.
#[test]
#[ignore]
fn hnsw_finds_indexed_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).expect("tmp/blob.bin not found");
    let vectors = blob.vectors();
    let n = vectors.len();
    eprintln!("blob: {n} vectors, hnsw_layers={} entry={}", blob.hnsw_num_layers(), blob.hnsw_entry_point());

    let queries: [u32; 5] = [
        blob.hnsw_entry_point(),
        0,
        (n / 4) as u32,
        (n / 2) as u32,
        (n - 1) as u32,
    ];

    let mut hits = 0usize;
    for &qi in &queries {
        let q = vectors[qi as usize];
        let top = search_top_k(&blob, &q);
        eprintln!("query={qi}: top5 = {:?}", top);
        let found_self = top.iter().any(|&(d, idx)| idx == qi && d == 0);
        if found_self {
            hits += 1;
            eprintln!("  ✓ found self");
        } else {
            eprintln!("  ✗ DID NOT find self");
        }
    }
    assert_eq!(hits, queries.len(), "HNSW failed to find indexed vectors as themselves");
}

#[test]
#[ignore]
fn hnsw_recall_with_real_queries() {
    // Use actual indexed vectors as queries (with tiny noise) so the queries
    // live in the same distribution as the data. random_query falls outside
    // the i8 manifold the index was built over, so HNSW does poorly there
    // even when the graph is well-connected.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).expect("blob");
    let vectors = blob.vectors();
    let n = vectors.len();

    let mut rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let n_queries = 100usize;
    let mut count_mismatches = 0usize;
    let mut count_diff_total: i32 = 0;
    let mut neighbor_overlap_total: usize = 0;
    let t0 = Instant::now();

    for q_idx in 0..n_queries {
        let r = xorshift64(&mut rng);
        let pick = (r as usize) % n;
        // Query = vector[pick] with at most ±2 noise per dimension
        let mut q = vectors[pick];
        for d in q.iter_mut() {
            let n2 = (xorshift64(&mut rng) as i8) % 3 - 1;
            *d = d.saturating_add(n2);
        }

        let exact = brute_force_top_k(&blob, &q);
        let exact_count = brute_force_fraud_count(&blob, &exact);

        let approx = search_top_k(&blob, &q);
        let approx_count = fraud_score(&blob, &q);

        let exact_ids: std::collections::HashSet<u32> =
            exact.iter().map(|&(_, n)| n).filter(|&n| n != u32::MAX).collect();
        let approx_ids: std::collections::HashSet<u32> =
            approx.iter().map(|&(_, n)| n).filter(|&n| n != u32::MAX).collect();
        let overlap = exact_ids.intersection(&approx_ids).count();
        neighbor_overlap_total += overlap;

        if exact_count != approx_count {
            count_mismatches += 1;
            count_diff_total += (exact_count as i32 - approx_count as i32).abs();
            if count_mismatches <= 5 {
                eprintln!(
                    "q{q_idx} (pick={pick}): exact_count={exact_count} approx_count={approx_count} overlap={overlap}/5"
                );
            }
        }
    }

    let elapsed = t0.elapsed();
    let mismatch_rate = count_mismatches as f64 / n_queries as f64;
    let avg_overlap = neighbor_overlap_total as f64 / (n_queries as f64 * 5.0);
    eprintln!("real-query recall test: {n_queries} queries in {:?}", elapsed);
    eprintln!(
        "  count mismatches: {count_mismatches}/{n_queries} ({:.1}%)",
        mismatch_rate * 100.0
    );
    eprintln!("  avg neighbor overlap: {:.1}%", avg_overlap * 100.0);
    eprintln!("  total count diff: {count_diff_total}");
}

#[test]
#[ignore]
fn hnsw_recall_matches_brute_force() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).expect("tmp/blob.bin not found");

    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let n_queries = 100usize;

    let mut count_mismatches = 0usize;
    let mut count_diff_total: i32 = 0;
    let mut neighbor_overlap_total: usize = 0;
    let t0 = Instant::now();

    for q_idx in 0..n_queries {
        let q = random_query(&mut rng);

        let exact = brute_force_top_k(&blob, &q);
        let exact_count = brute_force_fraud_count(&blob, &exact);

        let approx = search_top_k(&blob, &q);
        let approx_count = fraud_score(&blob, &q);

        // How many of the exact top-5 are in the approximate top-5?
        let exact_ids: std::collections::HashSet<u32> =
            exact.iter().map(|&(_, n)| n).filter(|&n| n != u32::MAX).collect();
        let approx_ids: std::collections::HashSet<u32> =
            approx.iter().map(|&(_, n)| n).filter(|&n| n != u32::MAX).collect();
        let overlap = exact_ids.intersection(&approx_ids).count();
        neighbor_overlap_total += overlap;

        if exact_count != approx_count {
            count_mismatches += 1;
            count_diff_total += (exact_count as i32 - approx_count as i32).abs();
            if count_mismatches <= 10 {
                eprintln!(
                    "q{q_idx}: exact_count={exact_count} approx_count={approx_count} overlap={overlap}/5"
                );
            }
        }
    }

    let elapsed = t0.elapsed();
    let mismatch_rate = count_mismatches as f64 / n_queries as f64;
    let avg_overlap = neighbor_overlap_total as f64 / (n_queries as f64 * 5.0);
    eprintln!("recall test: {n_queries} queries in {:?}", elapsed);
    eprintln!(
        "  count mismatches: {count_mismatches}/{n_queries} ({:.1}%)",
        mismatch_rate * 100.0
    );
    eprintln!("  avg neighbor overlap: {:.1}%", avg_overlap * 100.0);
    eprintln!("  total count diff (sum of |exact-approx|): {count_diff_total}");

    assert!(
        avg_overlap > 0.80,
        "neighbor overlap {} too low",
        avg_overlap
    );
    assert!(
        mismatch_rate < 0.05,
        "fraud-count mismatch rate {} >= 5%",
        mismatch_rate
    );
}
