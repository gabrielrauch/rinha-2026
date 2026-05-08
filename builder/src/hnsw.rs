//! Build a HNSW graph over i8 quantized vectors.
//!
//! This implementation follows Malkov & Yashunin (2018), Algorithms 1, 2, 4, 5,
//! including the diverse-neighbor heuristic for good graph connectivity.

#![allow(clippy::ptr_arg)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_else_if)]

use rayon::prelude::*;
use shared::{HNSW_M, HNSW_M0, HNSW_SENTINEL, VECTOR_DIM};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const EF_CONSTRUCTION: usize = 200;

/// Final, compact graph ready for serialization.
pub struct BuiltGraph {
    pub num_nodes: usize,
    pub entry_point: u32,
    /// Includes layer 0 plus all higher layers actually populated.
    pub num_layers: usize,
    /// Layer 0 dense block: `num_nodes * HNSW_M0` u32 neighbor slots
    /// (HNSW_SENTINEL for empty).
    pub layer0_neighbors: Vec<u32>,
    /// One entry per non-zero layer (layers[0] is layer 1, etc.).
    pub upper_layers: Vec<UpperLayer>,
}

pub struct UpperLayer {
    /// Zero-node IDs that are members of this layer, sorted ascending.
    pub nodes: Vec<u32>,
    /// `nodes.len() * HNSW_M` u32 neighbor slots, parallel to `nodes`.
    pub neighbors: Vec<u32>,
}

/// xorshift64* PRNG, seeded.
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
    /// Random f64 in (0, 1).
    fn next_f64_open(&mut self) -> f64 {
        let bits = self.next() >> 11;
        let v = (bits as f64) / (1u64 << 53) as f64;
        if v <= 0.0 { 1e-300 } else { v }
    }
}

#[inline]
fn dist(vectors: &[[i8; VECTOR_DIM]], a: u32, b: u32) -> u32 {
    let av = &vectors[a as usize];
    let bv = &vectors[b as usize];
    let mut s: i32 = 0;
    for i in 0..VECTOR_DIM {
        let d = av[i] as i32 - bv[i] as i32;
        s += d * d;
    }
    s as u32
}

#[inline]
fn dist_to_q(vectors: &[[i8; VECTOR_DIM]], q: &[i8; VECTOR_DIM], n: u32) -> u32 {
    let nv = &vectors[n as usize];
    let mut s: i32 = 0;
    for i in 0..VECTOR_DIM {
        let d = q[i] as i32 - nv[i] as i32;
        s += d * d;
    }
    s as u32
}

/// Iterate over a node's neighbors at a given layer.
/// For layer 0 we use `layer0` (flat). For higher layers, we use `upper`.
struct UpperEditable {
    /// Map zero-node id → flat index of its M-slot bucket in `flat`.
    /// We use a Vec<i32> indexed by zero-node id; -1 = absent.
    /// (3 M nodes × 4 bytes = 12 MB, acceptable during build.)
    pos: Vec<i32>,
    /// Flat M slots per node currently in this layer, packed.
    flat: Vec<u32>,
    /// Zero-node IDs present at this layer, ordered by insertion (slot index).
    nodes: Vec<u32>,
}
impl UpperEditable {
    fn new(num_nodes: usize) -> Self {
        Self {
            pos: vec![-1i32; num_nodes],
            flat: Vec::new(),
            nodes: Vec::new(),
        }
    }
    fn ensure_node(&mut self, id: u32) {
        if self.pos[id as usize] < 0 {
            let slot = self.nodes.len();
            self.pos[id as usize] = slot as i32;
            self.nodes.push(id);
            self.flat
                .extend_from_slice(&[HNSW_SENTINEL; HNSW_M]);
        }
    }
    fn slot_for(&self, id: u32) -> Option<usize> {
        let p = self.pos[id as usize];
        if p < 0 { None } else { Some(p as usize) }
    }
    #[inline]
    fn neighbors(&self, id: u32) -> Option<&[u32]> {
        self.slot_for(id).map(|s| {
            let start = s * HNSW_M;
            &self.flat[start..start + HNSW_M]
        })
    }
    #[inline]
    fn neighbors_mut(&mut self, id: u32) -> &mut [u32] {
        let s = self.slot_for(id).expect("node not in layer");
        let start = s * HNSW_M;
        &mut self.flat[start..start + HNSW_M]
    }
    #[allow(dead_code)]
    fn contains(&self, id: u32) -> bool {
        self.pos[id as usize] >= 0
    }
}

/// Linear "max-cap" heap of (-distance, id) for keeping top-K closest.
/// Implemented as a sorted Vec for simplicity; ef_construction is small (~200)
/// so O(n) insertion is fine.
struct TopK {
    /// Sorted ascending by distance.
    items: Vec<(u32, u32)>,
    cap: usize,
}
impl TopK {
    fn new(cap: usize) -> Self {
        Self { items: Vec::with_capacity(cap + 1), cap }
    }
    fn worst_dist(&self) -> u32 {
        self.items.last().map(|&(d, _)| d).unwrap_or(u32::MAX)
    }
    fn len(&self) -> usize {
        self.items.len()
    }
    /// Insert if better than current worst (or under capacity).
    fn try_insert(&mut self, item: (u32, u32)) -> bool {
        if self.items.len() == self.cap && item.0 >= self.worst_dist() {
            return false;
        }
        let pos = self.items.partition_point(|&(d, _)| d < item.0);
        self.items.insert(pos, item);
        if self.items.len() > self.cap {
            self.items.pop();
        }
        true
    }
    fn into_sorted(self) -> Vec<(u32, u32)> {
        self.items
    }
}

/// Greedy descent at a non-zero layer: from `ep`, walk to neighbor with
/// smallest distance to `q`, repeat until no improvement.
fn greedy_descend(
    vectors: &[[i8; VECTOR_DIM]],
    q: &[i8; VECTOR_DIM],
    mut ep: u32,
    mut ep_dist: u32,
    layer: &UpperEditable,
) -> (u32, u32) {
    loop {
        let mut improved = false;
        if let Some(neighbors) = layer.neighbors(ep) {
            for &n in neighbors {
                if n == HNSW_SENTINEL {
                    break;
                }
                let d = dist_to_q(vectors, q, n);
                if d < ep_dist {
                    ep_dist = d;
                    ep = n;
                    improved = true;
                }
            }
        }
        if !improved {
            return (ep, ep_dist);
        }
    }
}

/// Beam search with ef candidates over a non-zero layer (sparse). Returns
/// `ef` closest, sorted ascending by distance.
fn beam_search_upper(
    vectors: &[[i8; VECTOR_DIM]],
    q: &[i8; VECTOR_DIM],
    ep: u32,
    ep_dist: u32,
    ef: usize,
    layer: &UpperEditable,
    visited: &mut Vec<u8>,
    visited_log: &mut Vec<u32>,
) -> Vec<(u32, u32)> {
    visited[ep as usize] = 1;
    visited_log.push(ep);

    // Min-heap of (distance, id) candidates to expand.
    let mut cands: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(ef * 2);
    cands.push(Reverse((ep_dist, ep)));
    let mut found = TopK::new(ef);
    found.try_insert((ep_dist, ep));

    while let Some(Reverse((d_c, c))) = cands.pop() {
        if d_c > found.worst_dist() && found.len() >= ef {
            break;
        }

        if let Some(neighbors) = layer.neighbors(c) {
            for &n in neighbors {
                if n == HNSW_SENTINEL {
                    break;
                }
                if visited[n as usize] != 0 {
                    continue;
                }
                visited[n as usize] = 1;
                visited_log.push(n);
                let d = dist_to_q(vectors, q, n);
                if d < found.worst_dist() || found.len() < ef {
                    if found.try_insert((d, n)) {
                        cands.push(Reverse((d, n)));
                    }
                }
            }
        }
    }

    found.into_sorted()
}

fn beam_search_layer0(
    vectors: &[[i8; VECTOR_DIM]],
    q: &[i8; VECTOR_DIM],
    ep: u32,
    ep_dist: u32,
    ef: usize,
    layer0: &[u32],
    visited: &mut Vec<u8>,
    visited_log: &mut Vec<u32>,
) -> Vec<(u32, u32)> {
    visited[ep as usize] = 1;
    visited_log.push(ep);

    let mut cands: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(ef * 2);
    cands.push(Reverse((ep_dist, ep)));
    let mut found = TopK::new(ef);
    found.try_insert((ep_dist, ep));

    while let Some(Reverse((d_c, c))) = cands.pop() {
        if d_c > found.worst_dist() && found.len() >= ef {
            break;
        }

        let base = c as usize * HNSW_M0;
        for slot in 0..HNSW_M0 {
            let n = layer0[base + slot];
            if n == HNSW_SENTINEL {
                break;
            }
            if visited[n as usize] != 0 {
                continue;
            }
            visited[n as usize] = 1;
            visited_log.push(n);
            let d = dist_to_q(vectors, q, n);
            if d < found.worst_dist() || found.len() < ef {
                if found.try_insert((d, n)) {
                    cands.push(Reverse((d, n)));
                }
            }
        }
    }

    found.into_sorted()
}

#[inline]
fn clear_visited(visited: &mut [u8], log: &mut Vec<u32>) {
    for &id in log.iter() {
        visited[id as usize] = 0;
    }
    log.clear();
}

fn random_level(rng: &mut Rng, max_level: usize) -> usize {
    let m_l = 1.0 / (HNSW_M as f64).ln();
    let u = rng.next_f64_open();
    let lvl = (-u.ln() * m_l).floor() as usize;
    lvl.min(max_level)
}

/// Insert a new node `q_id` with its preassigned `q_level`.
/// Mutates `layer0`, `uppers`, returns nothing.
#[allow(clippy::too_many_arguments)]
fn insert_node(
    q_id: u32,
    q_level: usize,
    vectors: &[[i8; VECTOR_DIM]],
    entry: &mut u32,
    max_level: &mut usize,
    layer0: &mut Vec<u32>,
    uppers: &mut [UpperEditable],
    visited: &mut Vec<u8>,
    visited_log: &mut Vec<u32>,
) {
    let q = &vectors[q_id as usize];
    let mut ep = *entry;
    let mut ep_dist = dist_to_q(vectors, q, ep);

    // Pre-register q at its target levels (so neighbors_mut() doesn't panic)
    for l in 1..=q_level {
        uppers[l - 1].ensure_node(q_id);
    }

    // Greedy descent from max_level down to q_level + 1
    let mut cur = *max_level;
    while cur > q_level {
        let (e, d) = greedy_descend(vectors, q, ep, ep_dist, &uppers[cur - 1]);
        ep = e;
        ep_dist = d;
        cur -= 1;
    }

    // Insert at layers q_level..=0
    let mut cur = cur.min(q_level);
    loop {
        clear_visited(visited, visited_log);

        let candidates = if cur == 0 {
            beam_search_layer0(vectors, q, ep, ep_dist, EF_CONSTRUCTION, layer0, visited, visited_log)
        } else {
            beam_search_upper(vectors, q, ep, ep_dist, EF_CONSTRUCTION, &uppers[cur - 1], visited, visited_log)
        };

        let m_use = if cur == 0 { HNSW_M0 } else { HNSW_M };

        // Pick q's neighbors using the diverse heuristic (Algorithm 4).
        let mut q_neighbors: [u32; HNSW_M0] = [HNSW_SENTINEL; HNSW_M0];
        let mut q_cands: Vec<(u32, u32)> = candidates.iter().map(|&(d, n)| (d, n)).collect();
        select_neighbors_heuristic(vectors, q_id, &mut q_cands, m_use, &mut q_neighbors);

        // Wire q → selected
        if cur == 0 {
            let base = q_id as usize * HNSW_M0;
            for (slot, &n) in q_neighbors.iter().take(m_use).enumerate() {
                layer0[base + slot] = n;
            }
        } else {
            let arr = uppers[cur - 1].neighbors_mut(q_id);
            for (slot, &n) in q_neighbors.iter().take(m_use).enumerate() {
                arr[slot] = n;
            }
        }

        // Wire selected → q (with prune via heuristic)
        for &n in q_neighbors.iter().take(m_use) {
            if n == HNSW_SENTINEL {
                break;
            }
            add_back_edge(vectors, n, q_id, cur, m_use, layer0, uppers);
        }

        // Set ep for next iteration
        if let Some(&(d_first, n_first)) = candidates.first() {
            ep = n_first;
            ep_dist = d_first;
        }

        if cur == 0 {
            break;
        }
        cur -= 1;
    }

    if q_level > *max_level {
        *max_level = q_level;
        *entry = q_id;
    }
}

/// SELECT-NEIGHBORS-HEURISTIC from the HNSW paper (Algorithm 4). Picks up to
/// `m_use` neighbors from `candidates` that are diverse — each chosen neighbor
/// must be closer to the query than to any already-chosen neighbor. This is
/// the difference between "keep closest M" (pure greedy, asymmetric graph)
/// and "keep diverse M" (well-connected graph).
fn select_neighbors_heuristic(
    vectors: &[[i8; VECTOR_DIM]],
    target: u32,
    candidates: &mut Vec<(u32, u32)>, // (dist_to_target, id) — will be sorted asc
    m_use: usize,
    out: &mut [u32; HNSW_M0],
) {
    candidates.sort_by_key(|&(d, _)| d);
    let mut r_count: usize = 0;
    let mut discarded: Vec<(u32, u32)> = Vec::with_capacity(candidates.len());
    let _ = target; // silence unused warning if asserts off

    for &(d_to_target, e) in candidates.iter() {
        if r_count == m_use {
            break;
        }
        // e is included if it is closer to target than to any already-chosen r.
        let mut keep = true;
        for &r in out.iter().take(r_count) {
            let d_e_r = dist(vectors, e, r);
            if d_e_r < d_to_target {
                keep = false;
                break;
            }
        }
        if keep {
            out[r_count] = e;
            r_count += 1;
        } else {
            discarded.push((d_to_target, e));
        }
    }

    // If still under-filled, top up from discarded (closest first; already sorted).
    for &(_, e) in discarded.iter() {
        if r_count == m_use {
            break;
        }
        out[r_count] = e;
        r_count += 1;
    }
    for slot in r_count..HNSW_M0 {
        out[slot] = HNSW_SENTINEL;
    }
}

/// Add a back-edge from `target` to `new_neighbor`. Uses SELECT-NEIGHBORS-
/// HEURISTIC over (existing neighbors ∪ {new}) to maintain diverse, well-
/// connected graph rather than purely-closest.
#[allow(clippy::too_many_arguments)]
fn add_back_edge(
    vectors: &[[i8; VECTOR_DIM]],
    target: u32,
    new_neighbor: u32,
    layer: usize,
    m_use: usize,
    layer0: &mut Vec<u32>,
    uppers: &mut [UpperEditable],
) {
    // 1. Read current neighbors
    let mut current: [u32; HNSW_M0] = [HNSW_SENTINEL; HNSW_M0];
    {
        let arr: &[u32] = if layer == 0 {
            let base = target as usize * HNSW_M0;
            &layer0[base..base + HNSW_M0]
        } else {
            uppers[layer - 1].neighbors(target).expect("target in layer")
        };
        for slot in 0..m_use {
            current[slot] = arr[slot];
            if arr[slot] == HNSW_SENTINEL {
                break;
            }
            if arr[slot] == new_neighbor {
                return; // already present
            }
        }
    }

    // 2. Build candidate list = (current ∪ {new})
    let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(m_use + 1);
    for &c in current.iter().take(m_use) {
        if c == HNSW_SENTINEL {
            break;
        }
        candidates.push((dist(vectors, target, c), c));
    }
    candidates.push((dist(vectors, target, new_neighbor), new_neighbor));

    // 3. Apply heuristic
    let mut new_neighbors: [u32; HNSW_M0] = [HNSW_SENTINEL; HNSW_M0];
    select_neighbors_heuristic(vectors, target, &mut candidates, m_use, &mut new_neighbors);

    // 4. Write back
    let arr: &mut [u32] = if layer == 0 {
        let base = target as usize * HNSW_M0;
        &mut layer0[base..base + HNSW_M0]
    } else {
        uppers[layer - 1].neighbors_mut(target)
    };
    for slot in 0..m_use {
        arr[slot] = new_neighbors[slot];
    }
}

pub fn build(vectors: &[[i8; VECTOR_DIM]], seed: u64) -> BuiltGraph {
    let n = vectors.len();
    assert!(n > 0);
    assert!(n < (HNSW_SENTINEL as usize), "too many vectors for u24 ids");

    // Assign levels in parallel (deterministic by seed via per-node RNG).
    let levels: Vec<u8> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Per-node seeded xorshift to get reproducible level
            let mut rng = Rng::new(seed.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            random_level(&mut rng, 7) as u8
        })
        .collect();
    let max_level = *levels.iter().max().unwrap_or(&0) as usize;

    eprintln!("hnsw: N={}, max_level={}, M0={}, M={}", n, max_level, HNSW_M0, HNSW_M);
    for l in 0..=max_level {
        let count = levels.iter().filter(|&&x| (x as usize) >= l).count();
        eprintln!("  layer {}: {} nodes", l, count);
    }

    // Determine insertion order: HIGHER LEVEL FIRST (so the upper-graph backbone
    // is built before low-level fill-in), then random within same level. This
    // avoids the early-insertion correlation that ruins recall when inserting
    // 0..n in order (the seed node ends up with neighbors [1,2,3,4,...]).
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut sort_rng = Rng::new(seed ^ 0xA5A5A5A5_5A5A5A5A);
    // Fisher–Yates shuffle for tie-breaking
    for i in (1..n).rev() {
        let j = (sort_rng.next() as usize) % (i + 1);
        order.swap(i, j);
    }
    // Stable sort by level descending
    order.sort_by(|&a, &b| levels[b as usize].cmp(&levels[a as usize]));

    let mut layer0: Vec<u32> = vec![HNSW_SENTINEL; n * HNSW_M0];
    let mut uppers: Vec<UpperEditable> = (0..max_level).map(|_| UpperEditable::new(n)).collect();
    let mut visited = vec![0u8; n];
    let mut visited_log: Vec<u32> = Vec::with_capacity(8192);

    // Seed entry point: first node in order (highest level due to sort).
    let seed_id = order[0];
    let mut entry: u32 = seed_id;
    let mut max_lvl_so_far: usize = levels[seed_id as usize] as usize;
    for l in 1..=max_lvl_so_far {
        uppers[l - 1].ensure_node(seed_id);
    }

    // Insert remaining nodes in `order`
    let report_every = (n / 20).max(1);
    let t0 = std::time::Instant::now();
    for (i, &q_id) in order.iter().enumerate().skip(1) {
        let q_level = levels[q_id as usize] as usize;
        insert_node(
            q_id,
            q_level,
            vectors,
            &mut entry,
            &mut max_lvl_so_far,
            &mut layer0,
            &mut uppers,
            &mut visited,
            &mut visited_log,
        );
        if i % report_every == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            let eta = (n - i) as f64 / rate;
            eprintln!(
                "  hnsw insert {}/{} ({:.1}%) in {:.1}s, {:.0}/s, ETA {:.0}s",
                i, n, i as f64 * 100.0 / n as f64, elapsed, rate, eta
            );
        }
    }
    eprintln!("hnsw build done in {:.1}s", t0.elapsed().as_secs_f64());

    let upper_layers: Vec<UpperLayer> = uppers
        .into_iter()
        .map(|u| {
            let UpperEditable { mut nodes, mut flat, .. } = u;
            // Sort nodes ascending and reorder neighbors to match.
            // (We need stable lookup by zero-node id at runtime.)
            let mut paired: Vec<(u32, [u32; HNSW_M])> = nodes
                .iter()
                .enumerate()
                .map(|(i, &id)| {
                    let mut arr = [HNSW_SENTINEL; HNSW_M];
                    arr.copy_from_slice(&flat[i * HNSW_M..(i + 1) * HNSW_M]);
                    (id, arr)
                })
                .collect();
            paired.sort_by_key(|&(id, _)| id);
            nodes.clear();
            flat.clear();
            for (id, arr) in paired {
                nodes.push(id);
                flat.extend_from_slice(&arr);
            }
            UpperLayer { nodes, neighbors: flat }
        })
        .collect();

    BuiltGraph {
        num_nodes: n,
        entry_point: entry,
        num_layers: max_lvl_so_far + 1,
        layer0_neighbors: layer0,
        upper_layers,
    }
}
