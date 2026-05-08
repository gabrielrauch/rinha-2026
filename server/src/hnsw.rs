#![allow(clippy::needless_range_loop)]

//! HNSW search over the compact blob format.
//!
//! Memory model:
//! - The graph itself lives in mmap'd file pages (read-only)
//! - Per-query state (visited bitmap, candidate/found heaps) lives in a
//!   thread-local buffer that is reused across requests — no heap alloc
//!   per `fraud_score` call.
//!
//! Algorithm (Malkov & Yashunin, 2018):
//! 1. From `entry_point` at `max_layer`, greedy-descend to layer 1
//! 2. At layer 0, beam search with `ef_search` candidates
//! 3. Return the top-K closest

use crate::blob::Blob;
use crate::distance::l2_squared;
use shared::{HNSW_M, HNSW_M0, HNSW_SENTINEL, VECTOR_DIM};
use std::cell::RefCell;

/// Beam width at layer 0. Higher = more recall, more compute.
pub const EF_SEARCH: usize = 64;
/// We always pull the top-5 frauds (rinha k-NN with k=5).
pub const TOP_K: usize = 5;

/// Per-thread reusable state. Single monoio runtime per process means a single
/// thread; tasks cooperatively yield only at await points, and `search` has no
/// awaits, so no two queries run concurrently here.
struct SearchState {
    /// Bitmap, 1 bit per zero-node, sized for up to 4 M nodes (we have ~3 M).
    visited: Vec<u64>,
    /// IDs that we've set bits for — used to zero them out incrementally on reset.
    visited_ids: Vec<u32>,
    /// Min-heap of (distance, node) pairs that are candidates to expand.
    /// Stored as a Vec we manage as a binary heap manually for speed.
    candidates: Vec<(u32, u32)>,
    /// Max-heap (by distance) of (distance, node) pairs found so far, capped at ef.
    found: Vec<(u32, u32)>,
}

impl SearchState {
    fn new(max_nodes: usize) -> Self {
        let n_words = (max_nodes + 63) / 64;
        Self {
            visited: vec![0u64; n_words],
            visited_ids: Vec::with_capacity(8192),
            candidates: Vec::with_capacity(EF_SEARCH * 2),
            found: Vec::with_capacity(EF_SEARCH * 2),
        }
    }

    /// Mark `id` as visited. Returns true if newly visited.
    #[inline]
    fn mark_visited(&mut self, id: u32) -> bool {
        let i = id as usize;
        let word = i / 64;
        let bit = 1u64 << (i % 64);
        let was = self.visited[word] & bit != 0;
        if !was {
            self.visited[word] |= bit;
            self.visited_ids.push(id);
        }
        !was
    }

    /// Zero only the bits we set this query, then clear the working vecs.
    #[inline]
    fn reset(&mut self) {
        for &id in &self.visited_ids {
            let i = id as usize;
            let word = i / 64;
            let bit = 1u64 << (i % 64);
            self.visited[word] &= !bit;
        }
        self.visited_ids.clear();
        self.candidates.clear();
        self.found.clear();
    }
}

thread_local! {
    static STATE: RefCell<Option<SearchState>> = const { RefCell::new(None) };
}

/// Public entry. Returns the top-K (distance, zero_node_id) pairs sorted ascending
/// by distance. Pads with `(u32::MAX, u32::MAX)` if fewer than K were found.
pub fn search_top_k(blob: &Blob, query: &[i8; VECTOR_DIM]) -> [(u32, u32); TOP_K] {
    STATE.with(|s| {
        let mut borrow = s.borrow_mut();
        let state = borrow.get_or_insert_with(|| {
            let n = blob.header().total_vectors as usize;
            SearchState::new(n)
        });
        state.reset();

        let mut result = [(u32::MAX, u32::MAX); TOP_K];
        do_search(blob, query, state, &mut result);
        result
    })
}

#[inline]
fn distance_to(blob: &Blob, query: &[i8; VECTOR_DIM], node: u32) -> u32 {
    let v = &blob.vectors()[node as usize];
    l2_squared(query, v) as u32
}

fn do_search(
    blob: &Blob,
    query: &[i8; VECTOR_DIM],
    state: &mut SearchState,
    out: &mut [(u32, u32); TOP_K],
) {
    let num_layers = blob.hnsw_num_layers();
    if num_layers == 0 || blob.header().total_vectors == 0 {
        return;
    }

    let mut ep = blob.hnsw_entry_point();
    let mut ep_dist = distance_to(blob, query, ep);

    // 1. Greedy descend from top layer down to layer 1.
    for layer in (1..num_layers).rev() {
        loop {
            let mut improved = false;
            // Find ep's slot in this layer's nodes table → walk its M neighbors.
            // For non-zero layer we need the local node index for `ep`.
            // We linear-scan layer_nodes; but layer_nodes is small (375k for L1, much smaller above).
            // For greedy descent this is hot — we should cache positions, but in practice
            // upper layers are only entered through entry_point's path, and within a layer
            // we follow neighbors which are zero-node IDs that we then need to look up locally.
            //
            // Practical: descent happens once per query, and only on layers >0 where node
            // counts are small. Layer 1 has ~N/M = 375k nodes; iterating 8 neighbors per
            // step with O(1) lookup is what we want.
            //
            // To avoid the linear scan we keep a mapping zero_id -> layer_local_index per
            // layer in the blob. For now we do a linear scan and revisit if profiling needs it.
            let local = layer_local_index(blob, layer, ep);
            if local == u32::MAX {
                break;
            }
            // Walk M neighbors of `ep` at this layer
            let base_slot = local as usize * HNSW_M;
            for slot in 0..HNSW_M {
                let n = blob.hnsw_neighbor(layer, base_slot + slot);
                if n == HNSW_SENTINEL {
                    break;
                }
                let d = distance_to(blob, query, n);
                if d < ep_dist {
                    ep = n;
                    ep_dist = d;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }

    // 2. Beam search at layer 0 with ef = EF_SEARCH.
    state.mark_visited(ep);
    state.candidates.push((ep_dist, ep));
    state.found.push((ep_dist, ep));

    while let Some((d_c, c)) = pop_min(&mut state.candidates) {
        let worst = peek_max(&state.found).map(|x| x.0).unwrap_or(u32::MAX);
        if d_c > worst && state.found.len() >= EF_SEARCH {
            break;
        }
        // Walk the M0 neighbors of c at layer 0.
        let base_slot = c as usize * HNSW_M0;
        for slot in 0..HNSW_M0 {
            let n = blob.hnsw_neighbor(0, base_slot + slot);
            if n == HNSW_SENTINEL {
                break;
            }
            if !state.mark_visited(n) {
                continue;
            }
            let d = distance_to(blob, query, n);
            let worst = peek_max(&state.found).map(|x| x.0).unwrap_or(u32::MAX);
            if d < worst || state.found.len() < EF_SEARCH {
                push_min(&mut state.candidates, (d, n));
                push_max(&mut state.found, (d, n));
                if state.found.len() > EF_SEARCH {
                    pop_max(&mut state.found);
                }
            }
        }
    }

    // 3. Extract top-K from `found` (which is a max-heap by distance).
    // Sort the heap ascending and take the first K.
    state.found.sort_unstable_by_key(|&(d, _)| d);
    for (i, &pair) in state.found.iter().take(TOP_K).enumerate() {
        out[i] = pair;
    }
}

/// Given a layer (>0) and a zero-node id, find the layer-local index. Returns
/// u32::MAX if not in this layer.
///
/// The builder sorts each layer's node list ascending by zero-id, so we can
/// binary-search instead of doing an O(N) scan.
#[inline]
fn layer_local_index(blob: &Blob, layer: usize, zero_id: u32) -> u32 {
    debug_assert!(layer > 0);
    let nodes = blob.hnsw_layer_nodes(layer);
    match nodes.binary_search(&zero_id) {
        Ok(i) => i as u32,
        Err(_) => u32::MAX,
    }
}

// ---- tiny binary-heap helpers (ad-hoc to avoid std BinaryHeap allocations) ----
// We keep `candidates` as a min-heap (smallest distance at index 0) and
// `found` as a max-heap (largest at index 0). Both are flat Vecs.

#[inline]
fn pop_min(heap: &mut Vec<(u32, u32)>) -> Option<(u32, u32)> {
    if heap.is_empty() {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if !heap.is_empty() {
        heap[0] = last;
        sift_down_min(heap, 0);
    }
    Some(top)
}

#[inline]
fn push_min(heap: &mut Vec<(u32, u32)>, x: (u32, u32)) {
    heap.push(x);
    let last = heap.len() - 1;
    sift_up_min(heap, last);
}

#[inline]
fn sift_up_min(heap: &mut [(u32, u32)], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i].0 < heap[parent].0 {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

#[inline]
fn sift_down_min(heap: &mut [(u32, u32)], mut i: usize) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut best = i;
        if left < n && heap[left].0 < heap[best].0 {
            best = left;
        }
        if right < n && heap[right].0 < heap[best].0 {
            best = right;
        }
        if best == i {
            break;
        }
        heap.swap(i, best);
        i = best;
    }
}

#[inline]
fn pop_max(heap: &mut Vec<(u32, u32)>) -> Option<(u32, u32)> {
    if heap.is_empty() {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if !heap.is_empty() {
        heap[0] = last;
        sift_down_max(heap, 0);
    }
    Some(top)
}

#[inline]
fn peek_max(heap: &[(u32, u32)]) -> Option<&(u32, u32)> {
    heap.first()
}

#[inline]
fn push_max(heap: &mut Vec<(u32, u32)>, x: (u32, u32)) {
    heap.push(x);
    let last = heap.len() - 1;
    sift_up_max(heap, last);
}

#[inline]
fn sift_up_max(heap: &mut [(u32, u32)], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i].0 > heap[parent].0 {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

#[inline]
fn sift_down_max(heap: &mut [(u32, u32)], mut i: usize) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut best = i;
        if left < n && heap[left].0 > heap[best].0 {
            best = left;
        }
        if right < n && heap[right].0 > heap[best].0 {
            best = right;
        }
        if best == i {
            break;
        }
        heap.swap(i, best);
        i = best;
    }
}
