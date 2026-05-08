#![allow(clippy::needless_range_loop)]

use crate::blob::Blob;
use crate::hnsw::{search_top_k, TOP_K};
use shared::VECTOR_DIM;

/// Run a HNSW k-NN query over the blob and return how many of the top-5
/// neighbors are labeled fraud.
pub fn fraud_score(blob: &Blob, query: &[i8; VECTOR_DIM]) -> u8 {
    let top = search_top_k(blob, query);
    let mut count: u8 = 0;
    for i in 0..TOP_K {
        let (_, idx) = top[i];
        if idx == u32::MAX {
            continue;
        }
        if blob.is_fraud(idx) {
            count += 1;
        }
    }
    count
}
