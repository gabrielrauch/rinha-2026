use crate::blob::Blob;
use crate::ivf::knn5_fraud_count;
use shared::VECTOR_DIM;

pub fn fraud_score(blob: &Blob, query: &[f32; VECTOR_DIM]) -> u8 {
    knn5_fraud_count(blob, query)
}
