use crate::blob::Blob;
use crate::knn::fraud_count;
use shared::QueryVector;

#[inline]
pub fn fraud_score(blob: &Blob, query: &QueryVector) -> u8 {
    fraud_count(blob, query)
}
