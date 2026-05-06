use server::blob::Blob;
use std::path::PathBuf;

#[test]
fn loads_smoke_blob() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).unwrap();
    assert_eq!(blob.header().magic, shared::MAGIC);
    assert!(!blob.centroids().is_empty());
    assert!(!blob.vectors().is_empty());
    assert_eq!(blob.cluster_offsets().len(), blob.centroids().len() + 1);
    let total_from_offsets = *blob.cluster_offsets().last().unwrap() as usize;
    assert_eq!(total_from_offsets, blob.header().total_vectors as usize);
}
