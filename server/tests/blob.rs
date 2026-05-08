use server::blob::Blob;
use std::path::PathBuf;

#[test]
fn loads_smoke_blob() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).unwrap();
    assert_eq!(blob.header().magic, shared::MAGIC);
    assert_eq!(blob.header().version, shared::VERSION);
    assert!(!blob.vectors().is_empty());
    assert!(blob.hnsw_num_layers() >= 1);
    assert!(blob.hnsw_entry_point() < blob.header().total_vectors);
}
