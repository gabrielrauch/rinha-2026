use server::blob::Blob;
use std::path::PathBuf;

#[test]
fn loads_smoke_blob() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).unwrap();
    assert_eq!(blob.header().magic, shared::MAGIC);
    assert_eq!(blob.header().version, shared::VERSION);
    assert!(blob.header().total_vectors > 0);
    assert!(blob.header().k_centroids > 0);
    assert_eq!(
        blob.cluster_offsets().len(),
        blob.header().k_centroids as usize + 1
    );
}
