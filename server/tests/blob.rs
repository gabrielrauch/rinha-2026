use server::blob::Blob;
use std::path::PathBuf;

#[test]
#[ignore = "needs a real built blob at ../tmp/blob.bin"]
fn loads_smoke_blob() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
    let blob = Blob::open(&path).unwrap();
    assert!(blob.part_count() > 0);
    assert!(blob.node_count() > 0);
    assert!(blob.block_count() > 0);
}
