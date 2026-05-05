use builder::quantize::quantize_entry;
use builder::sources::ReferenceEntry;

#[test]
fn quantizes_in_range() {
    let e = ReferenceEntry {
        vector: vec![0.0, 0.5, 1.0, 0.25, 0.75, -1.0, 0.0, 0.1, 0.2, 1.0, 0.0, 1.0, 0.5, 0.3],
        label: "legit".into(),
    };
    let (v, is_fraud) = quantize_entry(&e);
    assert_eq!(v.len(), 14);
    assert_eq!(v[0], 0);
    assert_eq!(v[2], 127);
    assert_eq!(v[5], -127); // sentinel
    assert_eq!(v[9], 127);
    assert!(!is_fraud);
}
