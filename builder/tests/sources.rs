use builder::sources::{load_mcc_risk, load_normalization, load_references_json};
use std::path::Path;

#[test]
fn loads_normalization_constants() {
    let n = load_normalization(Path::new("../resources/normalization.json")).unwrap();
    assert_eq!(n.max_amount, 10_000.0);
    assert_eq!(n.max_installments, 12);
}

#[test]
fn loads_mcc_risk_table() {
    let table = load_mcc_risk(Path::new("../resources/mcc_risk.json")).unwrap();
    // 5411 = grocery → low risk
    assert!(table.get(&5411).copied().unwrap_or(0.5) < 0.3);
    // 7995 = gambling → high risk
    assert!(table.get(&7995).copied().unwrap_or(0.5) > 0.8);
}

#[test]
fn loads_example_references() {
    let entries = load_references_json(Path::new("../resources/example-references.json")).unwrap();
    assert!(!entries.is_empty());
    for e in &entries {
        assert_eq!(e.vector.len(), 14);
        assert!(e.label == "fraud" || e.label == "legit");
    }
}
