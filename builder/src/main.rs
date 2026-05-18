use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use builder::{blob, quantize, sources};
use shared::{quantize_value, MCC_TABLE_SIZE};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let resources_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "resources".into()));
    let out_path = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "blob.bin".into()));
    let leaf_size: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(blob::DEFAULT_LEAF_SIZE);

    eprintln!("loading mcc risk from {}", resources_dir.display());
    let mcc_risk = sources::load_mcc_risk(&resources_dir.join("mcc_risk.json"))?;
    let mut mcc_table = [0i16; MCC_TABLE_SIZE];
    for (&mcc, &risk) in mcc_risk.iter() {
        let idx = (mcc as usize) % MCC_TABLE_SIZE;
        mcc_table[idx] = quantize_value(risk as f64);
    }

    eprintln!("loading references from {}", resources_dir.display());
    let refs_path = resources_dir.join("references.json.gz");
    let entries = sources::load_references_gz(&refs_path)?;
    eprintln!("loaded {} reference entries", entries.len());

    let t = Instant::now();
    let mut vectors = Vec::with_capacity(entries.len());
    let mut labels = Vec::with_capacity(entries.len());
    for e in &entries {
        let (v, f) = quantize::entry_to_vector(e);
        vectors.push(v);
        labels.push(f);
    }
    drop(entries);
    eprintln!("quantized {} vectors in {:?}", vectors.len(), t.elapsed());

    let t = Instant::now();
    let blob_bytes = blob::build_blob_with_leaf(
        &blob::BuildInputs {
            vectors: &vectors,
            labels: &labels,
            mcc_table: &mcc_table,
        },
        leaf_size,
    );
    eprintln!(
        "built KD-tree blob ({} bytes, leaf={}) in {:?}",
        blob_bytes.len(),
        leaf_size,
        t.elapsed()
    );

    fs::write(&out_path, &blob_bytes)?;
    eprintln!(
        "wrote {} ({:.1} MB)",
        out_path.display(),
        blob_bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}
