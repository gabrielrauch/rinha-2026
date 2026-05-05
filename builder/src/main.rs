use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use builder::{blob, kmeans, quantize, sources};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let resources_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "resources".into()));
    let out_path = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "blob.bin".into()));

    eprintln!("loading sources from {}", resources_dir.display());
    let mcc = sources::load_mcc_risk(&resources_dir.join("mcc_risk.json"))?;
    let _norm = sources::load_normalization(&resources_dir.join("normalization.json"))?;

    let refs_path = resources_dir.join("references.json.gz");
    let entries = sources::load_references_gz(&refs_path)?;
    eprintln!("loaded {} reference entries", entries.len());

    let mut vectors = Vec::with_capacity(entries.len());
    let mut is_fraud = Vec::with_capacity(entries.len());
    for e in &entries {
        let (v, f) = quantize::quantize_entry(e);
        vectors.push(v);
        is_fraud.push(f);
    }
    drop(entries);

    eprintln!("running k-means");
    let t = Instant::now();
    let k = (shared::NUM_CENTROIDS as usize).min(vectors.len());
    let (centroids, assignments) = kmeans::kmeans(
        &vectors,
        k,
        15,
        0xDEADBEEF,
    );
    eprintln!("k-means took {:?}", t.elapsed());

    let blob_bytes = blob::build_blob(&blob::BuildInputs {
        centroids: &centroids,
        vectors: &vectors,
        assignments: &assignments,
        is_fraud: &is_fraud,
        mcc_risk: &mcc,
    });

    fs::write(&out_path, &blob_bytes)?;
    eprintln!("wrote {} ({} bytes)", out_path.display(), blob_bytes.len());
    Ok(())
}
