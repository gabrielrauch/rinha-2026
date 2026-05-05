use shared::*;
use std::collections::HashMap;

pub struct BuildInputs<'a> {
    pub centroids: &'a [[i8; VECTOR_DIM]],
    pub vectors: &'a [[i8; VECTOR_DIM]],
    pub assignments: &'a [u32],
    pub is_fraud: &'a [bool],
    pub mcc_risk: &'a HashMap<u32, f32>,
}

#[allow(clippy::missing_safety_doc)]
pub fn build_blob(inp: &BuildInputs) -> Vec<u8> {
    let n = inp.vectors.len();
    let k = inp.centroids.len();
    assert_eq!(inp.assignments.len(), n);
    assert_eq!(inp.is_fraud.len(), n);

    // group: cluster -> list of original indices
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, &a) in inp.assignments.iter().enumerate() {
        buckets[a as usize].push(i as u32);
    }

    // cluster_offsets: prefix sum of bucket sizes
    let mut cluster_offsets: Vec<u32> = Vec::with_capacity(k + 1);
    cluster_offsets.push(0);
    let mut acc: u32 = 0;
    for b in &buckets {
        acc += b.len() as u32;
        cluster_offsets.push(acc);
    }

    // ordered vectors and labels
    let mut ordered_vectors: Vec<i8> = Vec::with_capacity(n * VECTOR_DIM);
    let mut label_bits: Vec<u8> = vec![0u8; (n + 7) / 8];
    let mut out_idx: u32 = 0;
    for bucket in &buckets {
        for &orig in bucket {
            ordered_vectors.extend_from_slice(&inp.vectors[orig as usize]);
            if inp.is_fraud[orig as usize] {
                let byte = (out_idx as usize) / 8;
                let bit = (out_idx as usize) % 8;
                label_bits[byte] |= 1 << bit;
            }
            out_idx += 1;
        }
    }

    // mcc table: 1024 i8s, default ~0.5 → quantized; lookup by mcc % 1024
    let mut mcc_table = [quantize_unit(0.5); MCC_TABLE_SIZE];
    for (&mcc, &risk) in inp.mcc_risk {
        mcc_table[(mcc as usize) % MCC_TABLE_SIZE] = quantize_unit(clamp01(risk));
    }

    // layout
    let header_size = std::mem::size_of::<BlobHeader>() as u32;
    let centroids_size = (k * VECTOR_DIM) as u32;
    let cluster_offsets_size = ((k + 1) * 4) as u32;
    let vectors_size = (n * VECTOR_DIM) as u32;
    let labels_size = label_bits.len() as u32;
    let mcc_size = MCC_TABLE_SIZE as u32;

    let centroids_offset = header_size;
    let cluster_offsets_offset = centroids_offset + centroids_size;
    let vectors_offset = cluster_offsets_offset + cluster_offsets_size;
    let labels_offset = vectors_offset + vectors_size;
    let mcc_offset = labels_offset + labels_size;
    let blob_size = mcc_offset + mcc_size;

    let header = BlobHeader {
        magic: MAGIC,
        version: VERSION,
        num_centroids: k as u32,
        total_vectors: n as u32,
        centroids_offset,
        cluster_offsets_offset,
        vectors_offset,
        labels_offset,
        mcc_table_offset: mcc_offset,
        blob_size,
        _padding: [0; 20],
    };

    let mut out = Vec::with_capacity(blob_size as usize);
    // SAFETY: BlobHeader is repr(C); we transmute to bytes for serialization
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const BlobHeader as *const u8,
            std::mem::size_of::<BlobHeader>(),
        )
    };
    out.extend_from_slice(header_bytes);
    for c in inp.centroids {
        // SAFETY: [i8; 14] and [u8; 14] have the same layout
        let c_bytes = unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const u8, VECTOR_DIM)
        };
        out.extend_from_slice(c_bytes);
    }
    for &o in &cluster_offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    // SAFETY: Vec<i8> and Vec<u8> have the same layout
    let ordered_bytes = unsafe {
        std::slice::from_raw_parts(ordered_vectors.as_ptr() as *const u8, ordered_vectors.len())
    };
    out.extend_from_slice(ordered_bytes);
    out.extend_from_slice(&label_bits);
    // SAFETY: mcc_table is [i8; 1024], we transmute to bytes for serialization
    let mcc_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(mcc_table.as_ptr() as *const u8, MCC_TABLE_SIZE)
    };
    out.extend_from_slice(mcc_bytes);
    debug_assert_eq!(out.len(), blob_size as usize);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(n: usize, k: usize) -> (Vec<[i8; VECTOR_DIM]>, Vec<[i8; VECTOR_DIM]>, Vec<u32>, Vec<bool>) {
        let centroids = vec![[0i8; VECTOR_DIM]; k];
        let vectors = vec![[1i8; VECTOR_DIM]; n];
        let assignments = (0..n as u32).map(|i| i % k as u32).collect();
        let is_fraud = vec![false; n];
        (centroids, vectors, assignments, is_fraud)
    }

    #[test]
    fn blob_starts_with_magic() {
        let (c, v, a, f) = dummy(100, 4);
        let mcc = HashMap::new();
        let blob = build_blob(&BuildInputs {
            centroids: &c, vectors: &v, assignments: &a, is_fraud: &f, mcc_risk: &mcc,
        });
        assert_eq!(&blob[..8], &MAGIC);
    }

    #[test]
    fn blob_header_fields_correct() {
        let (c, v, a, f) = dummy(100, 4);
        let mcc = HashMap::new();
        let blob = build_blob(&BuildInputs {
            centroids: &c, vectors: &v, assignments: &a, is_fraud: &f, mcc_risk: &mcc,
        });
        let header: &BlobHeader = unsafe { &*(blob.as_ptr() as *const BlobHeader) };
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
        assert_eq!(header.num_centroids, 4);
        assert_eq!(header.total_vectors, 100);
        assert_eq!(header.blob_size, blob.len() as u32);
    }
}
