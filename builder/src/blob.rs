use crate::hnsw::BuiltGraph;
use shared::*;
use std::collections::HashMap;

pub struct BuildInputs<'a> {
    pub vectors: &'a [[i8; VECTOR_DIM]],
    pub is_fraud: &'a [bool],
    pub graph: &'a BuiltGraph,
    pub mcc_risk: &'a HashMap<u32, f32>,
}

/// Pack `value` (assumed `< 2^24`) as 3 little-endian bytes appended to `out`.
#[inline]
fn push_u24(out: &mut Vec<u8>, value: u32) {
    debug_assert!(value <= 0x00FF_FFFF);
    out.push((value & 0xFF) as u8);
    out.push(((value >> 8) & 0xFF) as u8);
    out.push(((value >> 16) & 0xFF) as u8);
}

pub fn build_blob(inp: &BuildInputs) -> Vec<u8> {
    let n = inp.vectors.len();
    let g = inp.graph;
    assert_eq!(inp.is_fraud.len(), n);
    assert_eq!(g.num_nodes, n);

    // Precompute sizes / offsets.
    let header_size = std::mem::size_of::<BlobHeader>() as u32;
    let vectors_size = (n * VECTOR_DIM) as u32;
    let labels_size = ((n + 7) / 8) as u32;
    let mcc_size = MCC_TABLE_SIZE as u32;
    let layer0_size = (n * HNSW_M0 * 3) as u32; // u24 packed

    let vectors_offset = header_size;
    let labels_offset = vectors_offset + vectors_size;
    let mcc_offset = labels_offset + labels_size;
    let layer0_offset = mcc_offset + mcc_size;

    let mut layer_node_count = [0u32; HNSW_MAX_LAYERS];
    let mut layer_nodes_offset = [0u32; HNSW_MAX_LAYERS];
    let mut layer_neighbors_offset = [0u32; HNSW_MAX_LAYERS];

    layer_node_count[0] = n as u32;
    layer_nodes_offset[0] = 0; // implicit (dense)
    layer_neighbors_offset[0] = layer0_offset;

    let mut cursor = layer0_offset + layer0_size;
    for (i, layer) in g.upper_layers.iter().enumerate() {
        let lid = i + 1;
        assert!(lid < HNSW_MAX_LAYERS, "more layers than HNSW_MAX_LAYERS");
        let cnt = layer.nodes.len() as u32;
        layer_node_count[lid] = cnt;
        layer_nodes_offset[lid] = cursor;
        cursor += cnt * 4; // u32 zero-node ids
        layer_neighbors_offset[lid] = cursor;
        cursor += cnt * HNSW_M as u32 * 3; // u24 packed
    }
    let blob_size = cursor;

    let header = BlobHeader {
        magic: MAGIC,
        version: VERSION,
        total_vectors: n as u32,
        vectors_offset,
        labels_offset,
        mcc_table_offset: mcc_offset,
        hnsw_entry_point: g.entry_point,
        hnsw_num_layers: g.num_layers as u8,
        hnsw_m0: HNSW_M0 as u8,
        hnsw_m: HNSW_M as u8,
        _hnsw_pad: 0,
        layer_node_count,
        layer_nodes_offset,
        layer_neighbors_offset,
        blob_size,
        _padding: [0; 120],
    };

    let mut out = Vec::with_capacity(blob_size as usize);

    // Header
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const BlobHeader as *const u8,
            std::mem::size_of::<BlobHeader>(),
        )
    };
    out.extend_from_slice(header_bytes);

    // Vectors
    for v in inp.vectors {
        let v_bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, VECTOR_DIM) };
        out.extend_from_slice(v_bytes);
    }

    // Labels (1 bit per node, packed)
    let mut label_bits = vec![0u8; (n + 7) / 8];
    for (i, &f) in inp.is_fraud.iter().enumerate() {
        if f {
            label_bits[i / 8] |= 1 << (i % 8);
        }
    }
    out.extend_from_slice(&label_bits);

    // MCC table (1024 i8)
    let mut mcc_table = [quantize_unit(0.5); MCC_TABLE_SIZE];
    for (&mcc, &risk) in inp.mcc_risk {
        mcc_table[(mcc as usize) % MCC_TABLE_SIZE] = quantize_unit(clamp01(risk));
    }
    let mcc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(mcc_table.as_ptr() as *const u8, MCC_TABLE_SIZE) };
    out.extend_from_slice(mcc_bytes);

    // Layer 0 neighbors (u24 packed; full N*M0 slots, sentinel for empty)
    debug_assert_eq!(g.layer0_neighbors.len(), n * HNSW_M0);
    for &v in &g.layer0_neighbors {
        push_u24(&mut out, v);
    }

    // Higher layers: nodes (u32 each) + neighbors (u24 packed M slots each)
    for layer in &g.upper_layers {
        for &id in &layer.nodes {
            out.extend_from_slice(&id.to_le_bytes());
        }
        debug_assert_eq!(layer.neighbors.len(), layer.nodes.len() * HNSW_M);
        for &v in &layer.neighbors {
            push_u24(&mut out, v);
        }
    }

    debug_assert_eq!(out.len(), blob_size as usize, "blob size mismatch");
    out
}
