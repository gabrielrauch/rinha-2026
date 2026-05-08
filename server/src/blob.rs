use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use shared::{BlobHeader, HNSW_M, HNSW_M0, MAGIC, MCC_TABLE_SIZE, VECTOR_DIM, VERSION};
use std::fs::File;
use std::path::Path;

pub struct Blob {
    _mmap: Mmap,
    base: *const u8,
    #[allow(dead_code)]
    len: usize,
}

unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}

impl Blob {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f) }?;
        let base = mmap.as_ptr();
        let len = mmap.len();

        if len < std::mem::size_of::<BlobHeader>() {
            return Err(anyhow!("blob too small"));
        }
        let header: &BlobHeader = unsafe { &*(base as *const BlobHeader) };
        if header.magic != MAGIC {
            return Err(anyhow!("bad magic"));
        }
        if header.version != VERSION {
            return Err(anyhow!("unsupported blob version {}", header.version));
        }
        if header.blob_size as usize != len {
            return Err(anyhow!(
                "blob size mismatch (header {} vs file {})",
                header.blob_size,
                len
            ));
        }
        if (header.hnsw_m0 as usize) != HNSW_M0 || (header.hnsw_m as usize) != HNSW_M {
            return Err(anyhow!(
                "hnsw M mismatch: blob M0={} M={}, code expects M0={} M={}",
                header.hnsw_m0,
                header.hnsw_m,
                HNSW_M0,
                HNSW_M
            ));
        }
        let blob = Self {
            _mmap: mmap,
            base,
            len,
        };
        blob.prefetch();
        Ok(blob)
    }

    /// Force every page of the blob into RAM so the hot path never pays a page-fault tail.
    fn prefetch(&self) {
        const PAGE: usize = 4096;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < self.len {
            // SAFETY: i < self.len, base..base+len is a valid mmap region.
            acc ^= unsafe { std::ptr::read_volatile(self.base.add(i)) };
            i += PAGE;
        }
        if acc == 0xFE {
            eprintln!("prefetch sentinel hit");
        }
        eprintln!(
            "prefetched {} pages ({} bytes)",
            self.len.div_ceil(PAGE),
            self.len
        );
    }

    #[inline]
    pub fn header(&self) -> &BlobHeader {
        unsafe { &*(self.base as *const BlobHeader) }
    }

    #[inline]
    pub fn vectors(&self) -> &[[i8; VECTOR_DIM]] {
        let h = self.header();
        let n = h.total_vectors as usize;
        let p = unsafe { self.base.add(h.vectors_offset as usize) };
        unsafe { std::slice::from_raw_parts(p as *const [i8; VECTOR_DIM], n) }
    }

    #[inline]
    pub fn label_bits(&self) -> &[u8] {
        let h = self.header();
        let n = (h.total_vectors as usize).div_ceil(8);
        let p = unsafe { self.base.add(h.labels_offset as usize) };
        unsafe { std::slice::from_raw_parts(p, n) }
    }

    #[inline]
    pub fn is_fraud(&self, idx: u32) -> bool {
        let bits = self.label_bits();
        let byte = bits[(idx as usize) / 8];
        (byte >> ((idx as usize) % 8)) & 1 == 1
    }

    #[inline]
    pub fn mcc_risk(&self, mcc: u32) -> i8 {
        let h = self.header();
        let p = unsafe { self.base.add(h.mcc_table_offset as usize) };
        let table = unsafe { std::slice::from_raw_parts(p as *const i8, MCC_TABLE_SIZE) };
        table[(mcc as usize) % MCC_TABLE_SIZE]
    }

    /// Number of HNSW layers (including layer 0).
    #[inline]
    pub fn hnsw_num_layers(&self) -> usize {
        self.header().hnsw_num_layers as usize
    }

    /// Entry point for HNSW search (top layer's start node, expressed as a zero-layer id).
    #[inline]
    pub fn hnsw_entry_point(&self) -> u32 {
        self.header().hnsw_entry_point
    }

    /// Number of nodes present at a given layer.
    #[inline]
    pub fn hnsw_layer_node_count(&self, layer: usize) -> usize {
        self.header().layer_node_count[layer] as usize
    }

    /// For non-zero layers, the slice of zero-node IDs that participate in this layer.
    /// (Layer 0 is dense — all nodes, indices implicit.)
    #[inline]
    pub fn hnsw_layer_nodes(&self, layer: usize) -> &[u32] {
        debug_assert!(layer > 0);
        let h = self.header();
        let n = h.layer_node_count[layer] as usize;
        let p = unsafe { self.base.add(h.layer_nodes_offset[layer] as usize) };
        unsafe { std::slice::from_raw_parts(p as *const u32, n) }
    }

    /// Read a u24-packed neighbor at slot `slot_index` of layer `layer`.
    /// Returns the zero-node id, or `0xFFFFFF` if the slot is empty.
    #[inline]
    pub fn hnsw_neighbor(&self, layer: usize, slot_index: usize) -> u32 {
        let h = self.header();
        let p = unsafe { self.base.add(h.layer_neighbors_offset[layer] as usize) };
        let off = slot_index * 3;
        // SAFETY: assumes the caller computed slot_index < (node_count_at_layer * M_at_layer)
        unsafe {
            let b0 = *p.add(off) as u32;
            let b1 = *p.add(off + 1) as u32;
            let b2 = *p.add(off + 2) as u32;
            b0 | (b1 << 8) | (b2 << 16)
        }
    }

    /// Returns a raw pointer to the start of layer L's neighbor table (u24 packed).
    /// Useful for hot paths that want to avoid repeated header lookups.
    #[inline]
    pub fn hnsw_layer_neighbors_ptr(&self, layer: usize) -> *const u8 {
        let h = self.header();
        unsafe { self.base.add(h.layer_neighbors_offset[layer] as usize) }
    }
}
