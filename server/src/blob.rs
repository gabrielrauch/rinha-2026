use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use shared::{BlobHeader, BLOCK_VECS, MAGIC, MCC_TABLE_SIZE, VECTOR_DIM, VERSION};
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
        let blob = Self {
            _mmap: mmap,
            base,
            len,
        };
        blob.prefetch();
        Ok(blob)
    }

    /// Walk every page once to populate the kernel's page cache.
    fn prefetch(&self) {
        const PAGE: usize = 4096;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < self.len {
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

    /// Pointer to centroids region (f32 SoA dim-major, K * VECTOR_DIM floats).
    #[inline]
    pub fn centroids_ptr(&self) -> *const f32 {
        let h = self.header();
        unsafe { self.base.add(h.centroids_offset as usize) as *const f32 }
    }

    /// `cluster_offsets[i]..cluster_offsets[i+1]` is the half-open range of
    /// block indices belonging to cluster i.
    #[inline]
    pub fn cluster_offsets(&self) -> &[u32] {
        let h = self.header();
        let n = h.k_centroids as usize + 1;
        let p = unsafe { self.base.add(h.cluster_offsets_offset as usize) };
        unsafe { std::slice::from_raw_parts(p as *const u32, n) }
    }

    /// Pointer to the start of the blocks region (i16 SoA).
    #[inline]
    pub fn blocks_ptr(&self) -> *const i16 {
        let h = self.header();
        unsafe { self.base.add(h.blocks_offset as usize) as *const i16 }
    }

    #[inline]
    pub fn labels_ptr(&self) -> *const u8 {
        let h = self.header();
        unsafe { self.base.add(h.labels_offset as usize) }
    }

    /// Whether the vector at slot `slot` of block `block_idx` is labeled fraud.
    #[inline]
    pub fn is_fraud_slot(&self, block_idx: u32, slot: u32) -> bool {
        let bit = block_idx as usize * BLOCK_VECS + slot as usize;
        let byte = unsafe { *self.labels_ptr().add(bit / 8) };
        (byte >> (bit % 8)) & 1 == 1
    }

    #[inline]
    pub fn mcc_risk(&self, mcc: u32) -> i8 {
        let h = self.header();
        let p = unsafe { self.base.add(h.mcc_table_offset as usize) };
        let table = unsafe { std::slice::from_raw_parts(p as *const i8, MCC_TABLE_SIZE) };
        table[(mcc as usize) % MCC_TABLE_SIZE]
    }

    #[inline]
    pub fn vector_dim(&self) -> usize {
        VECTOR_DIM
    }
}
