use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use shared::{BlobHeader, MAGIC, MCC_TABLE_SIZE, VECTOR_DIM, VERSION};
use std::fs::File;
use std::path::Path;

pub struct Blob {
    _mmap: Mmap,
    base: *const u8,
    #[allow(dead_code)]
    len: usize, // kept for future bounds checking
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
        Ok(Self {
            _mmap: mmap,
            base,
            len,
        })
    }

    #[inline]
    pub fn header(&self) -> &BlobHeader {
        unsafe { &*(self.base as *const BlobHeader) }
    }

    #[inline]
    pub fn centroids(&self) -> &[[i8; VECTOR_DIM]] {
        let h = self.header();
        let n = h.num_centroids as usize;
        let p = unsafe { self.base.add(h.centroids_offset as usize) };
        unsafe { std::slice::from_raw_parts(p as *const [i8; VECTOR_DIM], n) }
    }

    #[inline]
    pub fn cluster_offsets(&self) -> &[u32] {
        let h = self.header();
        let n = h.num_centroids as usize + 1;
        let p = unsafe { self.base.add(h.cluster_offsets_offset as usize) };
        unsafe { std::slice::from_raw_parts(p as *const u32, n) }
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
        let n = (h.total_vectors as usize + 7) / 8;
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
}
