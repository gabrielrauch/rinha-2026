//! Types shared between the offline builder and the runtime server.
//!
//! Blob layout v3 (IVF SoA, little-endian):
//!
//! ```text
//! [BlobHeader: 256 bytes]
//! [centroids: NUM_CENTROIDS * VECTOR_DIM * 4 bytes (f32, dim-major SoA: dim 0 of all 8192, dim 1 of all 8192, ...)]
//! [cluster_offsets: (NUM_CENTROIDS + 1) * 4 bytes (u32) — block offsets per cluster]
//! [blocks: total_blocks * BLOCK_BYTES bytes — each block packs 8 vectors as i16 SoA: dim 0 slot 0..7, dim 1 slot 0..7, ...]
//! [labels: padded_n bits packed in bytes, 1 bit per slot (block_id*8 + slot), 1 = fraud]
//! [mcc_risk_table: 1024 bytes (i8 × 1024)]
//! ```
//!
//! Vectors are stored quantized as i16 with scale 8192 (~13 fractional bits) so
//! each block is `BLOCK_VECS * VECTOR_DIM * 2` bytes. SoA layout lets a single
//! `_mm_loadu_si128` pull all 8 lanes of one dimension; we then upcast to f32
//! for FMA accumulation. Last block of each cluster pads with i16::MAX so the
//! distance compute is enormous and those slots never enter the top-K.

pub const MAGIC: [u8; 8] = *b"RINHA026";
pub const VERSION: u32 = 3;
pub const VECTOR_DIM: usize = 14;
pub const MCC_TABLE_SIZE: usize = 1024;

/// IVF parameters
pub const NUM_CENTROIDS: u32 = 16384;
pub const BLOCK_VECS: usize = 8;
pub const BLOCK_BYTES: usize = BLOCK_VECS * VECTOR_DIM * 2; // 8 vecs * 14 dims * i16 = 224 bytes
/// Quantization scale. Vectors are clamped to [-1, 1] then multiplied by SCALE
/// before rounding to i16. SCALE=8192 leaves headroom (i16 max = 32767, max possible
/// product after squaring = 67M; with FMA over 14 dims worst-case = 938M which fits in f32).
pub const QUANT_SCALE: f32 = 8192.0;
/// Inverse of `QUANT_SCALE`, applied at search time.
pub const QUANT_INV_SCALE: f32 = 1.0 / 8192.0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlobHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub total_vectors: u32,
    pub padded_n: u32, // total_blocks * BLOCK_VECS
    pub total_blocks: u32,
    pub k_centroids: u32, // NUM_CENTROIDS

    pub centroids_offset: u32,       // f32 SoA, dim-major
    pub cluster_offsets_offset: u32, // u32, length k_centroids + 1, block indices
    pub blocks_offset: u32,          // i16 SoA blocks
    pub labels_offset: u32,          // bits, length ceil(padded_n / 8)
    pub mcc_table_offset: u32,       // i8 * 1024

    pub blob_size: u32,
    pub _padding: [u8; 204],
}

const _: () = {
    assert!(std::mem::size_of::<BlobHeader>() == 256);
};

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn header_size_known() {
        // sanity: must be a multiple of 8 (alignment-friendly) and reasonable
        let sz = std::mem::size_of::<BlobHeader>();
        assert!(sz % 8 == 0, "header size {} not 8-aligned", sz);
        assert!(sz >= 200 && sz <= 320, "unexpected header size {sz}");
    }

    #[test]
    fn magic_constant_value() {
        assert_eq!(MAGIC, *b"RINHA026");
    }
}

pub const MAX_AMOUNT: f32 = 10_000.0;
pub const MAX_INSTALLMENTS: f32 = 12.0;
pub const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
pub const MAX_MINUTES: f32 = 1440.0;
pub const MAX_KM: f32 = 1000.0;
pub const MAX_TX_COUNT_24H: f32 = 20.0;
pub const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

/// Map a value already in roughly `[-1.0, 1.0]` to `i8`. Saturating.
/// Sentinel `-1.0` round-trips to `-127`. Values in `[0,1]` map to `[0, 127]`.
#[inline]
pub fn quantize_unit(v: f32) -> i8 {
    let scaled = (v * 127.0).round();
    if scaled >= 127.0 {
        127
    } else if scaled <= -127.0 {
        -127
    } else {
        scaled as i8
    }
}

#[inline]
pub fn clamp01(v: f32) -> f32 {
    if v.is_nan() || v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

#[cfg(test)]
mod feature_tests {
    use super::*;

    #[test]
    fn quantize_unit_clamps_to_127() {
        assert_eq!(quantize_unit(2.0), 127);
    }
    #[test]
    fn quantize_unit_zero_is_zero() {
        assert_eq!(quantize_unit(0.0), 0);
    }
    #[test]
    fn quantize_unit_half_is_64() {
        assert_eq!(quantize_unit(0.5), 64);
    }
    #[test]
    fn quantize_unit_negative_clamps_to_minus_127() {
        assert_eq!(quantize_unit(-1.0), -127);
    }
}

/// Pre-normalized, parsed-out fields. Caller is responsible for deriving
/// `hour_of_day`, `day_of_week`, `unknown_merchant`, `mcc_risk` from the raw payload.
#[derive(Debug, Clone, Copy)]
pub struct RawFeatures {
    pub amount: f32,
    pub installments: u32,
    pub hour_of_day: u8, // 0..=23 UTC
    pub day_of_week: u8, // Mon=0..Sun=6
    pub minutes_since_last_tx: Option<f32>,
    pub km_from_last_tx: Option<f32>,
    pub km_from_home: f32,
    pub customer_avg_amount: f32,
    pub tx_count_24h: u32,
    pub is_online: bool,
    pub card_present: bool,
    pub unknown_merchant: bool,
    pub mcc_risk: f32, // 0..=1, default 0.5
    pub merchant_avg_amount: f32,
}

/// Same canonical vectorization, but returning unquantized f32 in the range
/// [-1.0, 1.0] (with the `-1.0` sentinel where the original returned -127).
/// The IVF SoA scan does its compute in f32, so we hand it the float values
/// directly and let the search-side scan apply its own quantization scale.
#[inline]
pub fn vectorize_f32(r: &RawFeatures) -> [f32; VECTOR_DIM] {
    let mut v = [0.0f32; VECTOR_DIM];

    v[0] = clamp01(r.amount / MAX_AMOUNT);
    v[1] = clamp01(r.installments as f32 / MAX_INSTALLMENTS);

    let amount_vs_avg = if r.customer_avg_amount > 0.0 {
        clamp01((r.amount / r.customer_avg_amount) / AMOUNT_VS_AVG_RATIO)
    } else {
        0.0
    };
    v[2] = amount_vs_avg;

    v[3] = r.hour_of_day as f32 / 23.0;
    v[4] = r.day_of_week as f32 / 6.0;

    v[5] = match r.minutes_since_last_tx {
        Some(m) => clamp01(m / MAX_MINUTES),
        None => -1.0,
    };
    v[6] = match r.km_from_last_tx {
        Some(k) => clamp01(k / MAX_KM),
        None => -1.0,
    };

    v[7] = clamp01(r.km_from_home / MAX_KM);
    v[8] = clamp01(r.tx_count_24h as f32 / MAX_TX_COUNT_24H);
    v[9] = if r.is_online { 1.0 } else { 0.0 };
    v[10] = if r.card_present { 1.0 } else { 0.0 };
    v[11] = if r.unknown_merchant { 1.0 } else { 0.0 };
    v[12] = clamp01(r.mcc_risk);
    v[13] = clamp01(r.merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT);

    v
}

#[inline]
pub fn vectorize(r: &RawFeatures) -> [i8; VECTOR_DIM] {
    let v_f32 = vectorize_f32(r);
    let mut v = [0i8; VECTOR_DIM];
    for i in 0..VECTOR_DIM {
        v[i] = quantize_unit(v_f32[i]);
    }
    v
}

#[cfg(test)]
mod vectorize_tests {
    use super::*;

    #[test]
    fn vectorize_known_legit_example() {
        // tx-1329056812: amount 41.12, installments 2, hour 18 UTC, last_tx null
        let raw = RawFeatures {
            amount: 41.12,
            installments: 2,
            hour_of_day: 18,
            day_of_week: 2, // Wed
            minutes_since_last_tx: None,
            km_from_last_tx: None,
            km_from_home: 29.2331,
            customer_avg_amount: 82.24,
            tx_count_24h: 3,
            is_online: false,
            card_present: true,
            unknown_merchant: false,
            mcc_risk: 0.15,
            merchant_avg_amount: 60.25,
        };
        let v = vectorize(&raw);
        assert_eq!(v[5], -127); // sentinel for null minutes
        assert_eq!(v[6], -127); // sentinel for null km
        assert_eq!(v[9], 0); // is_online false
        assert_eq!(v[10], 127); // card_present true
        assert_eq!(v[11], 0); // merchant known
    }
}
