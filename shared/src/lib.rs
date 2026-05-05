//! Types shared between the offline builder and the runtime server.
//!
//! Blob layout (little-endian):
//!
//! ```text
//! [BlobHeader: 64 bytes]
//! [centroids: NUM_CENTROIDS * 14 bytes (i8)]
//! [cluster_offsets: (NUM_CENTROIDS + 1) * 4 bytes (u32)]
//! [vectors: total_vectors * 14 bytes (i8), grouped by cluster]
//! [labels: ceil(total_vectors / 8) bytes, 1 bit per vector (1 = fraud)]
//! [mcc_risk_table: 1024 bytes (i8 × 1024) — direct-index by mcc % 1024]
//! ```

pub const MAGIC: [u8; 8] = *b"RINHA026";
pub const VERSION: u32 = 1;
pub const VECTOR_DIM: usize = 14;
pub const NUM_CENTROIDS: u32 = 2048;
pub const MCC_TABLE_SIZE: usize = 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlobHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub num_centroids: u32,
    pub total_vectors: u32,
    pub centroids_offset: u32,
    pub cluster_offsets_offset: u32,
    pub vectors_offset: u32,
    pub labels_offset: u32,
    pub mcc_table_offset: u32,
    pub blob_size: u32,
    pub _padding: [u8; 20],
}

const _: () = {
    assert!(std::mem::size_of::<BlobHeader>() == 64);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_fixed_64_bytes() {
        assert_eq!(std::mem::size_of::<BlobHeader>(), 64);
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
    pub hour_of_day: u8,             // 0..=23 UTC
    pub day_of_week: u8,             // Mon=0..Sun=6
    pub minutes_since_last_tx: Option<f32>,
    pub km_from_last_tx: Option<f32>,
    pub km_from_home: f32,
    pub customer_avg_amount: f32,
    pub tx_count_24h: u32,
    pub is_online: bool,
    pub card_present: bool,
    pub unknown_merchant: bool,
    pub mcc_risk: f32,               // 0..=1, default 0.5
    pub merchant_avg_amount: f32,
}

#[inline]
pub fn vectorize(r: &RawFeatures) -> [i8; VECTOR_DIM] {
    let mut v = [0i8; VECTOR_DIM];

    v[0] = quantize_unit(clamp01(r.amount / MAX_AMOUNT));
    v[1] = quantize_unit(clamp01(r.installments as f32 / MAX_INSTALLMENTS));

    let amount_vs_avg = if r.customer_avg_amount > 0.0 {
        clamp01((r.amount / r.customer_avg_amount) / AMOUNT_VS_AVG_RATIO)
    } else {
        0.0
    };
    v[2] = quantize_unit(amount_vs_avg);

    v[3] = quantize_unit(r.hour_of_day as f32 / 23.0);
    v[4] = quantize_unit(r.day_of_week as f32 / 6.0);

    v[5] = match r.minutes_since_last_tx {
        Some(m) => quantize_unit(clamp01(m / MAX_MINUTES)),
        None => -127,
    };
    v[6] = match r.km_from_last_tx {
        Some(k) => quantize_unit(clamp01(k / MAX_KM)),
        None => -127,
    };

    v[7] = quantize_unit(clamp01(r.km_from_home / MAX_KM));
    v[8] = quantize_unit(clamp01(r.tx_count_24h as f32 / MAX_TX_COUNT_24H));
    v[9] = if r.is_online { 127 } else { 0 };
    v[10] = if r.card_present { 127 } else { 0 };
    v[11] = if r.unknown_merchant { 127 } else { 0 };
    v[12] = quantize_unit(clamp01(r.mcc_risk));
    v[13] = quantize_unit(clamp01(r.merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT));

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
        assert_eq!(v[9], 0);    // is_online false
        assert_eq!(v[10], 127); // card_present true
        assert_eq!(v[11], 0);   // merchant known
    }
}
