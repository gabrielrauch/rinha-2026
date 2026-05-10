use crate::blob::Blob;
use crate::payload::RawPayload;
use shared::{vectorize_f32, RawFeatures, VECTOR_DIM};

pub fn vectorize_payload(blob: &Blob, p: &RawPayload<'_>) -> Option<[f32; VECTOR_DIM]> {
    let req_minutes = parse_iso8601_minutes(p.requested_at)?;
    let (hour, dow) = hour_and_dow_from_minutes(req_minutes);
    let mcc = parse_ascii_u32(p.merchant_mcc)?;

    // unknown_merchant: search for `"<merchant_id>"` in known_merchants raw bytes
    let unknown = !contains_quoted(p.known_merchants, p.merchant_id);

    let mcc_risk_q = blob.mcc_risk(mcc);
    let mcc_risk = (mcc_risk_q as f32) / 127.0;

    // Compute minutes_since_last_tx as the actual time delta if last_transaction
    // is present and parseable; None otherwise so vectorize() emits the -127 sentinel.
    let minutes_since_last_tx = match p.last_timestamp {
        Some(ts) => parse_iso8601_minutes(ts).map(|last| ((req_minutes - last) as f32).max(0.0)),
        None => None,
    };

    let raw = RawFeatures {
        amount: p.amount,
        installments: p.installments,
        hour_of_day: hour,
        day_of_week: dow,
        minutes_since_last_tx,
        km_from_last_tx: p.last_km,
        km_from_home: p.km_from_home,
        customer_avg_amount: p.customer_avg_amount,
        tx_count_24h: p.tx_count_24h,
        is_online: p.is_online,
        card_present: p.card_present,
        unknown_merchant: unknown,
        mcc_risk,
        merchant_avg_amount: p.merchant_avg_amount,
    };
    Some(vectorize_f32(&raw))
}

#[inline]
fn parse_ascii_u32(s: &[u8]) -> Option<u32> {
    let mut acc: u32 = 0;
    for &c in s {
        if !c.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((c - b'0') as u32)?;
    }
    Some(acc)
}

/// Parse an ISO-8601 UTC timestamp ("YYYY-MM-DDTHH:MM:SSZ") into minutes since
/// the Unix epoch (1970-01-01). The absolute value is meaningful but only the
/// difference between two such values is consumed downstream.
#[inline]
fn parse_iso8601_minutes(ts: &[u8]) -> Option<i64> {
    if ts.len() < 19 {
        return None;
    }
    let year = parse_ascii_u32(&ts[0..4])? as i64;
    let month = parse_ascii_u32(&ts[5..7])? as i64;
    let day = parse_ascii_u32(&ts[8..10])? as i64;
    let hour = parse_ascii_u32(&ts[11..13])? as i64;
    let minute = parse_ascii_u32(&ts[14..16])? as i64;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 {
        return None;
    }
    let days = days_from_civil(year, month as u32, day as u32);
    Some(days * 1440 + hour * 60 + minute)
}

/// Howard Hinnant's `days_from_civil`: convert (y, m, d) to days since 1970-01-01
/// in the proleptic Gregorian calendar. Correct for all years and leap years.
#[inline]
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let m_adj: u64 = if m > 2 { (m - 3) as u64 } else { (m + 9) as u64 };
    let doy = (153 * m_adj + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

#[inline]
fn hour_and_dow_from_minutes(total_minutes: i64) -> (u8, u8) {
    let mins_in_day = total_minutes.rem_euclid(1440);
    let hour = (mins_in_day / 60) as u8;

    let days = total_minutes.div_euclid(1440);
    // 1970-01-01 was a Thursday. ISO Mon=0..Sun=6 → Thursday = 3.
    let dow = ((days + 3).rem_euclid(7)) as u8;
    (hour, dow)
}

#[inline]
fn contains_quoted(haystack: &[u8], needle: &[u8]) -> bool {
    let mut i = 0;
    while i + needle.len() + 1 < haystack.len() {
        if haystack[i] == b'"'
            && haystack[i + 1..].starts_with(needle)
            && haystack.get(i + 1 + needle.len()) == Some(&b'"')
        {
            return true;
        }
        i += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload;
    use std::path::PathBuf;

    const SAMPLE_NULL: &[u8] = br#"{"id":"tx-100","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.2331},"last_transaction":null}"#;

    const SAMPLE_PRESENT: &[u8] = br#"{"id":"tx-200","transaction":{"amount":50,"installments":1,"requested_at":"2026-03-11T20:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":2,"known_merchants":["MERC-1"]},"merchant":{"id":"MERC-1","mcc":"5411","avg_amount":50},"terminal":{"is_online":false,"card_present":true,"km_from_home":5},"last_transaction":{"timestamp":"2026-03-11T18:00:00Z","km_from_current":3.5}}"#;

    #[test]
    fn null_last_transaction_yields_sentinel() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
        let blob = Blob::open(&path).unwrap();
        let p = payload::extract(SAMPLE_NULL).unwrap();
        let v = vectorize_payload(&blob, &p).unwrap();
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
    }

    #[test]
    fn present_last_transaction_computes_real_minutes() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
        let blob = Blob::open(&path).unwrap();
        let p = payload::extract(SAMPLE_PRESENT).unwrap();
        let v = vectorize_payload(&blob, &p).unwrap();
        // 120 minutes between timestamps → 120/1440 ≈ 0.0833 → quantized ~10
        assert!((0.05f32..0.10).contains(&v[5]), "expected ~0.083 in v[5], got {}", v[5]);
        assert_ne!(v[6], -1.0);
    }

    #[test]
    fn zeller_2026_03_11_is_wednesday() {
        let mins = parse_iso8601_minutes(b"2026-03-11T18:45:53Z").unwrap();
        let (_, dow) = hour_and_dow_from_minutes(mins);
        assert_eq!(dow, 2); // Wed
    }

    #[test]
    fn delta_minutes_correct() {
        let a = parse_iso8601_minutes(b"2026-03-11T20:00:00Z").unwrap();
        let b = parse_iso8601_minutes(b"2026-03-11T18:00:00Z").unwrap();
        assert_eq!(a - b, 120);
    }

    #[test]
    fn delta_across_day_boundary() {
        let a = parse_iso8601_minutes(b"2026-03-12T00:30:00Z").unwrap();
        let b = parse_iso8601_minutes(b"2026-03-11T23:00:00Z").unwrap();
        assert_eq!(a - b, 90);
    }
}
