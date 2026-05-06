use crate::blob::Blob;
use crate::payload::RawPayload;
use shared::{vectorize, RawFeatures, VECTOR_DIM};

pub fn vectorize_payload(blob: &Blob, p: &RawPayload<'_>) -> Option<[i8; VECTOR_DIM]> {
    let (hour, dow) = parse_iso8601_hour_dow(p.requested_at)?;
    let mcc = parse_ascii_u32(p.merchant_mcc)?;

    // unknown_merchant: search for `"<merchant_id>"` in known_merchants raw bytes
    let unknown = !contains_quoted(p.known_merchants, p.merchant_id);

    let mcc_risk_q = blob.mcc_risk(mcc);
    let mcc_risk = (mcc_risk_q as f32) / 127.0;

    let raw = RawFeatures {
        amount: p.amount,
        installments: p.installments,
        hour_of_day: hour,
        day_of_week: dow,
        minutes_since_last_tx: p.last_minutes,
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
    Some(vectorize(&raw))
}

#[inline]
fn parse_ascii_u32(s: &[u8]) -> Option<u32> {
    let mut acc: u32 = 0;
    for &c in s {
        if !c.is_ascii_digit() { return None; }
        acc = acc.checked_mul(10)?.checked_add((c - b'0') as u32)?;
    }
    Some(acc)
}

#[inline]
fn parse_iso8601_hour_dow(ts: &[u8]) -> Option<(u8, u8)> {
    // Format: "YYYY-MM-DDTHH:MM:SSZ"
    if ts.len() < 19 { return None; }
    let year = parse_ascii_u32(&ts[0..4])? as i32;
    let month = parse_ascii_u32(&ts[5..7])? as i32;
    let day = parse_ascii_u32(&ts[8..10])? as i32;
    let hour = parse_ascii_u32(&ts[11..13])? as u8;
    if hour > 23 { return None; }

    // Zeller's congruence (Gregorian). h: 0=Sat..6=Fri.
    let (y, m) = if month < 3 { (year - 1, month + 12) } else { (year, month) };
    let k = y % 100;
    let j = y / 100;
    let h = (day + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 + 5 * j).rem_euclid(7);
    // Convert Zeller (Sat=0..Fri=6) to ISO-style (Mon=0..Sun=6).
    let dow = ((h + 5) % 7) as u8;
    Some((hour, dow))
}

#[inline]
fn contains_quoted(haystack: &[u8], needle: &[u8]) -> bool {
    // haystack is raw array contents like: "MERC-003","MERC-016"
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

    const SAMPLE: &[u8] = br#"{"id":"tx-100","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.2331},"last_transaction":null}"#;

    #[test]
    fn produces_14d_vector() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/blob.bin");
        let blob = Blob::open(&path).unwrap();
        let p = payload::extract(SAMPLE).unwrap();
        let v = vectorize_payload(&blob, &p).unwrap();
        assert_eq!(v[5], -127); // null minutes
        assert_eq!(v[6], -127); // null km
    }

    #[test]
    fn zeller_2026_03_11_is_wednesday() {
        // Wed maps to 2 in Mon=0..Sun=6 convention.
        let (_, dow) = parse_iso8601_hour_dow(b"2026-03-11T18:45:53Z").unwrap();
        assert_eq!(dow, 2);
    }
}
