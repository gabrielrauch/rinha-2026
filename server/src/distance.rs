use shared::VECTOR_DIM;

pub fn l2_squared_scalar(a: &[i8; VECTOR_DIM], b: &[i8; VECTOR_DIM]) -> i32 {
    let mut s: i32 = 0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = *av as i32 - *bv as i32;
        s += d * d;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure AVX2 is available at runtime. Use `l2_squared` for safe dispatch.
pub unsafe fn l2_squared_avx2(a: &[i8; VECTOR_DIM], b: &[i8; VECTOR_DIM]) -> i32 {
    use core::arch::x86_64::*;

    // Load 16 bytes (we only have 14 valid) into __m128i with high 2 lanes zeroed.
    let mut a_buf = [0i8; 16];
    let mut b_buf = [0i8; 16];
    a_buf[..VECTOR_DIM].copy_from_slice(a);
    b_buf[..VECTOR_DIM].copy_from_slice(b);

    let av = _mm_loadu_si128(a_buf.as_ptr() as *const __m128i);
    let bv = _mm_loadu_si128(b_buf.as_ptr() as *const __m128i);

    // Promote i8 → i16 (16 lanes via _mm256_cvtepi8_epi16)
    let a16 = _mm256_cvtepi8_epi16(av);
    let b16 = _mm256_cvtepi8_epi16(bv);

    let diff = _mm256_sub_epi16(a16, b16);
    // pmaddwd: multiply pairs of i16, sum into i32 (8 i32 lanes)
    let sq = _mm256_madd_epi16(diff, diff);

    // horizontal sum of 8 i32 lanes
    let lo = _mm256_castsi256_si128(sq);
    let hi = _mm256_extracti128_si256(sq, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf1 = _mm_shuffle_epi32(sum128, 0b01_00_11_10);
    let s2 = _mm_add_epi32(sum128, shuf1);
    let shuf2 = _mm_shuffle_epi32(s2, 0b00_00_00_01);
    let s3 = _mm_add_epi32(s2, shuf2);
    _mm_cvtsi128_si32(s3)
}

/// Dispatch wrapper. On x86_64 with AVX2 at runtime, uses SIMD; otherwise falls back to scalar.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn l2_squared(a: &[i8; VECTOR_DIM], b: &[i8; VECTOR_DIM]) -> i32 {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { l2_squared_avx2(a, b) }
    } else {
        l2_squared_scalar(a, b)
    }
}

#[inline]
#[cfg(not(target_arch = "x86_64"))]
pub fn l2_squared(a: &[i8; VECTOR_DIM], b: &[i8; VECTOR_DIM]) -> i32 {
    l2_squared_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_distance_to_self() {
        let v = [1i8; VECTOR_DIM];
        assert_eq!(l2_squared_scalar(&v, &v), 0);
    }

    #[test]
    fn known_distance() {
        let a = [0i8; VECTOR_DIM];
        let b = [3i8; VECTOR_DIM];
        // 14 dimensions * 9 = 126
        assert_eq!(l2_squared_scalar(&a, &b), 126);
    }

    #[test]
    fn negative_safe() {
        let a = [-100i8; VECTOR_DIM];
        let b = [100i8; VECTOR_DIM];
        // 14 * 200^2 = 14 * 40000 = 560_000
        assert_eq!(l2_squared_scalar(&a, &b), 560_000);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn simd_matches_scalar(
            a in proptest::collection::vec(any::<i8>(), VECTOR_DIM..=VECTOR_DIM),
            b in proptest::collection::vec(any::<i8>(), VECTOR_DIM..=VECTOR_DIM)
        ) {
            // Convert Vec<i8> to [i8; VECTOR_DIM]
            let a_arr: [i8; VECTOR_DIM] = a.try_into().unwrap();
            let b_arr: [i8; VECTOR_DIM] = b.try_into().unwrap();

            let scalar = l2_squared_scalar(&a_arr, &b_arr);
            // Only call AVX2 if the runtime CPU supports it (CI x86_64 should).
            if std::is_x86_feature_detected!("avx2") {
                let simd = unsafe { l2_squared_avx2(&a_arr, &b_arr) };
                prop_assert_eq!(scalar, simd);
            } else {
                // Fallback test: dispatch wrapper should equal scalar
                let dispatch = l2_squared(&a_arr, &b_arr);
                prop_assert_eq!(scalar, dispatch);
            }
        }
    }
}
