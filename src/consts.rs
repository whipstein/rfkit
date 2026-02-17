#![allow(dead_code)]

use num_complex::Complex;
use twofloat::TwoFloat;

/// Mathematical constants that work interchangeably across `f64`, `TwoFloat`,
/// `Complex<f64>`, and `Complex<TwoFloat>`.
///
/// For real types, the constants are the standard mathematical values.
/// For complex types, the constants are real-valued (imaginary part = 0).
///
/// ```
/// use rfkit::consts::MathConst;
///
/// // Works the same regardless of type:
/// let pi_f64: f64 = f64::PI_C;
/// let pi_tf: TwoFloat = TwoFloat::PI_C;
/// let pi_c64: Complex<f64> = Complex::<f64>::PI_C;
/// let pi_ctf: Complex<TwoFloat> = Complex::<TwoFloat>::PI_C;
/// ```
pub trait MathConst {
    // =========================================================================
    // Integer constants 0..=10
    // =========================================================================

    /// -2
    const CN2: Self;
    /// -1
    const CN1: Self;
    /// 0
    const C0: Self;
    /// 0.5
    const C05: Self;
    /// 1
    const C1: Self;
    /// 2
    const C2: Self;
    /// 3
    const C3: Self;
    /// 4
    const C4: Self;
    /// 5
    const C5: Self;
    /// 6
    const C6: Self;
    /// 7
    const C7: Self;
    /// 8
    const C8: Self;
    /// 9
    const C9: Self;
    /// 10
    const C10: Self;
    /// 20
    const C20: Self;
    /// 100
    const C100: Self;

    // =========================================================================
    // Mathematical constants
    // =========================================================================

    /// Archimedes' constant (pi)
    const PI_C: Self;

    /// Archimedes' constant (2*pi)
    const PI2_C: Self;

    /// The full circle constant (tau = 2 * pi)
    const TAU_C: Self;

    /// Euler's number (e)
    const E_C: Self;

    /// Speed of light in vacuum
    const C_C: Self;

    /// pi / 2
    const FRAC_PI_2_C: Self;

    /// pi / 3
    const FRAC_PI_3_C: Self;

    /// pi / 4
    const FRAC_PI_4_C: Self;

    /// pi / 6
    const FRAC_PI_6_C: Self;

    /// pi / 8
    const FRAC_PI_8_C: Self;

    /// 1 / pi
    const FRAC_1_PI_C: Self;

    /// 2 / pi
    const FRAC_2_PI_C: Self;

    /// 2 / sqrt(pi)
    const FRAC_2_SQRT_PI_C: Self;

    /// 1 / sqrt(2)
    const FRAC_1_SQRT_2_C: Self;

    /// sqrt(2)
    const SQRT_2_C: Self;

    /// ln(2)
    const LN_2_C: Self;

    /// ln(10)
    const LN_10_C: Self;

    /// log2(e)
    const LOG2_E_C: Self;

    /// log10(e)
    const LOG10_E_C: Self;

    /// log2(10)
    const LOG2_10_C: Self;

    /// log10(2)
    const LOG10_2_C: Self;
}

// =============================================================================
// f64
// =============================================================================

impl MathConst for f64 {
    const CN2: Self = -2.0;
    const CN1: Self = -1.0;
    const C0: Self = 0.0;
    const C05: Self = 0.5;
    const C1: Self = 1.0;
    const C2: Self = 2.0;
    const C3: Self = 3.0;
    const C4: Self = 4.0;
    const C5: Self = 5.0;
    const C6: Self = 6.0;
    const C7: Self = 7.0;
    const C8: Self = 8.0;
    const C9: Self = 9.0;
    const C10: Self = 10.0;
    const C20: Self = 20.0;
    const C100: Self = 100.0;
    const PI_C: Self = core::f64::consts::PI;
    const PI2_C: Self = 2.0 * core::f64::consts::PI;
    const TAU_C: Self = core::f64::consts::TAU;
    const E_C: Self = core::f64::consts::E;
    const C_C: Self = 3e8;
    const FRAC_PI_2_C: Self = core::f64::consts::FRAC_PI_2;
    const FRAC_PI_3_C: Self = core::f64::consts::FRAC_PI_3;
    const FRAC_PI_4_C: Self = core::f64::consts::FRAC_PI_4;
    const FRAC_PI_6_C: Self = core::f64::consts::FRAC_PI_6;
    const FRAC_PI_8_C: Self = core::f64::consts::FRAC_PI_8;
    const FRAC_1_PI_C: Self = core::f64::consts::FRAC_1_PI;
    const FRAC_2_PI_C: Self = core::f64::consts::FRAC_2_PI;
    const FRAC_2_SQRT_PI_C: Self = core::f64::consts::FRAC_2_SQRT_PI;
    const FRAC_1_SQRT_2_C: Self = core::f64::consts::FRAC_1_SQRT_2;
    const SQRT_2_C: Self = core::f64::consts::SQRT_2;
    const LN_2_C: Self = core::f64::consts::LN_2;
    const LN_10_C: Self = core::f64::consts::LN_10;
    const LOG2_E_C: Self = core::f64::consts::LOG2_E;
    const LOG10_E_C: Self = core::f64::consts::LOG10_E;
    const LOG2_10_C: Self = core::f64::consts::LOG2_10;
    const LOG10_2_C: Self = core::f64::consts::LOG10_2;
}

// =============================================================================
// TwoFloat
// =============================================================================

impl MathConst for TwoFloat {
    const CN2: Self = TwoFloat::from_f64(-2.0);
    const CN1: Self = TwoFloat::from_f64(-1.0);
    const C0: Self = twofloat::consts::ZERO;
    const C05: Self = TwoFloat::from_f64(0.5);
    const C1: Self = twofloat::consts::ONE;
    const C2: Self = TwoFloat::from_f64(2.0);
    const C3: Self = TwoFloat::from_f64(3.0);
    const C4: Self = TwoFloat::from_f64(4.0);
    const C5: Self = TwoFloat::from_f64(5.0);
    const C6: Self = TwoFloat::from_f64(6.0);
    const C7: Self = TwoFloat::from_f64(7.0);
    const C8: Self = TwoFloat::from_f64(8.0);
    const C9: Self = TwoFloat::from_f64(9.0);
    const C10: Self = TwoFloat::from_f64(10.0);
    const C20: Self = TwoFloat::from_f64(20.0);
    const C100: Self = TwoFloat::from_f64(100.0);
    const PI_C: Self = twofloat::consts::PI;
    const PI2_C: Self = TwoFloat::from_f64(2.0 * core::f64::consts::PI);
    const TAU_C: Self = twofloat::consts::TAU;
    const E_C: Self = twofloat::consts::E;
    const C_C: Self = TwoFloat::from_f64(3e8);
    const FRAC_PI_2_C: Self = twofloat::consts::FRAC_PI_2;
    const FRAC_PI_3_C: Self = twofloat::consts::FRAC_PI_3;
    const FRAC_PI_4_C: Self = twofloat::consts::FRAC_PI_4;
    const FRAC_PI_6_C: Self = twofloat::consts::FRAC_PI_6;
    const FRAC_PI_8_C: Self = twofloat::consts::FRAC_PI_8;
    const FRAC_1_PI_C: Self = twofloat::consts::FRAC_1_PI;
    const FRAC_2_PI_C: Self = twofloat::consts::FRAC_2_PI;
    const FRAC_2_SQRT_PI_C: Self = twofloat::consts::FRAC_2_SQRT_PI;
    const FRAC_1_SQRT_2_C: Self = twofloat::consts::FRAC_1_SQRT_2;
    const SQRT_2_C: Self = twofloat::consts::SQRT_2;
    const LN_2_C: Self = twofloat::consts::LN_2;
    const LN_10_C: Self = twofloat::consts::LN_10;
    const LOG2_E_C: Self = twofloat::consts::LOG2_E;
    const LOG10_E_C: Self = twofloat::consts::LOG10_E;
    const LOG2_10_C: Self = twofloat::consts::LOG2_10;
    const LOG10_2_C: Self = twofloat::consts::LOG10_2;
}

// =============================================================================
// Complex<TwoFloat> — real-valued constants (im = 0)
// =============================================================================

// TwoFloat's zero representation uses the TwoFloat constant, not a literal 0.0
impl<T: MathConst> MathConst for Complex<T> {
    const CN2: Self = Complex {
        re: T::CN2,
        im: T::C0,
    };
    const CN1: Self = Complex {
        re: T::CN1,
        im: T::C0,
    };
    const C0: Self = Complex {
        re: T::C0,
        im: T::C0,
    };
    const C05: Self = Complex {
        re: T::C05,
        im: T::C0,
    };
    const C1: Self = Complex {
        re: T::C1,
        im: T::C0,
    };
    const C2: Self = Complex {
        re: T::C2,
        im: T::C0,
    };
    const C3: Self = Complex {
        re: T::C3,
        im: T::C0,
    };
    const C4: Self = Complex {
        re: T::C4,
        im: T::C0,
    };
    const C5: Self = Complex {
        re: T::C5,
        im: T::C0,
    };
    const C6: Self = Complex {
        re: T::C6,
        im: T::C0,
    };
    const C7: Self = Complex {
        re: T::C7,
        im: T::C0,
    };
    const C8: Self = Complex {
        re: T::C8,
        im: T::C0,
    };
    const C9: Self = Complex {
        re: T::C9,
        im: T::C0,
    };
    const C10: Self = Complex {
        re: T::C10,
        im: T::C0,
    };
    const C20: Self = Complex {
        re: T::C20,
        im: T::C0,
    };
    const C100: Self = Complex {
        re: T::C100,
        im: T::C0,
    };
    const PI_C: Self = Complex {
        re: T::PI_C,
        im: T::C0,
    };
    const PI2_C: Self = Complex {
        re: T::PI2_C,
        im: T::C0,
    };
    const TAU_C: Self = Complex {
        re: T::TAU_C,
        im: T::C0,
    };
    const E_C: Self = Complex {
        re: T::E_C,
        im: T::C0,
    };
    const C_C: Self = Complex {
        re: T::C_C,
        im: T::C0,
    };
    const FRAC_PI_2_C: Self = Complex {
        re: T::FRAC_PI_2_C,
        im: T::C0,
    };
    const FRAC_PI_3_C: Self = Complex {
        re: T::FRAC_PI_3_C,
        im: T::C0,
    };
    const FRAC_PI_4_C: Self = Complex {
        re: T::FRAC_PI_4_C,
        im: T::C0,
    };
    const FRAC_PI_6_C: Self = Complex {
        re: T::FRAC_PI_6_C,
        im: T::C0,
    };
    const FRAC_PI_8_C: Self = Complex {
        re: T::FRAC_PI_8_C,
        im: T::C0,
    };
    const FRAC_1_PI_C: Self = Complex {
        re: T::FRAC_1_PI_C,
        im: T::C0,
    };
    const FRAC_2_PI_C: Self = Complex {
        re: T::FRAC_2_PI_C,
        im: T::C0,
    };
    const FRAC_2_SQRT_PI_C: Self = Complex {
        re: T::FRAC_2_SQRT_PI_C,
        im: T::C0,
    };
    const FRAC_1_SQRT_2_C: Self = Complex {
        re: T::FRAC_1_SQRT_2_C,
        im: T::C0,
    };
    const SQRT_2_C: Self = Complex {
        re: T::SQRT_2_C,
        im: T::C0,
    };
    const LN_2_C: Self = Complex {
        re: T::LN_2_C,
        im: T::C0,
    };
    const LN_10_C: Self = Complex {
        re: T::LN_10_C,
        im: T::C0,
    };
    const LOG2_E_C: Self = Complex {
        re: T::LOG2_E_C,
        im: T::C0,
    };
    const LOG10_E_C: Self = Complex {
        re: T::LOG10_E_C,
        im: T::C0,
    };
    const LOG2_10_C: Self = Complex {
        re: T::LOG2_10_C,
        im: T::C0,
    };
    const LOG10_2_C: Self = Complex {
        re: T::LOG10_2_C,
        im: T::C0,
    };
}

// =============================================================================
// Free functions — type-inferred access to MathConst constants
// =============================================================================

/// -2
#[inline]
pub fn cn2<T: MathConst>() -> T {
    T::CN2
}
/// -1
#[inline]
pub fn cn1<T: MathConst>() -> T {
    T::CN1
}
/// 0
#[inline]
pub fn c0<T: MathConst>() -> T {
    T::C0
}
/// 1
#[inline]
pub fn c1<T: MathConst>() -> T {
    T::C1
}
/// 2
#[inline]
pub fn c2<T: MathConst>() -> T {
    T::C2
}
/// 3
#[inline]
pub fn c3<T: MathConst>() -> T {
    T::C3
}
/// 4
#[inline]
pub fn c4<T: MathConst>() -> T {
    T::C4
}
/// 5
#[inline]
pub fn c5<T: MathConst>() -> T {
    T::C5
}
/// 6
#[inline]
pub fn c6<T: MathConst>() -> T {
    T::C6
}
/// 7
#[inline]
pub fn c7<T: MathConst>() -> T {
    T::C7
}
/// 8
#[inline]
pub fn c8<T: MathConst>() -> T {
    T::C8
}
/// 9
#[inline]
pub fn c9<T: MathConst>() -> T {
    T::C9
}
/// 10
#[inline]
pub fn c10<T: MathConst>() -> T {
    T::C10
}
/// pi
#[inline]
pub fn pi_c<T: MathConst>() -> T {
    T::PI_C
}
/// tau = 2 * pi
#[inline]
pub fn tau_c<T: MathConst>() -> T {
    T::TAU_C
}
/// e
#[inline]
pub fn e_c<T: MathConst>() -> T {
    T::E_C
}
/// pi / 2
#[inline]
pub fn frac_pi_2_c<T: MathConst>() -> T {
    T::FRAC_PI_2_C
}
/// pi / 3
#[inline]
pub fn frac_pi_3_c<T: MathConst>() -> T {
    T::FRAC_PI_3_C
}
/// pi / 4
#[inline]
pub fn frac_pi_4_c<T: MathConst>() -> T {
    T::FRAC_PI_4_C
}
/// pi / 6
#[inline]
pub fn frac_pi_6_c<T: MathConst>() -> T {
    T::FRAC_PI_6_C
}
/// pi / 8
#[inline]
pub fn frac_pi_8_c<T: MathConst>() -> T {
    T::FRAC_PI_8_C
}
/// 1 / pi
#[inline]
pub fn frac_1_pi_c<T: MathConst>() -> T {
    T::FRAC_1_PI_C
}
/// 2 / pi
#[inline]
pub fn frac_2_pi_c<T: MathConst>() -> T {
    T::FRAC_2_PI_C
}
/// 2 / sqrt(pi)
#[inline]
pub fn frac_2_sqrt_pi_c<T: MathConst>() -> T {
    T::FRAC_2_SQRT_PI_C
}
/// 1 / sqrt(2)
#[inline]
pub fn frac_1_sqrt_2_c<T: MathConst>() -> T {
    T::FRAC_1_SQRT_2_C
}
/// sqrt(2)
#[inline]
pub fn sqrt_2_c<T: MathConst>() -> T {
    T::SQRT_2_C
}
/// ln(2)
#[inline]
pub fn ln_2_c<T: MathConst>() -> T {
    T::LN_2_C
}
/// ln(10)
#[inline]
pub fn ln_10_c<T: MathConst>() -> T {
    T::LN_10_C
}
/// log2(e)
#[inline]
pub fn log2_e_c<T: MathConst>() -> T {
    T::LOG2_E_C
}
/// log10(e)
#[inline]
pub fn log10_e_c<T: MathConst>() -> T {
    T::LOG10_E_C
}
/// log2(10)
#[inline]
pub fn log2_10_c<T: MathConst>() -> T {
    T::LOG2_10_C
}
/// log10(2)
#[inline]
pub fn log10_2_c<T: MathConst>() -> T {
    T::LOG10_2_C
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_pi() {
        assert_eq!(f64::PI_C, core::f64::consts::PI);
    }

    #[test]
    fn twofloat_pi() {
        assert_eq!(TwoFloat::PI_C, twofloat::consts::PI);
    }

    #[test]
    fn complex_f64_pi() {
        let pi = Complex::<f64>::PI_C;
        assert_eq!(pi.re, core::f64::consts::PI);
        assert_eq!(pi.im, 0.0);
    }

    #[test]
    fn complex_twofloat_pi() {
        let pi = Complex::<TwoFloat>::PI_C;
        assert_eq!(pi.re, twofloat::consts::PI);
        assert_eq!(pi.im, twofloat::consts::ZERO);
    }

    #[test]
    fn generic_usage() {
        fn two_pi<T: MathConst + std::ops::Add<Output = T> + Copy>() -> T {
            T::PI_C + T::PI_C
        }

        let tau_f64: f64 = two_pi::<f64>();
        assert!((tau_f64 - f64::TAU_C).abs() < 1e-15);

        let tau_c64: Complex<f64> = two_pi::<Complex<f64>>();
        assert!((tau_c64.re - core::f64::consts::TAU).abs() < 1e-15);
        assert_eq!(tau_c64.im, 0.0);
    }

    #[test]
    fn all_constants_nonzero() {
        // Sanity check that none of the constants are accidentally zero
        assert_ne!(f64::PI_C, 0.0);
        assert_ne!(f64::TAU_C, 0.0);
        assert_ne!(f64::E_C, 0.0);
        assert_ne!(f64::FRAC_PI_2_C, 0.0);
        assert_ne!(f64::FRAC_PI_3_C, 0.0);
        assert_ne!(f64::FRAC_PI_4_C, 0.0);
        assert_ne!(f64::FRAC_PI_6_C, 0.0);
        assert_ne!(f64::FRAC_PI_8_C, 0.0);
        assert_ne!(f64::FRAC_1_PI_C, 0.0);
        assert_ne!(f64::FRAC_2_PI_C, 0.0);
        assert_ne!(f64::FRAC_2_SQRT_PI_C, 0.0);
        assert_ne!(f64::FRAC_1_SQRT_2_C, 0.0);
        assert_ne!(f64::SQRT_2_C, 0.0);
        assert_ne!(f64::LN_2_C, 0.0);
        assert_ne!(f64::LN_10_C, 0.0);
        assert_ne!(f64::LOG2_E_C, 0.0);
        assert_ne!(f64::LOG10_E_C, 0.0);
        assert_ne!(f64::LOG2_10_C, 0.0);
        assert_ne!(f64::LOG10_2_C, 0.0);
    }
}
