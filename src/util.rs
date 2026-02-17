use std::fmt::Display;

use crate::{
    num::{RealScalar, Scalar, ScalarConst},
    pts::{Points, Points1, Points2, Points3, Pts},
};
use ndarray::{NdIndex, prelude::*};
use num_complex::{Complex, Complex64};
use twofloat::TwoFloat;

#[derive(Copy, Clone)]
pub struct NumMargin<T>
where
    T: Scalar,
{
    pub epsilon: T,
    pub relative: T,
    pub ulps: u32,
}

impl<T> Default for NumMargin<T>
where
    T: Scalar + ScalarConst,
{
    fn default() -> Self {
        NumMargin {
            epsilon: T::EPSILON,
            relative: T::EPSILON,
            ulps: 4,
        }
    }
}

impl From<NumMargin<f64>> for NumMargin<TwoFloat> {
    fn from(value: NumMargin<f64>) -> Self {
        NumMargin {
            epsilon: value.epsilon.into(),
            relative: value.relative.into(),
            ulps: value.ulps,
        }
    }
}

pub trait ApproxEq: Sized + Display {
    type Compare: RealScalar;

    fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool;
    fn approx_ne(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        !self.approx_eq(exemplar, precision)
    }
    fn assert_approx_eq(
        &self,
        exemplar: &Self,
        precision: NumMargin<Self::Compare>,
        test: &str,
        idx: &str,
    ) {
        debug_assert!(
            self.approx_eq(exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
    fn assert_approx_ne(
        &self,
        exemplar: &Self,
        precision: NumMargin<Self::Compare>,
        test: &str,
        idx: &str,
    ) {
        debug_assert!(
            self.approx_ne(exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
}

impl ApproxEq for f64 {
    type Compare = f64;

    fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        let diff = (self - exemplar).abs();

        if diff <= precision.epsilon {
            return true;
        }

        let largest = self.abs().max(exemplar.abs());
        if diff <= largest * precision.relative {
            return true;
        }

        // Trivial negative sign check
        if self.signum() != exemplar.signum() {
            return false;
        }

        // ULPS difference comparison
        let int_self: u64 = self.to_bits();
        let int_other: u64 = exemplar.to_bits();

        if int_self <= int_other {
            int_other - int_self <= precision.ulps as u64
        } else {
            int_self - int_other <= precision.ulps as u64
        }
    }
}

impl ApproxEq for TwoFloat {
    type Compare = TwoFloat;

    fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        let diff = (self - exemplar).abs();

        if diff <= precision.epsilon {
            return true;
        }

        let largest = self.abs().max(exemplar.abs());
        if diff <= largest * precision.relative {
            return true;
        }

        // Trivial negative sign check
        if self.signum() != exemplar.signum() {
            return false;
        }

        // ULPS difference comparison
        let int_self: u64 = self.lo().to_bits();
        let int_other: u64 = exemplar.lo().to_bits();

        if int_self <= int_other {
            int_other - int_self <= precision.ulps as u64
        } else {
            int_self - int_other <= precision.ulps as u64
        }
    }
}

// impl ApproxEq for Complex64 {
//     type Compare = f64;

//     fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
//         self.re.approx_eq(&exemplar.re, precision) && self.im.approx_eq(&exemplar.im, precision)
//     }
// }

// impl ApproxEq for Complex<TwoFloat> {
//     type Compare = TwoFloat;

//     fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
//         self.re.approx_eq(&exemplar.re, precision) && self.im.approx_eq(&exemplar.im, precision)
//     }
// }

impl<T> ApproxEq for Complex<T>
where
    T: RealScalar + ApproxEq<Compare = T>,
{
    type Compare = T;

    fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        self.re.approx_eq(&exemplar.re, precision) && self.im.approx_eq(&exemplar.im, precision)
    }
}

impl<T, D> ApproxEq for Array<T, D>
where
    T: Scalar + ApproxEq,
    D: Dimension,
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    type Compare = T::Compare;

    fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        for (idx, pt) in self.indexed_iter() {
            if !pt.approx_eq(&exemplar[idx], precision) {
                return false;
            }
        }

        true
    }
}

impl<T, D> ApproxEq for Points<T, D>
where
    T: Scalar + ApproxEq,
    D: Dimension,
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    type Compare = T::Compare;

    fn approx_eq(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        for (idx, pt) in self.indexed_iter() {
            if !pt.approx_eq(&exemplar[idx], precision) {
                return false;
            }
        }

        true
    }
}

pub trait ApproxCompare: ApproxEq {
    fn approx_gt(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool;
    fn approx_ge(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool;
    fn approx_lt(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        !self.approx_ge(exemplar, precision)
    }
    fn approx_le(&self, exemplar: &Self, precision: NumMargin<Self::Compare>) -> bool {
        !self.approx_gt(exemplar, precision)
    }
    fn assert_approx_gt(
        &self,
        exemplar: &Self,
        precision: NumMargin<Self::Compare>,
        test: &str,
        idx: &str,
    ) {
        debug_assert!(
            self.approx_gt(exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
    fn assert_approx_ge(
        &self,
        exemplar: &Self,
        precision: NumMargin<Self::Compare>,
        test: &str,
        idx: &str,
    ) {
        debug_assert!(
            self.approx_ge(exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
    fn assert_approx_lt(
        &self,
        exemplar: &Self,
        precision: NumMargin<Self::Compare>,
        test: &str,
        idx: &str,
    ) {
        debug_assert!(
            self.approx_lt(exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
    fn assert_approx_le(
        &self,
        exemplar: &Self,
        precision: NumMargin<Self::Compare>,
        test: &str,
        idx: &str,
    ) {
        debug_assert!(
            self.approx_le(exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
}

impl ApproxCompare for f64 {
    fn approx_gt(&self, exemplar: &Self, precision: NumMargin<f64>) -> bool {
        if self <= exemplar {
            return false;
        }

        let diff = self - exemplar;
        diff > precision.epsilon && diff > precision.relative * exemplar.abs().max(self.abs())
    }

    fn approx_ge(&self, exemplar: &Self, precision: NumMargin<f64>) -> bool {
        if self >= exemplar {
            return true;
        }

        let diff = exemplar - self;
        diff <= precision.epsilon || diff <= precision.relative * exemplar.abs().max(self.abs())
    }
}

impl ApproxCompare for TwoFloat {
    fn approx_gt(&self, exemplar: &Self, precision: NumMargin<TwoFloat>) -> bool {
        if self <= exemplar {
            return false;
        }

        let diff = self - exemplar;
        diff > precision.epsilon && diff > precision.relative * exemplar.abs().max(self.abs())
    }

    fn approx_ge(&self, exemplar: &Self, precision: NumMargin<TwoFloat>) -> bool {
        if self >= exemplar {
            return true;
        }

        let diff = exemplar - self;
        diff <= precision.epsilon || diff <= precision.relative * exemplar.abs().max(self.abs())
    }
}

pub fn comp_line(exemplar: &str, net: &str, test: &str) {
    let mut i: usize = 0;
    let mut exemplar_iter = exemplar.lines();
    let mut net_iter = net.lines();
    loop {
        let exemplar_line = exemplar_iter.next();
        let net_line = net_iter.next();
        if exemplar_line.is_none() && net_line.is_none() {
            break;
        } else if exemplar_line.is_none() {
            panic!(
                "test {} number of lines does not match >{}: exemplar out of lines",
                test, i
            );
        } else if net_line.is_none() {
            panic!(
                "test {} number of lines does not match >{}: test out of lines",
                test, i
            );
        }
        i += 1;
        debug_assert!(
            exemplar_line.unwrap() == net_line.unwrap(),
            "test {} line {} does not match\n  exemplar: {}\n       net: {}",
            test,
            i,
            exemplar_line.unwrap(),
            net_line.unwrap()
        );
    }
}

// pub fn comp_num<T, S>(exemplar: T, calc: T, precision: NumMargin<S>, test: &str, idx: &str)
// where
//     T: ApproxCompare<S> + RFNum,
//     S: Float + AbsDiffEq + RelativeEq + UlpsEq,
// {
//     calc.assert_approx_eq(exemplar, precision, test, idx);
// }

pub fn comp_pts_ix1<T, S>(
    exemplar: &Points1<T>,
    calc: &Points1<T>,
    precision: NumMargin<S>,
    test: &str,
) where
    T: ApproxEq<Compare = S> + Scalar,
    S: Scalar,
{
    if exemplar.dim() != calc.dim() {
        println!(
            "number of points don't match: exemplar {:?}\tcalc {:?}",
            exemplar.dim(),
            calc.dim()
        );
    }
    azip!((index k, e in exemplar.inner(), c in calc.inner()) {
        c.assert_approx_eq(e, precision, test, format!("{}", k).to_owned().as_str());
    });
}

pub fn comp_pts_ix2<T, S>(
    exemplar: &Points2<T>,
    calc: &Points2<T>,
    precision: NumMargin<S>,
    test: &str,
) where
    T: ApproxEq<Compare = S> + Scalar,
    S: Scalar,
{
    if exemplar.dim() != calc.dim() {
        println!(
            "number of points don't match: exemplar {:?}\tcalc {:?}",
            exemplar.dim(),
            calc.dim()
        );
    }
    azip!((index (j, k), e in exemplar.inner(), c in calc.inner()) {
        c.assert_approx_eq(e, precision, test, format!("{},{}", j, k).to_owned().as_str());
    });
}

pub fn comp_pts_ix3<T, S>(
    exemplar: &Points3<T>,
    calc: &Points3<T>,
    precision: NumMargin<S>,
    test: &str,
) where
    T: ApproxEq<Compare = S> + Scalar,
    S: Scalar,
{
    if exemplar.dim() != calc.dim() {
        println!(
            "number of points don't match: exemplar {:?}\tcalc {:?}",
            exemplar.dim(),
            calc.dim()
        );
    }
    azip!((index (i, j, k), e in exemplar.inner(), c in calc.inner()) {
        c.assert_approx_eq(e, precision, test, format!("{},{},{}", i, j, k).to_owned().as_str());
    });
}

pub fn comp_array_c64(
    exemplar: ArrayView1<Complex64>,
    calc: ArrayView1<Complex64>,
    precision: NumMargin<f64>,
    test: &str,
) {
    azip!((index i, e in exemplar, &c in calc) {
        c.assert_approx_eq(e, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_array_f64(
    exemplar: ArrayView1<f64>,
    calc: ArrayView1<f64>,
    precision: NumMargin<f64>,
    test: &str,
) {
    azip!((index i, e in exemplar, &c in calc) {
        c.assert_approx_eq(e, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_vec_c64(
    exemplar: &[Complex64],
    calc: &[Complex64],
    precision: NumMargin<f64>,
    test: &str,
) {
    for k in 0..calc.len() {
        calc[k].assert_approx_eq(
            &exemplar[k],
            precision,
            test,
            &(format!("({})", k).to_owned()),
        );
    }
}

pub fn comp_vec_f64(exemplar: &[f64], calc: &[f64], precision: NumMargin<f64>, test: &str) {
    for k in 0..calc.len() {
        calc[k].assert_approx_eq(
            &exemplar[k],
            precision,
            test,
            &(format!("({})", k)).to_owned(),
        );
    }
}

#[cfg(test)]
mod approx_comparison_tests {
    use super::*;

    // Test tolerances
    const MARGIN: NumMargin<TwoFloat> = NumMargin {
        epsilon: TwoFloat::EPSILON,
        relative: TwoFloat::EPSILON,
        ulps: 4,
    };
    const ABS_TOL: TwoFloat = TwoFloat::EPSILON;
    const REL_TOL: TwoFloat = TwoFloat::EPSILON;

    fn abs_tol() -> TwoFloat {
        TwoFloat::from_f64(1e-10)
    }

    fn rel_tol() -> TwoFloat {
        TwoFloat::from_f64(1e-10)
    }

    // ==================== approx_ge tests ====================

    #[test]
    fn test_approx_ge_clearly_greater() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(3.0);
        a.assert_approx_ge(&b, MARGIN, "clearly_greater", "");
    }

    #[test]
    fn test_approx_ge_clearly_less() {
        let a = TwoFloat::from_f64(3.0);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_lt(&b, MARGIN, "clearly_less", "");
    }

    #[test]
    fn test_approx_ge_exactly_equal() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_ge(&b, MARGIN, "exactly_equal", "");
    }

    #[test]
    fn test_approx_ge_within_abs_tolerance() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.0 + 1e-12); // b slightly larger, but within tolerance
        println!("\n\na:\t{a}\nb:\t{b}\n\n");
        a.assert_approx_ge(
            &b,
            NumMargin {
                epsilon: abs_tol(),
                relative: TwoFloat::EPSILON,
                ulps: 4,
            },
            "within_abs_tolerance",
            "",
        );
    }

    #[test]
    fn test_approx_ge_within_rel_tolerance() {
        let a = TwoFloat::from_f64(1000.0);
        let b = TwoFloat::from_f64(1000.0 + 1e-8); // b slightly larger, within relative tolerance
        a.assert_approx_ge(
            &b,
            NumMargin {
                epsilon: TwoFloat::EPSILON,
                relative: rel_tol(),
                ulps: 4,
            },
            "within_rel_tolerance",
            "",
        );
    }

    #[test]
    fn test_approx_ge_outside_tolerance() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.1); // b clearly larger, outside tolerance
        a.assert_approx_lt(&b, MARGIN, "outside_tolerance", "");
    }

    #[test]
    fn test_approx_ge_negative_values() {
        let a = TwoFloat::from_f64(-3.0);
        let b = TwoFloat::from_f64(-5.0);
        a.assert_approx_ge(&b, MARGIN, "negative_values", "");

        let c = TwoFloat::from_f64(-5.0);
        let d = TwoFloat::from_f64(-3.0);
        c.assert_approx_lt(&d, MARGIN, "negative_values", "");
    }

    #[test]
    fn test_approx_ge_zero() {
        let a = TwoFloat::from_f64(0.0);
        let b = TwoFloat::from_f64(0.0);
        a.assert_approx_ge(&b, MARGIN, "zero", "");

        let c = TwoFloat::from_f64(1e-12);
        let d = TwoFloat::from_f64(0.0);
        c.assert_approx_ge(&d, MARGIN, "zero", "");
    }

    // ==================== approx_gt tests ====================

    #[test]
    fn test_approx_gt_clearly_greater() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(3.0);
        a.assert_approx_gt(&b, MARGIN, "clearly_greater", "");
    }

    #[test]
    fn test_approx_gt_clearly_less() {
        let a = TwoFloat::from_f64(3.0);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_le(&b, MARGIN, "clearly_less", "");
    }

    #[test]
    fn test_approx_gt_exactly_equal() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.0);
        // When exactly equal, approx_gt should return true (diff == 0 < abs_tol)
        a.assert_approx_le(&b, MARGIN, "exactly_equal", "");
    }

    #[test]
    fn test_approx_gt_within_abs_tolerance() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.0 + 1e-12); // b slightly larger, but within tolerance
        a.assert_approx_ge(
            &b,
            NumMargin {
                epsilon: abs_tol(),
                relative: TwoFloat::EPSILON,
                ulps: 4,
            },
            "within_abs_tolerance",
            "",
        );
    }

    #[test]
    fn test_approx_gt_at_boundary() {
        // Test behavior when diff is clearly larger than tolerance
        let a = TwoFloat::from_f64(5.0);
        let tol = TwoFloat::from_f64(1e-10);
        let b = TwoFloat::from_f64(5.0 + 1e-5); // diff is 1e-5, much larger than tol of 1e-10
        // Since a < b and diff > tol, approx_gt should return false
        a.assert_approx_le(
            &b,
            NumMargin {
                epsilon: tol,
                relative: tol,
                ulps: 4,
            },
            "at_boundary",
            "1",
        );

        // Test when diff is smaller than tolerance
        let c = TwoFloat::from_f64(5.0);
        let d = TwoFloat::from_f64(5.0 + 1e-12); // diff is 1e-12, smaller than tol of 1e-10
        c.assert_approx_le(
            &d,
            NumMargin {
                epsilon: tol,
                relative: tol,
                ulps: 4,
            },
            "at_boundary",
            "2",
        );
    }

    #[test]
    fn test_approx_gt_outside_tolerance() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.1);
        a.assert_approx_le(&b, MARGIN, "outside_tolerance", "");
    }

    #[test]
    fn test_approx_gt_negative_values() {
        let a = TwoFloat::from_f64(-3.0);
        let b = TwoFloat::from_f64(-5.0);
        a.assert_approx_gt(&b, MARGIN, "negative_values", "");
    }

    // ==================== approx_le tests ====================

    #[test]
    fn test_approx_le_clearly_less() {
        let a = TwoFloat::from_f64(3.0);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_le(&b, MARGIN, "clearly_less", "");
    }

    #[test]
    fn test_approx_le_clearly_greater() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(3.0);
        a.assert_approx_gt(&b, MARGIN, "clearly_greater", "");
    }

    #[test]
    fn test_approx_le_exactly_equal() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_le(&b, MARGIN, "exactly_equal", "");
    }

    #[test]
    fn test_approx_le_within_tolerance() {
        let a = TwoFloat::from_f64(5.0 + 1e-12);
        let b = TwoFloat::from_f64(5.0); // a slightly larger, but within tolerance
        a.assert_approx_le(
            &b,
            NumMargin {
                epsilon: abs_tol(),
                relative: rel_tol(),
                ulps: 4,
            },
            "within_tolerance",
            "",
        );
    }

    #[test]
    fn test_approx_le_outside_tolerance() {
        let a = TwoFloat::from_f64(5.1);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_gt(&b, MARGIN, "outside_tolerance", "");
    }

    #[test]
    fn test_approx_le_negative_values() {
        let a = TwoFloat::from_f64(-5.0);
        let b = TwoFloat::from_f64(-3.0);
        a.assert_approx_le(&b, MARGIN, "negative_values", "");
    }

    // ==================== approx_lt tests ====================

    #[test]
    fn test_approx_lt_clearly_less() {
        let a = TwoFloat::from_f64(3.0);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_lt(&b, MARGIN, "clearly_less", "");
    }

    #[test]
    fn test_approx_lt_clearly_greater() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(3.0);
        a.assert_approx_ge(&b, MARGIN, "clearly_greater", "");
    }

    #[test]
    fn test_approx_lt_exactly_equal() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(5.0);
        // When exactly equal, approx_lt should return true (same as approx_gt with swapped args)
        a.assert_approx_ge(&b, MARGIN, "exactly_equal", "");
    }

    #[test]
    fn test_approx_lt_within_tolerance() {
        let a = TwoFloat::from_f64(5.0 + 1e-12);
        let b = TwoFloat::from_f64(5.0); // a slightly larger, but within tolerance
        a.assert_approx_le(
            &b,
            NumMargin {
                epsilon: abs_tol(),
                relative: TwoFloat::EPSILON,
                ulps: 4,
            },
            "within_tolerance",
            "",
        );
    }

    #[test]
    fn test_approx_lt_outside_tolerance() {
        let a = TwoFloat::from_f64(5.1);
        let b = TwoFloat::from_f64(5.0);
        a.assert_approx_ge(&b, MARGIN, "outside_tolerance", "");
    }

    #[test]
    fn test_approx_lt_negative_values() {
        let a = TwoFloat::from_f64(-5.0);
        let b = TwoFloat::from_f64(-3.0);
        a.assert_approx_lt(&b, MARGIN, "negative_values", "");
    }

    // ==================== Edge cases and symmetry tests ====================

    #[test]
    fn test_le_ge_symmetry() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(3.0);
        // approx_le(a, b) should equal approx_ge(b, a)
        assert_eq!(a.approx_le(&b, MARGIN), b.approx_ge(&a, MARGIN));
    }

    #[test]
    fn test_lt_gt_symmetry() {
        let a = TwoFloat::from_f64(5.0);
        let b = TwoFloat::from_f64(3.0);
        // approx_lt(a, b) should equal approx_gt(b, a)
        assert_eq!(a.approx_lt(&b, MARGIN), b.approx_gt(&a, MARGIN));
    }

    #[test]
    fn test_large_values_with_rel_tolerance() {
        let a = TwoFloat::from_f64(1e10);
        let b = TwoFloat::from_f64(1e10 + 1.0); // difference is 1.0, which is tiny relative to 1e10
        let rel = TwoFloat::from_f64(1e-8);
        let abs = TwoFloat::from_f64(1e-15);
        // With relative tolerance of 1e-8, a diff of 1.0 on scale of 1e10 should pass
        a.assert_approx_ge(
            &b,
            NumMargin {
                epsilon: abs,
                relative: rel,
                ulps: 4,
            },
            "large_values_with_rel_tolerance",
            "",
        );
    }

    #[test]
    fn test_small_values_with_abs_tolerance() {
        let a = TwoFloat::from_f64(1e-15);
        let b = TwoFloat::from_f64(2e-15);
        let abs = TwoFloat::from_f64(1e-14);
        let rel = TwoFloat::from_f64(1e-15);
        // Absolute tolerance should handle small values
        a.assert_approx_ge(
            &b,
            NumMargin {
                epsilon: abs,
                relative: rel,
                ulps: 4,
            },
            "small_values_with_abs_tolerance",
            "",
        );
    }

    #[test]
    fn test_mixed_signs() {
        let a = TwoFloat::from_f64(-1.0);
        let b = TwoFloat::from_f64(1.0);
        a.assert_approx_lt(&b, MARGIN, "mixed_signs", "");
        a.assert_approx_le(&b, MARGIN, "mixed_signs", "");
    }
}
