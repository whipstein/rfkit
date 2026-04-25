use crate::{num::RealScalar, units::UnitValue};
use ndarray::prelude::*;
use num_complex::{Complex, Complex64, ComplexFloat};
use std::f64::consts::PI;

pub fn y_to_z<T: RealScalar>(y: Complex<T>) -> Complex<T> {
    y.recip()
}

pub fn z_to_y<T: RealScalar>(z: Complex<T>) -> Complex<T> {
    z.recip()
}

pub fn z_to_gamma<T: RealScalar>(z: Complex<T>, z0: Complex<T>) -> Complex<T> {
    (z - z0) / (z + z0)
}

pub fn rc_to_gamma<T: RealScalar, U: UnitValue<T, Value = T>>(
    r: T,
    c: T,
    z0: T,
    freq: &U,
) -> Complex<T> {
    let z = Complex::new(r.recip(), freq.val() * c * 2.0 * PI).recip();

    (z - z0) / (z + z0)
}

pub fn gamma_to_z<T: RealScalar>(gamma: Complex<T>, z0: Complex<T>) -> Complex<T> {
    z0 * (Complex::<T>::ONE + gamma) / (Complex::<T>::ONE - gamma)
}

pub fn gamma_to_z_normalized<T: RealScalar>(gamma: Complex<T>, z0: Complex<T>) -> Complex<T> {
    gamma_to_z(gamma, z0) / z0
}

pub fn rpcp_to_rscs<T: RealScalar, U: UnitValue<T, Value = T>>(rp: T, cp: T, freq: &U) -> (T, T) {
    if cp == T::ZERO {
        (rp, T::ZERO)
    } else {
        let q = freq.val() * cp * rp * 2.0 * PI;
        (rp / (T::ONE + q * q), cp * (T::ONE + q * q) / (q * q))
    }
}

pub fn rscs_to_rpcp<T: RealScalar, U: UnitValue<T, Value = T>>(rs: T, cs: T, freq: &U) -> (T, T) {
    if cs == T::ZERO {
        (rs, T::ZERO)
    } else {
        let q = (freq.val() * cs * rs * 2.0 * PI).recip();
        (rs * (q * q + 1.0), cs * (q * q) / (q * q + 1.0))
    }
}

pub fn rpcp_to_z<T: RealScalar, U: UnitValue<T, Value = T>>(rp: T, cp: T, freq: &U) -> Complex<T> {
    Complex::new(rp.recip(), freq.val() * cp * 2.0 * PI).recip()
}

pub fn z_to_rpcp<T: RealScalar, U: UnitValue<T, Value = T>>(z: Complex<T>, freq: &U) -> (T, T) {
    let y = z.recip();

    if y.im == T::ZERO {
        (y.re.recip(), T::ZERO)
    } else {
        (y.re.recip(), y.im / (freq.val() * 2.0 * PI))
    }
}

pub fn rscs_to_z<T: RealScalar, U: UnitValue<T, Value = T>>(rs: T, cs: T, freq: &U) -> Complex<T> {
    Complex::new(rs, -(freq.val() * cs * 2.0 * PI).recip())
}

pub fn z_to_rscs<T: RealScalar, U: UnitValue<T, Value = T>>(z: Complex<T>, freq: &U) -> (T, T) {
    if z.im == T::ZERO {
        (z.re, T::ZERO)
    } else {
        (z.re, -(freq.val() * z.im * 2.0 * PI).recip())
    }
}

pub fn abs_vec_c64(a: &Vec<Complex64>) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];
    for val in a {
        out.push(val.norm());
    }
    out
}

pub fn abs_vec_f64(a: &Vec<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];
    for val in a {
        out.push(val.abs());
    }
    out
}

pub fn add_vec_c64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<Complex64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    out
}

pub fn add_vec_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<f64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    out
}

pub fn div_vec_c64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<Complex64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] / b[i]);
    }
    out
}

pub fn div_vec_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<f64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] / b[i]);
    }
    out
}

pub fn from_polar_deg<T: RealScalar>(r: T, theta: T) -> Complex<T> {
    Complex::from_polar(r, theta.to_radians())
}

pub fn gradient<T: RealScalar>(pts: &Array1<T>) -> Array1<T> {
    if pts.len() <= 1 {
        return pts.clone();
    }

    let mut out = Array1::zeros(pts.len());

    for (i, pt) in out.iter_mut().enumerate() {
        if i == 0 {
            *pt = pts[i + 1] - pts[i];
        } else if i == pts.len() - 1 {
            *pt = pts[i] - pts[i - 1];
        } else {
            *pt = pts[i + 1] - pts[i - 1];
        }
    }

    out
}

pub fn mul_vec_c64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<Complex64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    out
}

pub fn mul_vec_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<f64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    out
}

pub fn powi_vec_c64(a: &Vec<Complex64>, b: i32) -> Vec<Complex64> {
    let mut out: Vec<Complex64> = vec![];
    for val in a {
        out.push(val.powi(b));
    }
    out
}

pub fn powi_vec_f64(a: &Vec<f64>, b: i32) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];
    for val in a {
        out.push(val.powi(b));
    }
    out
}

pub fn row_to_vec_c64(x: Array1<Complex64>) -> Vec<Complex64> {
    let mut out: Vec<Complex64> = vec![];

    for i in 0..x.len() {
        out.push(*x.get(i).unwrap());
    }

    out
}

pub fn row_to_vec_f64(x: Array1<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];

    for i in 0..x.len() {
        out.push(*x.get(i).unwrap());
    }

    out
}

pub fn scale_vec_c64(a: Complex64, b: &Vec<Complex64>) -> Vec<Complex64> {
    let mut out: Vec<Complex64> = vec![];
    for val in b {
        out.push(a * val);
    }
    out
}

pub fn scale_vec_f64(a: f64, b: &Vec<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];
    for val in b {
        out.push(a * val);
    }
    out
}

pub fn sqrt_phase_unwrap<T: RealScalar>(pts: Array1<Complex<T>>) -> Array1<Complex<T>> {
    // let mut out: Vec<Complex64> = vec![];
    let mut phi = Array1::from_shape_fn(pts.dim(), |i| pts[i].arg());

    phi = unwrap(&phi);

    Array1::from_shape_fn(phi.dim(), |i| {
        Complex::new(pts[i].norm().sqrt(), T::ZERO) * Complex::new(T::ZERO, phi[i] * 0.5).exp()
    })
}

pub fn sqrt_mat_f64(a: Array2<f64>) -> Array2<f64> {
    let mut out: Array2<f64> = Array2::zeros((a.nrows(), a.ncols()));

    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[(i, j)] = a.get((i, j)).unwrap().sqrt();
        }
    }

    out
}

pub fn sqrt_vec_f64(a: &Vec<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];
    for val in a {
        out.push(val.sqrt());
    }
    out
}

pub fn sub_vec_c64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<Complex64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] - b[i]);
    }
    out
}

pub fn sub_vec_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        panic!(
            "{}",
            format!(
                "Vector lengths do not match! a.len() = {}, b.len() = {}",
                a.len(),
                b.len()
            )
        );
    }
    let mut out: Vec<f64> = vec![];
    for i in 0..a.len() {
        out.push(a[i] - b[i]);
    }
    out
}

/// Compute real eigenvalues of a Hermitian (self-adjoint) complex matrix
/// using the Jacobi eigenvalue algorithm on its 2n×2n real symmetric
/// embedding. Since the matrix is Hermitian, all eigenvalues are real.
pub fn eig_hermitian_real<T: RealScalar>(mat: &Array2<Complex<T>>) -> Array1<T> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "Matrix must be square");

    // Build the 2n×2n real symmetric embedding:
    //   [ Re(A)  -Im(A) ]
    //   [ Im(A)   Re(A) ]
    // Each eigenvalue of A appears twice in this embedding.
    let n2 = 2 * n;
    let zero_r = T::ZERO;
    let mut a = Array2::from_elem((n2, n2), zero_r);

    for i in 0..n {
        for j in 0..n {
            let re = mat[[i, j]].re();
            let im = mat[[i, j]].im();
            a[[i, j]] = re;
            a[[i, n + j]] = -im;
            a[[n + i, j]] = im;
            a[[n + i, n + j]] = re;
        }
    }

    eig_symmetric_real_jacobi(&mut a, n)
}

/// Jacobi eigenvalue algorithm for a real symmetric matrix stored in `a`.
/// Returns `n` unique eigenvalues (the 2n×2n embedding produces pairs).
// fn eig_symmetric_real_jacobi<R: RealScalar + MathConst>(a: &mut Array2<R>, n: usize) -> Array1<R> {
fn eig_symmetric_real_jacobi<T: RealScalar>(a: &mut Array2<T>, n: usize) -> Array1<T> {
    let n2 = a.nrows();
    let max_iter = 100;
    let tol = 1e-14;
    let pi_4 = std::f64::consts::FRAC_PI_4;

    for _ in 0..max_iter {
        let mut max_val = T::ZERO;
        let (mut p, mut q) = (0, 1);

        for i in 0..n2 {
            for j in i + 1..n2 {
                let v = a[[i, j]].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        let diff = a[[q, q]] - a[[p, p]];
        let theta = if diff.abs() < tol {
            pi_4.into()
        } else {
            (a[[p, q]] * 2.0 / diff).atan() / 2.0
        };

        let c = theta.cos();
        let s = theta.sin();

        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        a[[p, p]] = c * c * app + s * s * aqq - c * s * apq * 2.0;
        a[[q, q]] = s * s * app + c * c * aqq + c * s * apq * 2.0;
        a[[p, q]] = T::ZERO;
        a[[q, p]] = T::ZERO;

        for i in 0..n2 {
            if i != p && i != q {
                let aip = a[[i, p]];
                let aiq = a[[i, q]];
                a[[i, p]] = c * aip - s * aiq;
                a[[p, i]] = a[[i, p]];
                a[[i, q]] = s * aip + c * aiq;
                a[[q, i]] = a[[i, q]];
            }
        }
    }

    let mut eigs: Vec<T> = (0..n2).map(|i| a[[i, i]]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Each eigenvalue appears twice in the 2n embedding; take every other one.
    Array1::from_vec((0..n).map(|i| eigs[2 * i]).collect())
}

pub fn unwrap<T: RealScalar>(phi: &Array1<T>) -> Array1<T> {
    let mut out = phi.clone();
    for i in 1..out.len() {
        let diff = out[i] - out[i - 1];
        if diff > PI {
            for val in out.iter_mut().skip(i) {
                *val -= 2.0 * PI;
            }
        } else if diff < -PI {
            for val in out.iter_mut().skip(i) {
                *val += 2.0 * PI;
            }
        }
    }
    out
}

#[cfg(test)]
mod math_tests {
    use super::*;
    use crate::{
        units::{ScalarUnitValue, Scale},
        util::{ApproxEq, NumMargin, comp_vec_c64, comp_vec_f64},
    };
    use num_complex::c64;

    #[test]
    fn math_abs_vec() {
        let a: Vec<f64> = vec![10.0, -11.0, 12.0];
        let exemplar: Vec<f64> = vec![10.0, 11.0, 12.0];
        comp_vec_f64(
            &exemplar,
            &abs_vec_f64(&a),
            NumMargin::default(),
            "abs_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<f64> = vec![14.142135623730951, 15.556349186104045, 16.97056274847714];
        comp_vec_f64(
            &exemplar,
            &abs_vec_c64(&a),
            NumMargin::default(),
            "abs_vec_c64()",
        );
    }

    #[test]
    fn math_add_vec() {
        let a: Vec<f64> = vec![0.0, 1.0, 2.0];
        let b: Vec<f64> = vec![10.0, 11.0, 12.0];
        let exemplar: Vec<f64> = vec![10.0, 12.0, 14.0];
        comp_vec_f64(
            &exemplar,
            &add_vec_f64(&a, &b),
            NumMargin::default(),
            "add_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(10.0, -10.0), c64(12.0, -12.0), c64(14.0, -14.0)];
        comp_vec_c64(
            &exemplar,
            &add_vec_c64(&a, &b),
            NumMargin::default(),
            "add_vec_c64()",
        );
    }

    #[test]
    fn math_div_vec() {
        let a: Vec<f64> = vec![0.0, 1.0, 2.0];
        let b: Vec<f64> = vec![10.0, 11.0, 12.0];
        let exemplar: Vec<f64> = vec![0.0, 0.09090909090909091, 0.16666666666666666];
        comp_vec_f64(
            &exemplar,
            &div_vec_f64(&a, &b),
            NumMargin::default(),
            "div_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![
            c64(0.0, 0.0),
            c64(0.09090909090909091, 0.0),
            c64(0.16666666666666666, 0.0),
        ];
        comp_vec_c64(
            &exemplar,
            &div_vec_c64(&a, &b),
            NumMargin::default(),
            "div_vec_c64()",
        );
    }

    #[test]
    fn math_mul_vec() {
        let a: Vec<f64> = vec![0.0, 1.0, 2.0];
        let b: Vec<f64> = vec![10.0, 11.0, 12.0];
        let exemplar: Vec<f64> = vec![0.0, 11.0, 24.0];
        comp_vec_f64(
            &exemplar,
            &mul_vec_f64(&a, &b),
            NumMargin::default(),
            "mul_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(0.0, 0.0), c64(0.0, -22.0), c64(0.0, -48.0)];
        comp_vec_c64(
            &exemplar,
            &mul_vec_c64(&a, &b),
            NumMargin::default(),
            "mul_vec_c64()",
        );
    }

    #[test]
    fn math_powi_vec() {
        let a: Vec<f64> = vec![10.0, 11.0, 12.0];
        let exemplar: Vec<f64> = vec![100.0, 121.0, 144.0];
        comp_vec_f64(
            &exemplar,
            &powi_vec_f64(&a, 2),
            NumMargin::default(),
            "powi_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(0.0, -200.0), c64(0.0, -242.0), c64(0.0, -288.0)];
        comp_vec_c64(
            &exemplar,
            &powi_vec_c64(&a, 2),
            NumMargin::default(),
            "powi_vec_c64()",
        );
    }

    #[test]
    fn math_scale_vec() {
        let a: Vec<f64> = vec![10.0, 11.0, 12.0];
        let exemplar: Vec<f64> = vec![14.1, 15.51, 16.92];
        comp_vec_f64(
            &exemplar,
            &scale_vec_f64(1.41, &a),
            NumMargin::default(),
            "scale_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> =
            vec![c64(-9.5, -37.7), c64(-10.45, -41.47), c64(-11.4, -45.24)];
        comp_vec_c64(
            &exemplar,
            &scale_vec_c64(Complex64::new(1.41, -2.36), &a),
            NumMargin::default(),
            "scale_vec_c64()",
        );
    }

    #[test]
    fn math_sub_vec() {
        let a: Vec<f64> = vec![0.0, 1.0, 2.0];
        let b: Vec<f64> = vec![10.0, 11.0, 12.0];
        let exemplar: Vec<f64> = vec![-10.0, -10.0, -10.0];
        comp_vec_f64(
            &exemplar,
            &sub_vec_f64(&a, &b),
            NumMargin::default(),
            "sub_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(-10.0, 10.0), c64(-10.0, 10.0), c64(-10.0, 10.0)];
        comp_vec_c64(
            &exemplar,
            &sub_vec_c64(&a, &b),
            NumMargin::default(),
            "sub_vec_c64()",
        );
    }

    #[test]
    fn test_z_to_y() {
        let z = c64(42.4, -19.6);
        let y = c64(0.01943242648676395, 0.008982914130673902);

        z_to_y(z).assert_approx_eq(&y, NumMargin::default(), "z_to_y()", "y");
        y_to_z(z).assert_approx_eq(&y, NumMargin::default(), "y_to_z()", "z");
    }

    #[test]
    fn test_z_to_gamma() {
        let z = c64(42.4, -19.6);
        let z0 = 50.0;
        let gamma = c64(-0.03565151895556114, -0.21968365553602814);
        let test = z_to_gamma(z, z0.into());

        test.assert_approx_eq(&gamma, NumMargin::default(), "z_to_gamma()", "gamma");
    }

    #[test]
    fn test_gamma_to_z() {
        let gamma = c64(0.2464, -0.8745);
        let z0 = 100.0;
        let z = c64(13.096841624374102, -131.24096072255193);
        let test = gamma_to_z(gamma, z0.into());

        test.assert_approx_eq(&z, NumMargin::default(), "gamma_to_z()", "z");
    }

    #[test]
    fn test_z_to_rpcp() {
        let z = c64(42.4, -19.6);
        let f = 275.0;
        let rp = 51.46037735849057;
        let cp = 5.198818862788317e-15;
        let test = z_to_rpcp(
            z,
            &ScalarUnitValue::builder()
                .val_scaled(&f, Scale::Giga)
                .build()
                .unwrap(),
        );

        test.0
            .assert_approx_eq(&rp, NumMargin::default(), "z_to_rpcp()", "rp");
        test.1
            .assert_approx_eq(&cp, NumMargin::default(), "z_to_rpcp()", "cp");
    }

    #[test]
    fn test_rpcp_to_z() {
        let z = c64(42.4, -19.6);
        let f = 275.0;
        let rp = 51.46037735849057;
        let cp = 5.198818862788317e-15;
        let test = rpcp_to_z(
            rp,
            cp,
            &ScalarUnitValue::builder()
                .val_scaled(&f, Scale::Giga)
                .build()
                .unwrap(),
        );

        test.assert_approx_eq(&z, NumMargin::default(), "rpcp_to_z()", "z");
    }

    #[test]
    fn test_z_to_rscs() {
        let z = c64(42.4, -19.6);
        let f = 275.0;
        let rs = 42.4;
        let cs = 2.952781875545368e-14;
        let test = z_to_rscs(
            z,
            &ScalarUnitValue::builder()
                .val_scaled(&f, Scale::Giga)
                .build()
                .unwrap(),
        );

        test.0
            .assert_approx_eq(&rs, NumMargin::default(), "z_to_rscs()", "rs");
        test.1
            .assert_approx_eq(&cs, NumMargin::default(), "z_to_rscs()", "cs");
    }

    #[test]
    fn test_rscs_to_y() {
        let y = c64(0.01943242648676395, 0.008982914130673902);
        let f = 275.0;
        let rs = 42.4;
        let cs = 2.952781875545368e-14;
        let test = 1.0
            / rscs_to_z(
                rs,
                cs,
                &ScalarUnitValue::builder()
                    .val_scaled(&f, Scale::Giga)
                    .build()
                    .unwrap(),
            );

        test.assert_approx_eq(&y, NumMargin::default(), "rscs_to_y()", "y");
    }

    #[test]
    fn test_rscs_to_z() {
        let z = c64(42.4, -19.6);
        let f = 275.0;
        let rs = 42.4;
        let cs = 2.952781875545368e-14;
        let test = rscs_to_z(
            rs,
            cs,
            &ScalarUnitValue::builder()
                .val_scaled(&f, Scale::Giga)
                .build()
                .unwrap(),
        );

        test.assert_approx_eq(&z, NumMargin::default(), "rscs_to_z()", "z");
    }

    #[test]
    fn test_rpcp_to_rscs() {
        let f = 275.0;
        let rp = 51.46037735849057;
        let cp = 5.198818862788317e-15;
        let rs = 42.4;
        let cs = 2.952781875545368e-14;
        let test = rpcp_to_rscs(
            rp,
            cp,
            &ScalarUnitValue::builder()
                .val_scaled(&f, Scale::Giga)
                .build()
                .unwrap(),
        );

        test.0
            .assert_approx_eq(&rs, NumMargin::default(), "rpcp_to_rscs()", "rs");
        test.1
            .assert_approx_eq(&cs, NumMargin::default(), "rpcp_to_rscs()", "cs");
    }

    #[test]
    fn test_rscs_to_rpcp() {
        let f = 275.0;
        let rp = 51.46037735849057;
        let cp = 5.198818862788317e-15;
        let rs = 42.4;
        let cs = 2.952781875545368e-14;
        let test = rscs_to_rpcp(
            rs,
            cs,
            &ScalarUnitValue::builder()
                .val_scaled(&f, Scale::Giga)
                .build()
                .unwrap(),
        );

        test.0
            .assert_approx_eq(&rp, NumMargin::default(), "rscs_to_rpcp()", "rp");
        test.1
            .assert_approx_eq(&cp, NumMargin::default(), "rscs_to_rpcp()", "cp");
    }
}
