use crate::unit::UnitVal;
use ndarray::prelude::*;
use num::complex::{Complex64, c64};
use rug::Assign;
use std::f64::consts::PI;

pub fn y_to_z(y: Complex64) -> Complex64 {
    1.0 / y
}

pub fn z_to_y(z: Complex64) -> Complex64 {
    1.0 / z
}

pub fn z_to_gamma(z: Complex64, z0: f64) -> Complex64 {
    let z0: f64 = z0;

    (z - z0) / (z + z0)
}

pub fn rc_to_gamma(r: f64, c: f64, z0: f64, freq: &UnitVal) -> Complex64 {
    let z = 1.0 / c64(1.0 / r, 2.0 * std::f64::consts::PI * freq.val() * c);

    (z - z0) / (z + z0)
}

pub fn gamma_to_z(gamma: Complex64, z0: f64) -> Complex64 {
    z0 * (1.0 + gamma) / (1.0 - gamma)
}

pub fn gamma_to_z_normalized(gamma: Complex64, z0: f64) -> Complex64 {
    gamma_to_z(gamma, z0) / z0
}

pub fn rc_to_z(r: f64, c: f64, freq: &UnitVal) -> Complex64 {
    1.0 / c64(1.0 / r, 2.0 * std::f64::consts::PI * freq.val() * c)
}

pub fn z_to_rc(z: Complex64, freq: &UnitVal) -> (f64, f64) {
    let y = 1.0 / z;

    (1.0 / y.re, y.im / (2.0 * std::f64::consts::PI * freq.val()))
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

pub fn from_polar_deg(r: f64, theta: f64) -> Complex64 {
    Complex64::from_polar(r, theta * 180.0 / PI)
}

pub fn gradient(pts: Array1<f64>) -> Array1<f64> {
    if pts.len() <= 1 {
        return pts.clone();
    }

    let mut out = Array1::<f64>::zeros(pts.len());

    for (i, pt) in out.iter_mut().enumerate() {
        if i == 0 {
            pt.assign(pts[i + 1] - pts[i]);
        } else if i == pts.len() - 1 {
            pt.assign(pts[i] - pts[i - 1]);
        } else {
            pt.assign(pts[i + 1] - pts[i - 1]);
        }
    }
    // match pts.len() {
    //     0 | 1 => return pts.clone(),
    //     2 => {
    //         out.push(pts[1] - pts[0]);
    //         out.push(pts[2] - pts[1]);
    //     }
    //     n => {
    //         out.push(pts[1] - pts[0]);
    //         for i in 1..(n - 1) {
    //             out.push((pts[i + 1] - pts[i - 1]) / 2.0);
    //         }
    //         out.push(pts[n - 1] - pts[n - 2]);
    //     }
    // }

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

pub fn sqrt_phase_unwrap(pts: Array1<Complex64>) -> Array1<Complex64> {
    // let mut out: Vec<Complex64> = vec![];
    let mut phi = Array1::<f64>::from_shape_fn(pts.dim(), |i| pts[i].arg());

    phi = unwrap(&phi);

    Array1::<Complex64>::from_shape_fn(phi.dim(), |i| {
        pts[i].norm().sqrt() * Complex64::new(0.0, 0.5 * phi[i]).exp()
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

pub fn unwrap(phi: &Array1<f64>) -> Array1<f64> {
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
    use crate::scale::Scale;
    use crate::unit::UnitValBuilder;
    use crate::util::{comp_c64, comp_f64, comp_vec_c64, comp_vec_f64};
    use float_cmp::F64Margin;

    #[test]
    fn math_abs_vec() {
        let a: Vec<f64> = vec![10.0, -11.0, 12.0];
        let exemplar: Vec<f64> = vec![10.0, 11.0, 12.0];
        comp_vec_f64(
            &exemplar,
            &abs_vec_f64(&a),
            F64Margin::default(),
            "abs_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<f64> = vec![14.142135623730951, 15.556349186104045, 16.97056274847714];
        comp_vec_f64(
            &exemplar,
            &abs_vec_c64(&a),
            F64Margin::default(),
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
            F64Margin::default(),
            "add_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(10.0, -10.0), c64(12.0, -12.0), c64(14.0, -14.0)];
        comp_vec_c64(
            &exemplar,
            &add_vec_c64(&a, &b),
            F64Margin::default(),
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
            F64Margin::default(),
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
            F64Margin::default(),
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
            F64Margin::default(),
            "mul_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(0.0, 0.0), c64(0.0, -22.0), c64(0.0, -48.0)];
        comp_vec_c64(
            &exemplar,
            &mul_vec_c64(&a, &b),
            F64Margin::default(),
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
            F64Margin::default(),
            "powi_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(0.0, -200.0), c64(0.0, -242.0), c64(0.0, -288.0)];
        comp_vec_c64(
            &exemplar,
            &powi_vec_c64(&a, 2),
            F64Margin::default(),
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
            F64Margin::default(),
            "scale_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> =
            vec![c64(-9.5, -37.7), c64(-10.45, -41.47), c64(-11.4, -45.24)];
        comp_vec_c64(
            &exemplar,
            &scale_vec_c64(Complex64::new(1.41, -2.36), &a),
            F64Margin::default(),
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
            F64Margin::default(),
            "sub_vec_f64()",
        );

        let a: Vec<Complex64> = vec![c64(0.0, 0.0), c64(1.0, -1.0), c64(2.0, -2.0)];
        let b: Vec<Complex64> = vec![c64(10.0, -10.0), c64(11.0, -11.0), c64(12.0, -12.0)];
        let exemplar: Vec<Complex64> = vec![c64(-10.0, 10.0), c64(-10.0, 10.0), c64(-10.0, 10.0)];
        comp_vec_c64(
            &exemplar,
            &sub_vec_c64(&a, &b),
            F64Margin::default(),
            "sub_vec_c64()",
        );
    }

    #[test]
    fn test_z_to_y() {
        let z = c64(42.4, -19.6);
        let y = c64(0.01943242648676395, 0.008982914130673902);

        comp_c64(&z_to_y(z), &y, F64Margin::default(), "z_to_y()", "y");
        comp_c64(&y_to_z(z), &y, F64Margin::default(), "y_to_z()", "z");
    }

    #[test]
    fn test_z_to_gamma() {
        let z = c64(42.4, -19.6);
        let z0 = 50.0;
        let gamma = c64(-0.03565151895556114, -0.21968365553602814);
        let test = z_to_gamma(z, z0);

        comp_c64(&test, &gamma, F64Margin::default(), "z_to_gamma()", "gamma");
    }

    #[test]
    fn test_gamma_to_z() {
        let gamma = c64(0.2464, -0.8745);
        let z0 = 100.0;
        let z = c64(13.096841624374102, -131.24096072255193);
        let test = gamma_to_z(gamma, z0);

        comp_c64(&test, &z, F64Margin::default(), "gamma_to_z()", "z");
    }

    #[test]
    fn test_z_to_rc() {
        let z = c64(42.4, -19.6);
        let f = 275.0;
        let r = 51.46037735849057;
        let c = 5.198818862788317e-15;
        let test = z_to_rc(z, &UnitValBuilder::new().val_scaled(f, Scale::Giga).build());

        comp_f64(&test.0, &r, F64Margin::default(), "z_to_rc()", "r");
        comp_f64(&test.1, &c, F64Margin::default(), "z_to_rc()", "c");
    }

    #[test]
    fn test_rc_to_z() {
        let z = c64(42.4, -19.6);
        let f = 275.0;
        let r = 51.46037735849057;
        let c = 5.198818862788317e-15;
        let test = rc_to_z(
            r,
            c,
            &UnitValBuilder::new().val_scaled(f, Scale::Giga).build(),
        );

        comp_c64(&test, &z, F64Margin::default(), "rc_to_z()", "z");
    }
}
