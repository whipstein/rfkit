use faer::complex_native::c64;
use faer::row::Row;
use std::f64::consts::PI;
use crate::network::{Point, Pointf64};

pub fn sqrt_phase_unwrap(pts: Vec<c64>) -> Vec<c64> {
    let mut out: Vec<c64> = vec![];
    let mut phi: Vec<f64> = vec![];
    for pt in pts.iter() {
        phi.push(pt.arg());
    }

    for i in 1..phi.len() {
        let diff = phi[i] - phi[i - 1];
        if diff > PI {
            for j in i..phi.len() {
                phi[j] -= 2.0*PI;
            }
        } else if diff > PI {
            for j in i..phi.len() {
                phi[j] += 2.0*PI;
            }
        }
    }

    for i in 0..pts.len() {
        out.push(pts[i].abs().sqrt() * c64::new(0.0, 0.5 * phi[i]).exp());
    }

    out
}

pub fn from_polar_deg(r: f64, theta: f64) -> c64 {
    c64::from_polar(r, theta*180.0/PI)
}

pub fn comp_points_c64(exemplar: &Vec<Point>, calc: &Vec<Point>, test: &str) {
    for i in 0..calc.len() {
        for j in 0..calc[i].nrows() {
            for k in 0..calc[i].ncols() {
                comp_point_c64(&exemplar[i][(j, k)], &calc[i][(j, k)], test, format!("{},{},{}", i, j, k));
            }
        }
    }
}

pub fn comp_points_f64(
    exemplar: &Vec<Pointf64>,
    calc: &Vec<Pointf64>,
    test: &str,
) {
    for i in 0..calc.len() {
        for j in 0..calc[i].nrows() {
            for k in 0..calc[i].ncols() {
                comp_point(&exemplar[i].get(j, k), &calc[i].get(j, k), test, format!("({},{},{})", i, j, k));
            }
        }
    }
}

pub fn comp_point_c64(exemplar: &c64, calc: &c64, test: &str, idx: String) {
    comp_point(&exemplar.re, &calc.re, test, String::from(vec!["re(", idx.as_str(), ")"].join("")));
    comp_point(&exemplar.im, &calc.im, test, String::from(vec!["re(", idx.as_str(), ")"].join("")));
}

pub fn comp_point_f64(
    exemplar: &Pointf64,
    calc: &Pointf64,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_point(exemplar.get(j, k), calc.get(j, k), test, format!("({},{})", j, k));
        }
    }
}

pub fn comp_row_f64(
    exemplar: &Row<f64>,
    calc: &Row<f64>,
    test: &str,
) {
    for k in 0..calc.ncols() {
        comp_point(exemplar.get(k), calc.get(k), test, format!("({})", k));
    }
}

pub fn comp_point(exemplar: &f64, calc: &f64, test: &str, idx: String) {
    if exemplar == calc {
        return ();
    }

    let base: f64 = 10.0;
    let eps = base.powi(-10);

    debug_assert!((calc - exemplar).abs() < (eps * exemplar).abs(), " Failed test {} at location {}\n  exemplar: {}\n      calc: {}", test, idx, exemplar, calc);
}
