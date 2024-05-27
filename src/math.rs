use faer::complex_native::c64;
use std::f64::consts::PI;

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
