use crate::myfloat::MyFloat;
use crate::point::{Point, Pt};
use crate::points::{Points, Pts};
use float_cmp::{F64Margin, approx_eq};
use ndarray::prelude::*;
use num::complex::Complex64;

pub fn comp_line(exemplar: &str, net: &str, test: &str) {
    let mut i: usize = 0;
    let mut exemplar_iter = exemplar.lines();
    let mut net_iter = net.lines();
    loop {
        let exemplar_line = exemplar_iter.next();
        let net_line = net_iter.next();
        if exemplar_line.is_none() && net_line.is_none() {
            break;
        } else if exemplar_line.is_none() || net_line.is_none() {
            panic!("test {} number of lines does not match >{}", test, i);
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

pub fn comp_points_c64(
    exemplar: &Points<Complex64>,
    calc: &Points<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    // azip!((index (i,j,k), &e in exemplar, &c in calc) {
    //     comp_c64(&e, &c, precision, test, format!("{},{},{}", i,j,k).to_owned().as_str());
    // });
    for i in 0..calc.len_of(Axis(0)) {
        for j in 0..calc.len_of(Axis(1)) {
            for k in 0..calc.len_of(Axis(2)) {
                comp_point(
                    &exemplar[(i, j, k)].re,
                    &calc[(i, j, k)].re,
                    precision,
                    test,
                    format!("({},{},{}).re", i, j, k),
                );
                comp_point(
                    &exemplar[(i, j, k)].im,
                    &calc[(i, j, k)].im,
                    precision,
                    test,
                    format!("({},{},{}).im", i, j, k),
                );
            }
        }
    }
}

pub fn comp_points_f64(
    exemplar: &Points<f64>,
    calc: &Points<f64>,
    precision: F64Margin,
    test: &str,
) {
    // azip!((index (i,j,k), &e in exemplar, &c in calc) {
    //     comp_f64(&e, &c, precision, test, format!("{},{},{}", i,j,k).to_owned().as_str());
    // });
    for i in 0..calc.len_of(Axis(0)) {
        for j in 0..calc.len_of(Axis(1)) {
            for k in 0..calc.len_of(Axis(2)) {
                comp_f64(
                    &exemplar[(i, j, k)],
                    &calc[(i, j, k)],
                    precision,
                    test,
                    format!("({},{},{})", i, j, k).as_str(),
                );
            }
        }
    }
}

pub fn comp_point_c64(
    exemplar: &Point<Complex64>,
    calc: &Point<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_point(
                &exemplar[(j, k)].re,
                &calc[(j, k)].re,
                precision,
                test,
                format!("({},{}).re", j, k),
            );
            comp_point(
                &exemplar[(j, k)].im,
                &calc[(j, k)].im,
                precision,
                test,
                format!("({},{}).im", j, k),
            );
        }
    }
}

pub fn comp_point_f64(exemplar: &Point<f64>, calc: &Point<f64>, precision: F64Margin, test: &str) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_point(
                &exemplar[(j, k)],
                &calc[(j, k)],
                precision,
                test,
                format!("({},{})", j, k),
            );
        }
    }
}

pub fn comp_mat_f64(exemplar: &Array2<f64>, calc: &Array2<f64>, precision: F64Margin, test: &str) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_point(
                &exemplar[(j, k)],
                &calc[(j, k)],
                precision,
                test,
                format!("({}, {})", j, k),
            );
        }
    }
}

pub fn comp_row_f64(exemplar: &Array1<f64>, calc: &Array1<f64>, precision: F64Margin, test: &str) {
    for k in 0..calc.len() {
        comp_point(
            exemplar.get(k).unwrap(),
            calc.get(k).unwrap(),
            precision,
            test,
            format!("({})", k),
        );
    }
}

pub fn comp_point_myfloat(
    exemplar: &Point<MyFloat>,
    calc: &Point<MyFloat>,
    precision: F64Margin,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_myfloat(
                &exemplar[(j, k)],
                &calc[(j, k)],
                precision,
                test,
                &format!("({},{})", j, k).to_owned(),
            );
        }
    }
}

pub fn comp_mat_myfloat(
    exemplar: &Array2<MyFloat>,
    calc: &Array2<MyFloat>,
    precision: F64Margin,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_myfloat(
                &exemplar[(j, k)],
                &calc[(j, k)],
                precision,
                test,
                &format!("({}, {})", j, k).to_owned(),
            );
        }
    }
}

pub fn comp_row_myfloat(
    exemplar: &Array1<MyFloat>,
    calc: &Array1<MyFloat>,
    precision: F64Margin,
    test: &str,
) {
    for k in 0..calc.len() {
        comp_myfloat(
            exemplar.get(k).unwrap(),
            calc.get(k).unwrap(),
            precision,
            test,
            &format!("({})", k).to_owned(),
        );
    }
}

pub fn comp_row_c64(
    exemplar: &Array1<Complex64>,
    calc: &Array1<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    for k in 0..calc.len() {
        comp_c64(
            exemplar.get(k).unwrap(),
            calc.get(k).unwrap(),
            precision,
            test,
            format!("({})", k).to_owned().as_str(),
        );
    }
}

pub fn comp_array_c64(
    exemplar: &Array1<Complex64>,
    calc: &Array1<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    azip!((index i, &e in exemplar, &c in calc) {
        comp_c64(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_array_f64(
    exemplar: &Array1<f64>,
    calc: &Array1<f64>,
    precision: F64Margin,
    test: &str,
) {
    azip!((index i, &e in exemplar, &c in calc) {
        comp_f64(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_array_myfloat(
    exemplar: &Array1<MyFloat>,
    calc: &Array1<MyFloat>,
    precision: F64Margin,
    test: &str,
) {
    azip!((index i, e in exemplar, c in calc) {
        comp_myfloat(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_vec_c64(exemplar: &[Complex64], calc: &[Complex64], precision: F64Margin, test: &str) {
    for k in 0..calc.len() {
        comp_c64(
            &exemplar[k],
            &calc[k],
            precision,
            test,
            &(format!("({})", k).to_owned()),
        );
    }
}

pub fn comp_vec_f64(exemplar: &[f64], calc: &[f64], precision: F64Margin, test: &str) {
    for k in 0..calc.len() {
        comp_f64(
            &exemplar[k],
            &calc[k],
            precision,
            test,
            &(format!("({})", k)).to_owned(),
        );
    }
}

pub fn comp_point(exemplar: &f64, calc: &f64, precision: F64Margin, test: &str, idx: String) {
    comp_f64(exemplar, calc, precision, test, &(idx.to_owned()));
}

pub fn comp_c64(
    exemplar: &Complex64,
    calc: &Complex64,
    precision: F64Margin,
    test: &str,
    idx: &str,
) {
    comp_f64(
        &(exemplar.re),
        &(calc.re),
        precision,
        test,
        &(idx.to_owned() + ".re"),
    );
    comp_f64(
        &(exemplar.im),
        &(calc.im),
        precision,
        test,
        &(idx.to_owned() + ".im"),
    );
}

pub fn comp_f64(exemplar: &f64, calc: &f64, precision: F64Margin, test: &str, idx: &str) {
    debug_assert!(
        // approx_eq!(f64, *calc, *exemplar, F64Margin::default()),
        approx_eq!(f64, *calc, *exemplar, precision),
        " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
        test,
        idx,
        exemplar,
        calc
    );
}

pub fn comp_myfloat(
    exemplar: &MyFloat,
    calc: &MyFloat,
    precision: F64Margin,
    test: &str,
    idx: &str,
) {
    debug_assert!(
        // approx_eq!(f64, *calc, *exemplar, F64Margin::default()),
        approx_eq!(f64, calc.to_f64(), exemplar.to_f64(), precision),
        " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
        test,
        idx,
        exemplar,
        calc
    );
}
