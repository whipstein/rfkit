use crate::myfloat::MyFloat;
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
    exemplar: ArrayView3<Complex64>,
    calc: ArrayView3<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    // azip!((index (i,j,k), &e in exemplar, &c in calc) {
    //     comp_c64(&e, &c, precision, test, format!("{},{},{}", i,j,k).to_owned().as_str());
    // });
    for i in 0..calc.len_of(Axis(0)) {
        for j in 0..calc.len_of(Axis(1)) {
            for k in 0..calc.len_of(Axis(2)) {
                comp_f64(
                    &exemplar[(i, j, k)].re,
                    &calc[(i, j, k)].re,
                    precision,
                    test,
                    format!("({},{},{}).re", i, j, k).to_owned().as_str(),
                );
                comp_f64(
                    &exemplar[(i, j, k)].im,
                    &calc[(i, j, k)].im,
                    precision,
                    test,
                    format!("({},{},{}).im", i, j, k).to_owned().as_str(),
                );
            }
        }
    }
}

pub fn comp_points_f64(
    exemplar: ArrayView3<f64>,
    calc: ArrayView3<f64>,
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
    exemplar: ArrayView2<Complex64>,
    calc: ArrayView2<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_f64(
                &exemplar[(j, k)].re,
                &calc[(j, k)].re,
                precision,
                test,
                format!("({},{}).re", j, k).to_owned().as_str(),
            );
            comp_f64(
                &exemplar[(j, k)].im,
                &calc[(j, k)].im,
                precision,
                test,
                format!("({},{}).im", j, k).to_owned().as_str(),
            );
        }
    }
}

pub fn comp_point_f64(
    exemplar: ArrayView2<f64>,
    calc: ArrayView2<f64>,
    precision: F64Margin,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_f64(
                &exemplar[(j, k)],
                &calc[(j, k)],
                precision,
                test,
                format!("({},{})", j, k).to_owned().as_str(),
            );
        }
    }
}

pub fn comp_mat_f64(
    exemplar: ArrayView2<f64>,
    calc: ArrayView2<f64>,
    precision: F64Margin,
    test: &str,
) {
    for j in 0..calc.nrows() {
        for k in 0..calc.ncols() {
            comp_f64(
                &exemplar[(j, k)],
                &calc[(j, k)],
                precision,
                test,
                format!("({}, {})", j, k).to_owned().as_str(),
            );
        }
    }
}

pub fn comp_row_f64(
    exemplar: ArrayView1<f64>,
    calc: ArrayView1<f64>,
    precision: F64Margin,
    test: &str,
) {
    for k in 0..calc.len() {
        comp_f64(
            exemplar.get(k).unwrap(),
            calc.get(k).unwrap(),
            precision,
            test,
            format!("({})", k).to_owned().as_str(),
        );
    }
}

pub fn comp_point_myfloat(
    exemplar: ArrayView2<MyFloat>,
    calc: ArrayView2<MyFloat>,
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
    exemplar: ArrayView2<MyFloat>,
    calc: ArrayView2<MyFloat>,
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
    exemplar: ArrayView1<MyFloat>,
    calc: ArrayView1<MyFloat>,
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
    exemplar: ArrayView1<Complex64>,
    calc: ArrayView1<Complex64>,
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
    exemplar: ArrayView1<Complex64>,
    calc: ArrayView1<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    azip!((index i, &e in exemplar, &c in calc) {
        comp_c64(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_array_f64(
    exemplar: ArrayView1<f64>,
    calc: ArrayView1<f64>,
    precision: F64Margin,
    test: &str,
) {
    azip!((index i, &e in exemplar, &c in calc) {
        comp_f64(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
    });
}

pub fn comp_array_myfloat(
    exemplar: ArrayView1<MyFloat>,
    calc: ArrayView1<MyFloat>,
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
