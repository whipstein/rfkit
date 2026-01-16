use crate::{
    num::{MyComplex, MyFloat, RFNum},
    pts::{Points1, Points2, Points3},
};
use float_cmp::{F64Margin, approx_eq};
use ndarray::prelude::*;
use num::complex::Complex64;

pub trait ApproxCompare {
    fn approx_compare(&self, exemplar: &Self, precision: F64Margin, test: &str, idx: &str);
}

impl ApproxCompare for f64 {
    fn approx_compare(&self, exemplar: &Self, precision: F64Margin, test: &str, idx: &str) {
        debug_assert!(
            approx_eq!(f64, *self, *exemplar, precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
}

impl ApproxCompare for MyFloat {
    fn approx_compare(&self, exemplar: &Self, precision: F64Margin, test: &str, idx: &str) {
        debug_assert!(
            approx_eq!(f64, self.to_f64(), exemplar.to_f64(), precision),
            " Failed test {} at location {}\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar,
            self
        );
    }
}

impl ApproxCompare for Complex64 {
    fn approx_compare(&self, exemplar: &Self, precision: F64Margin, test: &str, idx: &str) {
        debug_assert!(
            approx_eq!(f64, self.re, exemplar.re, precision),
            " Failed test {} at location {}.re\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar.re,
            self.re
        );
        debug_assert!(
            approx_eq!(f64, self.im, exemplar.im, precision),
            " Failed test {} at location {}.im\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar.im,
            self.im
        );
    }
}

impl ApproxCompare for MyComplex {
    fn approx_compare(&self, exemplar: &Self, precision: F64Margin, test: &str, idx: &str) {
        debug_assert!(
            approx_eq!(f64, self.re().to_f64(), exemplar.re().to_f64(), precision),
            " Failed test {} at location {}.re\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar.re(),
            self.re()
        );
        debug_assert!(
            approx_eq!(f64, self.im().to_f64(), exemplar.im().to_f64(), precision),
            " Failed test {} at location {}.im\n  exemplar: {}\n      calc: {}",
            test,
            idx,
            exemplar.im(),
            self.im()
        );
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

pub fn comp_num<T>(exemplar: &T, calc: &T, precision: F64Margin, test: &str, idx: &str)
where
    T: ApproxCompare + RFNum,
{
    calc.approx_compare(exemplar, precision, test, idx);
}

pub fn comp_pts_ix1<T>(exemplar: &Points1<T>, calc: &Points1<T>, precision: F64Margin, test: &str)
where
    T: ApproxCompare + RFNum,
{
    azip!((index k, e in exemplar.inner(), c in calc.inner()) {
        c.approx_compare(e, precision, test, format!("{}", k).to_owned().as_str());
    });
}

pub fn comp_pts_ix2<T>(exemplar: &Points2<T>, calc: &Points2<T>, precision: F64Margin, test: &str)
where
    T: ApproxCompare + RFNum,
{
    azip!((index (j, k), e in exemplar.inner(), c in calc.inner()) {
        c.approx_compare(e, precision, test, format!("{},{}", j, k).to_owned().as_str());
    });
}

pub fn comp_pts_ix3<T>(exemplar: &Points3<T>, calc: &Points3<T>, precision: F64Margin, test: &str)
where
    T: ApproxCompare + RFNum,
{
    azip!((index (i, j, k), e in exemplar.inner(), c in calc.inner()) {
        c.approx_compare(e, precision, test, format!("{},{},{}", i, j, k).to_owned().as_str());
    });
}

// pub fn comp_pts_ix3_c64(
//     exemplar: Points3<Complex64>,
//     calc: Points3<Complex64>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index (i,j,k), &e in exemplar.inner(), &c in calc.inner()) {
//         comp_c64(&e, &c, precision, test, format!("{},{},{}", i,j,k).to_owned().as_str());
//     });
//     // for i in 0..calc.len_of(Axis(0)) {
//     //     for j in 0..calc.len_of(Axis(1)) {
//     //         for k in 0..calc.len_of(Axis(2)) {
//     //             comp_f64(
//     //                 &exemplar[(i, j, k)].re,
//     //                 &calc[(i, j, k)].re,
//     //                 precision,
//     //                 test,
//     //                 format!("({},{},{}).re", i, j, k).to_owned().as_str(),
//     //             );
//     //             comp_f64(
//     //                 &exemplar[(i, j, k)].im,
//     //                 &calc[(i, j, k)].im,
//     //                 precision,
//     //                 test,
//     //                 format!("({},{},{}).im", i, j, k).to_owned().as_str(),
//     //             );
//     //         }
//     //     }
//     // }
// }

// pub fn comp_pts_ix3_f64(
//     exemplar: Points3<f64>,
//     calc: Points3<f64>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index (i,j,k), &e in exemplar.inner(), &c in calc.inner()) {
//         comp_f64(&e, &c, precision, test, format!("{},{},{}", i,j,k).to_owned().as_str());
//     });
//     // for i in 0..calc.len_of(Axis(0)) {
//     //     for j in 0..calc.len_of(Axis(1)) {
//     //         for k in 0..calc.len_of(Axis(2)) {
//     //             comp_f64(
//     //                 &exemplar[(i, j, k)],
//     //                 &calc[(i, j, k)],
//     //                 precision,
//     //                 test,
//     //                 format!("({},{},{})", i, j, k).as_str(),
//     //             );
//     //         }
//     //     }
//     // }
// }

// pub fn comp_pts_ix2_c64(
//     exemplar: Points2<Complex64>,
//     calc: Points2<Complex64>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index (j,k), &e in exemplar.inner(), &c in calc.inner()) {
//         comp_c64(&e, &c, precision, test, format!("{},{}", j,k).to_owned().as_str());
//     });
//     // for j in 0..calc.nrows() {
//     //     for k in 0..calc.ncols() {
//     //         comp_f64(
//     //             &exemplar[(j, k)].re,
//     //             &calc[(j, k)].re,
//     //             precision,
//     //             test,
//     //             format!("({},{}).re", j, k).to_owned().as_str(),
//     //         );
//     //         comp_f64(
//     //             &exemplar[(j, k)].im,
//     //             &calc[(j, k)].im,
//     //             precision,
//     //             test,
//     //             format!("({},{}).im", j, k).to_owned().as_str(),
//     //         );
//     //     }
//     // }
// }

// pub fn comp_pts_ix2_f64(
//     exemplar: Points2<f64>,
//     calc: Points2<f64>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index (j,k), &e in exemplar.inner(), &c in calc.inner()) {
//         comp_f64(&e, &c, precision, test, format!("{},{}", j,k).to_owned().as_str());
//     });
//     // for j in 0..calc.nrows() {
//     //     for k in 0..calc.ncols() {
//     //         comp_f64(
//     //             &exemplar[(j, k)],
//     //             &calc[(j, k)],
//     //             precision,
//     //             test,
//     //             format!("({},{})", j, k).to_owned().as_str(),
//     //         );
//     //     }
//     // }
// }

// pub fn comp_pts_ix1_f64(
//     exemplar: Points1<f64>,
//     calc: Points1<f64>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index k, &e in exemplar.inner(), &c in calc.inner()) {
//         comp_f64(&e, &c, precision, test, format!("{}", k).to_owned().as_str());
//     });
//     // for k in 0..calc.len() {
//     //     comp_f64(
//     //         exemplar.get(k).unwrap(),
//     //         calc.get(k).unwrap(),
//     //         precision,
//     //         test,
//     //         format!("({})", k).to_owned().as_str(),
//     //     );
//     // }
// }

// pub fn comp_point_myfloat(
//     exemplar: ArrayView2<MyFloat>,
//     calc: ArrayView2<MyFloat>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     for j in 0..calc.nrows() {
//         for k in 0..calc.ncols() {
//             comp_myfloat(
//                 &exemplar[(j, k)],
//                 &calc[(j, k)],
//                 precision,
//                 test,
//                 &format!("({},{})", j, k).to_owned(),
//             );
//         }
//     }
// }

// pub fn comp_pts_ix2_myfloat(
//     exemplar: Points2<MyFloat>,
//     calc: Points2<MyFloat>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     for j in 0..calc.nrows() {
//         for k in 0..calc.ncols() {
//             comp_myfloat(
//                 &exemplar[(j, k)],
//                 &calc[(j, k)],
//                 precision,
//                 test,
//                 &format!("({}, {})", j, k).to_owned(),
//             );
//         }
//     }
// }

// pub fn comp_pts_ix1_myfloat(
//     exemplar: Points1<MyFloat>,
//     calc: Points1<MyFloat>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index k, e in exemplar.inner(), c in calc.inner()) {
//         comp_myfloat(e, c, precision, test, format!("{}", k).to_owned().as_str());
//     });
// }

// pub fn comp_pts_ix1_c64(
//     exemplar: Points1<Complex64>,
//     calc: Points1<Complex64>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index k, &e in exemplar.inner(), &c in calc.inner()) {
//         comp_c64(&e, &c, precision, test, format!("{}", k).to_owned().as_str());
//     });
//     // for k in 0..calc.len() {
//     //     comp_c64(
//     //         exemplar.get(k).unwrap(),
//     //         calc.get(k).unwrap(),
//     //         precision,
//     //         test,
//     //         format!("({})", k).to_owned().as_str(),
//     //     );
//     // }
// }

pub fn comp_array_c64(
    exemplar: ArrayView1<Complex64>,
    calc: ArrayView1<Complex64>,
    precision: F64Margin,
    test: &str,
) {
    azip!((index i, &e in exemplar, &c in calc) {
        comp_num(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
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

// pub fn comp_array_myfloat(
//     exemplar: ArrayView1<MyFloat>,
//     calc: ArrayView1<MyFloat>,
//     precision: F64Margin,
//     test: &str,
// ) {
//     azip!((index i, e in exemplar, c in calc) {
//         comp_myfloat(&e, &c, precision, test, format!("({})", i).to_owned().as_str());
//     });
// }

pub fn comp_vec_c64(exemplar: &[Complex64], calc: &[Complex64], precision: F64Margin, test: &str) {
    for k in 0..calc.len() {
        comp_num(
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

// pub fn comp_c64(
//     exemplar: &Complex64,
//     calc: &Complex64,
//     precision: F64Margin,
//     test: &str,
//     idx: &str,
// ) {
//     comp_f64(
//         &(exemplar.re),
//         &(calc.re),
//         precision,
//         test,
//         &(idx.to_owned() + ".re"),
//     );
//     comp_f64(
//         &(exemplar.im),
//         &(calc.im),
//         precision,
//         test,
//         &(idx.to_owned() + ".im"),
//     );
// }

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
