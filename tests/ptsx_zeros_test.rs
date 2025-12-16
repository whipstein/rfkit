use ndarray::prelude::*;
use num::complex::Complex64;
use rfkit::pts::{Points, Pts};

// #[test]
// fn test_zeros_1d() {
//     let zeros: Points<f64, Ix1> = Points::zeros(10);
//     assert_eq!(zeros.0.shape(), &[10]);
//     assert!(zeros.0.iter().all(|&x| x == 0.0));
// }

// #[test]
// fn test_zeros_2d() {
//     let zeros: Points<f64, Ix2> = Points::zeros((3, 4));
//     assert_eq!(zeros.0.shape(), &[3, 4]);
//     assert!(zeros.0.iter().all(|&x| x == 0.0));
// }

#[test]
fn test_zeros_3d() {
    let zeros: Points<f64, Ix3> = Points::zeros((2, 3, 4));
    assert_eq!(zeros.0.shape(), &[2, 3, 4]);
    assert!(zeros.0.iter().all(|&x| x == 0.0));
}

// #[test]
// fn test_zeros_4d() {
//     let zeros: Points<f64, Ix4> = Points::zeros((2, 2, 2, 2));
//     assert_eq!(zeros.0.shape(), &[2, 2, 2, 2]);
//     assert!(zeros.0.iter().all(|&x| x == 0.0));
// }

// #[test]
// fn test_zeros_5d() {
//     let zeros: Points<f64, Ix5> = Points::zeros((2, 2, 2, 2, 2));
//     assert_eq!(zeros.0.shape(), &[2, 2, 2, 2, 2]);
//     assert!(zeros.0.iter().all(|&x| x == 0.0));
// }

// #[test]
// fn test_zeros_complex_1d() {
//     let zeros: Points<Complex64, Ix1> = Points::zeros(5);
//     assert_eq!(zeros.0.shape(), &[5]);
//     assert!(zeros.0.iter().all(|x| x.re == 0.0 && x.im == 0.0));
// }

// #[test]
// fn test_zeros_complex_2d() {
//     let zeros: Points<Complex64, Ix2> = Points::zeros((3, 3));
//     assert_eq!(zeros.0.shape(), &[3, 3]);
//     assert!(zeros.0.iter().all(|x| x.re == 0.0 && x.im == 0.0));
// }

#[test]
fn test_zeros_complex_3d() {
    let zeros: Points<Complex64, Ix3> = Points::zeros((2, 3, 4));
    assert_eq!(zeros.0.shape(), &[2, 3, 4]);
    assert!(zeros.0.iter().all(|x| x.re == 0.0 && x.im == 0.0));
}

// #[test]
// fn test_zeros_i32() {
//     let zeros: Points<i32, Ix2> = Points::zeros((5, 5));
//     assert_eq!(zeros.0.shape(), &[5, 5]);
//     assert!(zeros.0.iter().all(|&x| x == 0));
// }

// #[test]
// fn test_zeros_u64() {
//     let zeros: Points<u64, Ix3> = Points::zeros((2, 2, 2));
//     assert_eq!(zeros.0.shape(), &[2, 2, 2]);
//     assert!(zeros.0.iter().all(|&x| x == 0));
// }
