#![allow(dead_code)]
// #![feature(f128)]

pub mod circuit;
pub mod element;
pub mod error;
pub mod file;
pub mod frequency;
pub mod impedance;
pub mod math;
pub mod minimize;
pub mod mycomplex;
pub mod myfloat;
pub mod network;
pub mod parameter;
pub mod point;
pub mod points;
pub mod prelude;
pub mod scale;
pub mod unit;
pub mod util;

/// Create an **[`Point`]** with two dimensions.
///
/// ```
/// use ndarray::prelude::*;
/// use rfkit::prelude::*;
///
/// let a = point![f64, [1.0, 2.0],
///                 [3.0, 4.0]];
///
/// assert_eq!(a.shape(), (2, 2));
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
///
#[macro_export]
macro_rules! point {
    ($t: ty, $([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::point::Point::<$t>::new(ndarray::Array2::from(vec![$([$($x,)*],)*]))}
    };
}

/// Create an **[`Points`]** with three dimensions.
///
/// ```
/// use rfkit::prelude::*;
///
/// let a = points![f64, [[1.0, 2.0], [3.0, 4.0]],
///                 [[5.0, 6.0], [7.0, 8.0]]];
///
/// assert_eq!(a.shape(), (2, 2, 2));
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
///
#[macro_export]
macro_rules! points {
    ($t: ty, $([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::points::Points::<$t>::new(ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]))
    }};
}

#[cfg(test)]
mod lib_tests {
    use num::complex::{Complex64, c64};

    #[test]
    fn test_point() {
        let test = point![
            Complex64,
            [
                c64(-0.4285714285714286, 0.0),
                c64(1.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.4285714285714286, 0.0),
                c64(1.4285714285714284, 0.0),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
            ]
        ];

        assert_eq!(test[[0, 0]].re, -0.4285714285714286);
        assert_eq!(test[[0, 1]].re, 1.4285714285714284);
        assert_eq!(test[[1, 1]].re, 0.4285714285714284);
        assert_eq!(test[[2, 1]].re, 0.0);
    }

    #[test]
    fn test_points() {
        let test = points![
            Complex64,
            [
                [
                    c64(-0.4285714285714286, 0.0),
                    c64(1.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    c64(0.5714285714285714, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(1.4285714285714284, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5714285714285714, 0.0),
                    c64(0.4285714285714284, 0.0),
                ]
            ],
            [
                [
                    c64(-0.4285714285714286, 0.0),
                    c64(1.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    c64(0.5714285714285714, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(1.4285714285714284, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5714285714285714, 0.0),
                    c64(0.4285714285714284, 0.0),
                ]
            ],
            [
                [
                    c64(-0.4285714285714286, 0.0),
                    c64(1.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    c64(0.5714285714285714, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(1.4285714285714284, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5714285714285714, 0.0),
                    c64(0.4285714285714284, 0.0),
                ]
            ]
        ];

        assert_eq!(test[[0, 0, 0]].re, -0.4285714285714286);
        assert_eq!(test[[0, 0, 1]].re, 1.4285714285714284);
        assert_eq!(test[[1, 1, 1]].re, 0.4285714285714284);
        assert_eq!(test[[2, 2, 1]].re, 0.0);

        let test = points![
            f64,
            [
                [-0.4285714285714286, 1.4285714285714284, 0.0, 0.0],
                [0.5714285714285714, 0.4285714285714284, 0.0, 0.0],
                [0.0, 0.0, -0.4285714285714286, 1.4285714285714284],
                [0.0, 0.0, 0.5714285714285714, 0.4285714285714284]
            ],
            [
                [-0.4285714285714286, 1.4285714285714284, 0.0, 0.0],
                [0.5714285714285714, 0.4285714285714284, 0.0, 0.0],
                [0.0, 0.0, -0.4285714285714286, 1.4285714285714284],
                [0.0, 0.0, 0.5714285714285714, 0.4285714285714284]
            ],
            [
                [-0.4285714285714286, 1.4285714285714284, 0.0, 0.0],
                [0.5714285714285714, 0.4285714285714284, 0.0, 0.0],
                [0.0, 0.0, -0.4285714285714286, 1.4285714285714284],
                [0.0, 0.0, 0.5714285714285714, 0.4285714285714284]
            ]
        ];

        assert_eq!(test[[0, 0, 0]], -0.4285714285714286);
        assert_eq!(test[[0, 0, 1]], 1.4285714285714284);
        assert_eq!(test[[1, 1, 1]], 0.4285714285714284);
        assert_eq!(test[[2, 2, 1]], 0.0);
    }
}
