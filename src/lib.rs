#![allow(dead_code)]
pub mod circuit;
pub mod elements;
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
/// use rfkit_base::point;
/// let a2 = array![[1, 2],
///                 [3, 4]];
///
/// assert_eq!(a.shape(), &[2, 2]);
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
///
#[macro_export]
macro_rules! point {
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::point::Point::new(ndarray::Array2::from(vec![$([$($x,)*],)*]))
    }};
}

/// Create an **[`Points`]** with three dimensions.
///
/// ```
/// use rfkit_base::points;
/// let a = points![[[1, 2], [3, 4]],
///                 [[5, 6], [7, 8]]];
///
/// assert_eq!(a.shape(), &[2, 2, 2]);
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
///
#[macro_export]
macro_rules! points {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::points::Points::new(ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]))
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::{Complex64, c64};

    #[test]
    fn test_point() {
        let test = point![
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
    }
}
