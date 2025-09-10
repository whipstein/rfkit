use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use ndarray::prelude::*;
use num_traits::{One, Zero};

/// Additional error types for matrix inversion
#[derive(Debug, PartialEq)]
pub enum InversionError {
    NotSquare(String),
    Singular(String),
    DimensionMismatch(String),
}

impl std::fmt::Display for InversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InversionError::NotSquare(msg) => write!(f, "Matrix is not square: {}", msg),
            InversionError::Singular(msg) => write!(f, "Matrix is singular: {}", msg),
            InversionError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
        }
    }
}

impl std::error::Error for InversionError {}

impl MyComplex {
    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    ///
    /// This function computes A^(-1) such that A * A^(-1) = I, where I is the identity matrix.
    ///
    /// # Arguments
    /// * `matrix` - A square matrix to invert
    ///
    /// # Returns
    /// * `Ok(Array2<MyComplex>)` - The inverted matrix
    /// * `Err(InversionError)` - If the matrix is not square, singular, or other errors
    ///
    /// # Examples
    /// ```rust
    /// use ndarray::Array2;
    ///
    /// // Create a 2x2 matrix
    /// let matrix = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(1.0, 0.0), MyComplex::from_f64(2.0, 0.0),
    ///     MyComplex::from_f64(3.0, 0.0), MyComplex::from_f64(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let inv_matrix = MyComplex::matrix_inverse(&matrix.view())?;
    /// ```
    pub fn matrix_inverse(
        matrix: &ArrayView2<MyComplex>,
    ) -> Result<Array2<MyComplex>, InversionError> {
        let (rows, cols) = matrix.dim();

        // Check if matrix is square
        if rows != cols {
            return Err(InversionError::NotSquare(format!(
                "Matrix dimensions are {}x{}, expected square matrix",
                rows, cols
            )));
        }

        let n = rows;

        // Create augmented matrix [A | I]
        let mut augmented = Array2::zeros((n, 2 * n));

        // Copy original matrix to left half
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]].clone();
            }
        }

        // Create identity matrix in right half
        for i in 0..n {
            augmented[[i, i + n]] = MyComplex::one();
        }

        // Perform Gauss-Jordan elimination with partial pivoting
        for i in 0..n {
            // Find pivot row (row with largest absolute value in column i)
            let mut pivot_row = i;
            let mut max_abs = augmented[[i, i]].abs();

            for k in (i + 1)..n {
                let abs_val = augmented[[k, i]].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                    pivot_row = k;
                }
            }

            // Check for singularity
            if max_abs < MyFloat::new(1e-12) {
                return Err(InversionError::Singular(format!(
                    "Matrix is singular or nearly singular at pivot {}",
                    i
                )));
            }

            // Swap rows if necessary
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]].clone();
                    augmented[[i, j]] = augmented[[pivot_row, j]].clone();
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Scale pivot row to make diagonal element 1
            let pivot = augmented[[i, i]].clone();
            for j in 0..(2 * n) {
                augmented[[i, j]] /= &pivot;
            }

            // Eliminate column i in all other rows
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]].clone();
                    for j in 0..(2 * n) {
                        let temp = &augmented[[i, j]] * &factor;
                        augmented[[k, j]] -= &temp;
                    }
                }
            }
        }

        // Extract the inverse matrix from the right half
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]].clone();
            }
        }

        Ok(inverse)
    }

    /// Compute the determinant of a square matrix using LU decomposition
    ///
    /// # Arguments
    /// * `matrix` - A square matrix
    ///
    /// # Returns
    /// * `Ok(MyComplex)` - The determinant
    /// * `Err(InversionError)` - If the matrix is not square
    pub fn matrix_determinant(matrix: &ArrayView2<MyComplex>) -> Result<MyComplex, InversionError> {
        let (rows, cols) = matrix.dim();

        if rows != cols {
            return Err(InversionError::NotSquare(format!(
                "Matrix dimensions are {}x{}, expected square matrix",
                rows, cols
            )));
        }

        let n = rows;

        // Create a copy for LU decomposition
        let mut lu = matrix.to_owned();
        let mut det = MyComplex::one();
        let mut sign = 1;

        // Perform LU decomposition with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            let mut max_abs = lu[[i, i]].abs();

            for k in (i + 1)..n {
                let abs_val = lu[[k, i]].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                    pivot_row = k;
                }
            }

            // Swap rows if necessary
            if pivot_row != i {
                for j in 0..n {
                    let temp = lu[[i, j]].clone();
                    lu[[i, j]] = lu[[pivot_row, j]].clone();
                    lu[[pivot_row, j]] = temp;
                }
                sign *= -1; // Row swap changes sign of determinant
            }

            // Check for zero pivot (singular matrix)
            if lu[[i, i]].abs() < MyFloat::new(1e-12) {
                return Ok(MyComplex::zero());
            }

            // Update determinant with diagonal element
            det *= &lu[[i, i]];

            // Eliminate below pivot
            for k in (i + 1)..n {
                let factor = &lu[[k, i]] / &lu[[i, i]];
                for j in i..n {
                    let temp = &lu[[i, j]] * &factor;
                    lu[[k, j]] -= &temp;
                }
            }
        }

        if sign == -1 {
            det = -det;
        }

        Ok(det)
    }

    /// Solve the linear system Ax = b using LU decomposition
    ///
    /// # Arguments
    /// * `a` - Coefficient matrix A (n×n)
    /// * `b` - Right-hand side vector b (n×1)
    ///
    /// # Returns
    /// * `Ok(Array2<MyComplex>)` - Solution vector x
    /// * `Err(InversionError)` - If the system cannot be solved
    pub fn solve_linear_system(
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
    ) -> Result<Array2<MyComplex>, InversionError> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        if a_rows != a_cols {
            return Err(InversionError::NotSquare(format!(
                "Coefficient matrix is {}x{}, expected square matrix",
                a_rows, a_cols
            )));
        }

        if b_cols != 1 {
            return Err(InversionError::DimensionMismatch(format!(
                "Right-hand side must be a column vector, got {}x{}",
                b_rows, b_cols
            )));
        }

        if a_rows != b_rows {
            return Err(InversionError::DimensionMismatch(format!(
                "Matrix A is {}x{} but vector b has {} rows",
                a_rows, a_cols, b_rows
            )));
        }

        let n = a_rows;

        // Create augmented matrix [A | b]
        let mut augmented = Array2::zeros((n, n + 1));

        // Copy A
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = a[[i, j]].clone();
            }
        }

        // Copy b
        for i in 0..n {
            augmented[[i, n]] = b[[i, 0]].clone();
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            let mut max_abs = augmented[[i, i]].abs();

            for k in (i + 1)..n {
                let abs_val = augmented[[k, i]].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                    pivot_row = k;
                }
            }

            // Check for singularity
            if max_abs < MyFloat::new(1e-12) {
                return Err(InversionError::Singular(format!(
                    "Matrix is singular at pivot {}",
                    i
                )));
            }

            // Swap rows if necessary
            if pivot_row != i {
                for j in 0..(n + 1) {
                    let temp = augmented[[i, j]].clone();
                    augmented[[i, j]] = augmented[[pivot_row, j]].clone();
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Eliminate below pivot
            for k in (i + 1)..n {
                let factor = &augmented[[k, i]] / &augmented[[i, i]];
                for j in i..(n + 1) {
                    let temp = &augmented[[i, j]] * &factor;
                    augmented[[k, j]] -= &temp;
                }
            }
        }

        // Back substitution
        let mut x = Array2::zeros((n, 1));
        for i in (0..n).rev() {
            let mut sum = augmented[[i, n]].clone();
            for j in (i + 1)..n {
                sum -= &augmented[[i, j]] * &x[[j, 0]];
            }
            x[[i, 0]] = sum / &augmented[[i, i]];
        }

        Ok(x)
    }

    /// Check if a matrix is approximately equal to another matrix within a tolerance
    ///
    /// # Arguments
    /// * `a` - First matrix
    /// * `b` - Second matrix  
    /// * `tol` - Tolerance for comparison
    ///
    /// # Returns
    /// * `true` if matrices are approximately equal, `false` otherwise
    pub fn matrix_approx_eq(
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        tol: f64,
    ) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        let (rows, cols) = a.dim();
        let tolerance = MyFloat::new(tol);

        for i in 0..rows {
            for j in 0..cols {
                let diff = &a[[i, j]] - &b[[i, j]];
                if diff.abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::*;
    use ndarray::Array2;

    #[test]
    fn test_2x2_matrix_inversion() {
        // Test inverting a simple 2x2 matrix
        // A = [1 2]  =>  A^(-1) = [-2  1]
        //     [3 4]               [1.5 -0.5]

        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
            ],
        )
        .unwrap();

        let inverse = MyComplex::matrix_inverse(&matrix.view()).unwrap();

        // Check specific values
        approx_eq!(
            f64,
            inverse[[0, 0]].real().to_f64(),
            -2.0,
            F64Margin::default()
        );
        approx_eq!(
            f64,
            inverse[[0, 1]].real().to_f64(),
            1.0,
            F64Margin::default()
        );
        approx_eq!(
            f64,
            inverse[[1, 0]].real().to_f64(),
            1.5,
            F64Margin::default()
        );
        approx_eq!(
            f64,
            inverse[[1, 1]].real().to_f64(),
            -0.5,
            F64Margin::default()
        );

        // Verify A * A^(-1) = I
        let product = MyComplex::matrix_multiply(&matrix.view(), &inverse.view());
        let identity = MyComplex::identity_matrix(2);

        assert!(MyComplex::matrix_approx_eq(
            &product.view(),
            &identity.view(),
            1e-10
        ));
    }

    #[test]
    fn test_complex_matrix_inversion() {
        // Test with complex numbers
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 1.0),
                MyComplex::from_f64(0.0, 1.0),
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(1.0, 1.0),
            ],
        )
        .unwrap();

        let inverse = MyComplex::matrix_inverse(&matrix.view()).unwrap();

        // Verify A * A^(-1) = I
        let product = MyComplex::matrix_multiply(&matrix.view(), &inverse.view());
        let identity = MyComplex::identity_matrix(2);

        assert!(MyComplex::matrix_approx_eq(
            &product.view(),
            &identity.view(),
            1e-10
        ));
    }

    #[test]
    fn test_3x3_matrix_inversion() {
        // Test with a 3x3 matrix
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(-1.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(-1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(-1.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(-1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
            ],
        )
        .unwrap();

        let inverse = MyComplex::matrix_inverse(&matrix.view()).unwrap();

        // Verify A * A^(-1) = I
        let product = MyComplex::matrix_multiply(&matrix.view(), &inverse.view());
        let identity = MyComplex::identity_matrix(3);

        assert!(MyComplex::matrix_approx_eq(
            &product.view(),
            &identity.view(),
            1e-10
        ));
    }

    #[test]
    fn test_singular_matrix() {
        // Test with a singular matrix (determinant = 0)
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(4.0, 0.0), // Second row is 2x first row
            ],
        )
        .unwrap();

        let result = MyComplex::matrix_inverse(&matrix.view());
        assert!(result.is_err());

        match result {
            Err(InversionError::Singular(_)) => {}
            _ => panic!("Expected Singular error"),
        }
    }

    #[test]
    fn test_non_square_matrix() {
        // Test with a non-square matrix
        let matrix = Array2::from_shape_vec(
            (2, 3),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
                MyComplex::from_f64(5.0, 0.0),
                MyComplex::from_f64(6.0, 0.0),
            ],
        )
        .unwrap();

        let result = MyComplex::matrix_inverse(&matrix.view());
        assert!(result.is_err());

        match result {
            Err(InversionError::NotSquare(_)) => {}
            _ => panic!("Expected NotSquare error"),
        }
    }

    #[test]
    fn test_determinant() {
        // Test 2x2 determinant
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
            ],
        )
        .unwrap();

        let det = MyComplex::matrix_determinant(&matrix.view()).unwrap();
        // det = 1*4 - 2*3 = -2
        approx_eq!(f64, det.real().to_f64(), -2.0, F64Margin::default());
        approx_eq!(f64, det.imag().to_f64(), 0.0, F64Margin::default());

        // Test 3x3 identity matrix
        let identity = MyComplex::identity_matrix(3);
        let det_identity = MyComplex::matrix_determinant(&identity.view()).unwrap();
        approx_eq!(f64, det_identity.real().to_f64(), 1.0, F64Margin::default());

        // Test singular matrix (determinant should be 0)
        let singular = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
            ],
        )
        .unwrap();

        let det_singular = MyComplex::matrix_determinant(&singular.view()).unwrap();
        assert!(det_singular.abs().to_f64() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve Ax = b where:
        // A = [2 -1]  b = [1]  =>  x = [1]
        //     [-1 2]      [2]       [1.5]

        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(-1.0, 0.0),
                MyComplex::from_f64(-1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
            ],
        )
        .unwrap();

        let b = Array2::from_shape_vec(
            (2, 1),
            vec![MyComplex::from_f64(1.0, 0.0), MyComplex::from_f64(2.0, 0.0)],
        )
        .unwrap();

        let x = MyComplex::solve_linear_system(&a.view(), &b.view()).unwrap();

        // Verify the solution
        approx_eq!(
            f64,
            x[[0, 0]].real().to_f64(),
            4.0 / 3.0,
            F64Margin::default()
        );
        approx_eq!(
            f64,
            x[[1, 0]].real().to_f64(),
            5.0 / 3.0,
            F64Margin::default()
        );

        // Verify Ax = b
        let ax = MyComplex::matrix_multiply(&a.view(), &x.view());
        assert!(MyComplex::matrix_approx_eq(&ax.view(), &b.view(), 1e-10));
    }

    #[test]
    fn test_identity_matrix_inversion() {
        // Identity matrix should be its own inverse
        let identity = MyComplex::identity_matrix(4);
        let inverse = MyComplex::matrix_inverse(&identity.view()).unwrap();

        assert!(MyComplex::matrix_approx_eq(
            &identity.view(),
            &inverse.view(),
            1e-10
        ));
    }

    #[test]
    fn test_matrix_approx_eq() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
            ],
        )
        .unwrap();

        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.000001, 0.0),
                MyComplex::from_f64(2.000001, 0.0),
                MyComplex::from_f64(3.000001, 0.0),
                MyComplex::from_f64(4.000001, 0.0),
            ],
        )
        .unwrap();

        assert!(MyComplex::matrix_approx_eq(&a.view(), &b.view(), 1e-5));
        assert!(!MyComplex::matrix_approx_eq(&a.view(), &b.view(), 1e-7));
    }
}
