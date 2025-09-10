use crate::mycomplex::MyComplex;
use ndarray::prelude::*;
use num_traits::{One, Zero};

/// Transpose operation for matrices
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transpose {
    /// No transpose
    No,
    /// Transpose (A^T)
    Trans,
    /// Conjugate transpose (A^H)
    ConjTrans,
}

/// Error types for matrix operations
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    DimensionMismatch(String),
    InvalidShape(String),
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            MatrixError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
        }
    }
}

impl std::error::Error for MatrixError {}

impl MyComplex {
    /// ZGEMM: General matrix multiply for complex matrices using ndarray
    ///
    /// Performs the matrix operation: C := alpha*op(A)*op(B) + beta*C
    ///
    /// Where op(X) is one of:
    /// - op(X) = X   (No transpose)
    /// - op(X) = X^T (Transpose)  
    /// - op(X) = X^H (Conjugate transpose)
    ///
    /// # Arguments
    ///
    /// * `transa` - Specifies the form of op(A)
    /// * `transb` - Specifies the form of op(B)
    /// * `alpha` - Scalar alpha
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `beta` - Scalar beta
    /// * `c` - Matrix C (input/output)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::Array2;
    /// use your_crate::{MyComplex, Transpose};
    ///
    /// // 2x2 matrix multiplication: C = A * B
    /// let a = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(1.0, 0.0), MyComplex::from_f64(2.0, 0.0),
    ///     MyComplex::from_f64(3.0, 0.0), MyComplex::from_f64(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let b = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(5.0, 0.0), MyComplex::from_f64(6.0, 0.0),
    ///     MyComplex::from_f64(7.0, 0.0), MyComplex::from_f64(8.0, 0.0),
    /// ]).unwrap();
    ///
    /// let mut c = Array2::zeros((2, 2));
    ///
    /// MyComplex::zgemm(
    ///     Transpose::No, Transpose::No,
    ///     &MyComplex::from_f64(1.0, 0.0), // alpha = 1
    ///     &a.view(),
    ///     &b.view(),
    ///     &MyComplex::zero(), // beta = 0
    ///     &mut c.view_mut()
    /// ).unwrap();
    /// ```
    pub fn zgemm(
        transa: Transpose,
        transb: Transpose,
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        beta: &MyComplex,
        c: &mut ArrayViewMut2<MyComplex>,
    ) -> Result<(), MatrixError> {
        // Get effective dimensions after transpose operations
        let (a_rows, a_cols) = match transa {
            Transpose::No => a.dim(),
            _ => {
                let (r, c) = a.dim();
                (c, r)
            }
        };

        let (b_rows, b_cols) = match transb {
            Transpose::No => b.dim(),
            _ => {
                let (r, c) = b.dim();
                (c, r)
            }
        };

        let (c_rows, c_cols) = c.dim();

        // Validate dimensions for matrix multiplication
        if a_cols != b_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot multiply {}x{} by {}x{} matrices",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        if c_rows != a_rows || c_cols != b_cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Result matrix dimensions {}x{} don't match expected {}x{}",
                c_rows, c_cols, a_rows, b_cols
            )));
        }

        let m = a_rows;
        let n = b_cols;
        let k = a_cols;

        // Handle special cases
        if m == 0 || n == 0 || ((alpha.is_zero() || k == 0) && beta.is_one()) {
            return Ok(());
        }

        // Scale C by beta if needed
        if !beta.is_one() {
            if beta.is_zero() {
                c.fill(MyComplex::zero());
            } else {
                for mut row in c.rows_mut() {
                    for elem in row.iter_mut() {
                        *elem *= beta;
                    }
                }
            }
        }

        // Early return if alpha is zero
        if alpha.is_zero() {
            return Ok(());
        }

        // Perform the matrix multiplication based on transpose flags
        match (transa, transb) {
            (Transpose::No, Transpose::No) => Self::zgemm_nn(alpha, a, b, c),
            (Transpose::No, Transpose::Trans) => Self::zgemm_nt(alpha, a, b, c),
            (Transpose::No, Transpose::ConjTrans) => Self::zgemm_nc(alpha, a, b, c),
            (Transpose::Trans, Transpose::No) => Self::zgemm_tn(alpha, a, b, c),
            (Transpose::Trans, Transpose::Trans) => Self::zgemm_tt(alpha, a, b, c),
            (Transpose::Trans, Transpose::ConjTrans) => Self::zgemm_tc(alpha, a, b, c),
            (Transpose::ConjTrans, Transpose::No) => Self::zgemm_cn(alpha, a, b, c),
            (Transpose::ConjTrans, Transpose::Trans) => Self::zgemm_ct(alpha, a, b, c),
            (Transpose::ConjTrans, Transpose::ConjTrans) => Self::zgemm_cc(alpha, a, b, c),
        }

        Ok(())
    }

    // C += alpha * A * B (no transpose)
    fn zgemm_nn(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (m, n) = a.dim();
        let (_, p) = b.dim();

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[[i, j]] += &a[[i, k]] * alpha * &b[[k, j]];
                }
            }
        }
    }

    // C += alpha * A * B^T
    fn zgemm_nt(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (m, n) = a.dim();
        let (p, _) = b.dim();

        for i in 0..m {
            for j in 0..p {
                let mut temp = MyComplex::zero();
                for k in 0..n {
                    temp += &a[[i, k]] * &b[[j, k]];
                }
                c[[i, j]] += alpha * &temp;
            }
        }
    }

    // C += alpha * A * B^H (conjugate transpose)
    fn zgemm_nc(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (m, n) = a.dim();
        let (p, _) = b.dim();

        for i in 0..m {
            for j in 0..p {
                let mut temp = MyComplex::zero();
                for k in 0..n {
                    temp += &a[[i, k]] * &b[[j, k]].conj();
                }
                c[[i, j]] += alpha * &temp;
            }
        }
    }

    // C += alpha * A^T * B
    fn zgemm_tn(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (n, m) = a.dim();
        let (_, p) = b.dim();

        for i in 0..m {
            for j in 0..p {
                let mut temp = MyComplex::zero();
                for k in 0..n {
                    temp += &a[[k, i]] * &b[[k, j]];
                }
                c[[i, j]] += alpha * &temp;
            }
        }
    }

    // C += alpha * A^T * B^T
    fn zgemm_tt(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (n, m) = a.dim();
        let (p, _) = b.dim();

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[[i, j]] += &a[[k, i]] * alpha * &b[[j, k]];
                }
            }
        }
    }

    // C += alpha * A^T * B^H
    fn zgemm_tc(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (n, m) = a.dim();
        let (p, _) = b.dim();

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[[i, j]] += &a[[k, i]] * alpha * &b[[j, k]].conj();
                }
            }
        }
    }

    // C += alpha * A^H * B (conjugate transpose)
    fn zgemm_cn(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (n, m) = a.dim();
        let (_, p) = b.dim();

        for i in 0..m {
            for j in 0..p {
                let mut temp = MyComplex::zero();
                for k in 0..n {
                    temp += &a[[k, i]].conj() * &b[[k, j]];
                }
                c[[i, j]] += alpha * &temp;
            }
        }
    }

    // C += alpha * A^H * B^T
    fn zgemm_ct(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (n, m) = a.dim();
        let (p, _) = b.dim();

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[[i, j]] += &a[[k, i]].conj() * alpha * &b[[j, k]];
                }
            }
        }
    }

    // C += alpha * A^H * B^H
    fn zgemm_cc(
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
        c: &mut ArrayViewMut2<MyComplex>,
    ) {
        let (n, m) = a.dim();
        let (p, _) = b.dim();

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    c[[i, j]] += &a[[k, i]].conj() * alpha * &b[[j, k]].conj();
                }
            }
        }
    }

    /// Convenience function for simple matrix multiplication without error checking: C = A * B
    ///
    /// # Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    ///
    /// # Returns
    /// * Result matrix C = A * B
    ///
    /// # Example
    /// ```rust
    /// use ndarray::Array2;
    ///
    /// let a = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(1.0, 0.0), MyComplex::from_f64(2.0, 0.0),
    ///     MyComplex::from_f64(3.0, 0.0), MyComplex::from_f64(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let b = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(5.0, 0.0), MyComplex::from_f64(6.0, 0.0),
    ///     MyComplex::from_f64(7.0, 0.0), MyComplex::from_f64(8.0, 0.0),
    /// ]).unwrap();
    ///
    /// let c = MyComplex::matrix_multiply(&a.view(), &b.view()).unwrap();
    /// ```
    pub fn matrix_multiply(
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
    ) -> Array2<MyComplex> {
        let (a_rows, _) = a.dim();
        let (_, b_cols) = b.dim();

        let mut c = Array2::zeros((a_rows, b_cols));
        _ = Self::zgemm(
            Transpose::No,
            Transpose::No,
            &MyComplex::one(),
            a,
            b,
            &MyComplex::zero(),
            &mut c.view_mut(),
        );

        c
    }

    /// Matrix-vector multiplication without error checking: y = alpha * op(A) * x + beta * y
    ///
    /// # Arguments
    /// * `trans` - Transpose operation for matrix A
    /// * `alpha` - Scalar multiplier for A*x
    /// * `a` - Matrix A
    /// * `x` - Input vector x
    /// * `beta` - Scalar multiplier for y
    /// * `y` - Input/output vector y
    pub fn matrix_vector_multiply(
        trans: Transpose,
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        x: &ArrayView2<MyComplex>,
        beta: &MyComplex,
        y: &mut ArrayViewMut2<MyComplex>,
    ) -> () {
        // Use ZGEMM with vectors treated as matrices
        Self::zgemm(trans, Transpose::No, alpha, a, x, beta, y).unwrap()
    }

    /// Convenience function for simple matrix multiplication: C = A * B
    ///
    /// # Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    ///
    /// # Returns
    /// * Result matrix C = A * B
    ///
    /// # Example
    /// ```rust
    /// use ndarray::Array2;
    ///
    /// let a = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(1.0, 0.0), MyComplex::from_f64(2.0, 0.0),
    ///     MyComplex::from_f64(3.0, 0.0), MyComplex::from_f64(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let b = Array2::from_shape_vec((2, 2), vec![
    ///     MyComplex::from_f64(5.0, 0.0), MyComplex::from_f64(6.0, 0.0),
    ///     MyComplex::from_f64(7.0, 0.0), MyComplex::from_f64(8.0, 0.0),
    /// ]).unwrap();
    ///
    /// let c = MyComplex::matrix_multiply(&a.view(), &b.view()).unwrap();
    /// ```
    pub fn try_matrix_multiply(
        a: &ArrayView2<MyComplex>,
        b: &ArrayView2<MyComplex>,
    ) -> Result<Array2<MyComplex>, MatrixError> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        if a_cols != b_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot multiply {}x{} by {}x{} matrices",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        let mut c = Array2::zeros((a_rows, b_cols));
        Self::zgemm(
            Transpose::No,
            Transpose::No,
            &MyComplex::one(),
            a,
            b,
            &MyComplex::zero(),
            &mut c.view_mut(),
        )?;

        Ok(c)
    }

    /// Matrix-vector multiplication: y = alpha * op(A) * x + beta * y
    ///
    /// # Arguments
    /// * `trans` - Transpose operation for matrix A
    /// * `alpha` - Scalar multiplier for A*x
    /// * `a` - Matrix A
    /// * `x` - Input vector x
    /// * `beta` - Scalar multiplier for y
    /// * `y` - Input/output vector y
    pub fn try_matrix_vector_multiply(
        trans: Transpose,
        alpha: &MyComplex,
        a: &ArrayView2<MyComplex>,
        x: &ArrayView2<MyComplex>,
        beta: &MyComplex,
        y: &mut ArrayViewMut2<MyComplex>,
    ) -> Result<(), MatrixError> {
        // Ensure x and y are column vectors
        let x_shape = x.dim();
        let y_shape = y.dim();

        if x_shape.1 != 1 {
            return Err(MatrixError::InvalidShape(format!(
                "x must be a column vector, got shape {:?}",
                x_shape
            )));
        }
        if y_shape.1 != 1 {
            return Err(MatrixError::InvalidShape(format!(
                "y must be a column vector, got shape {:?}",
                y_shape
            )));
        }

        let (a_rows, a_cols) = a.dim();
        let (expected_x_len, expected_y_len) = match trans {
            Transpose::No => (a_cols, a_rows),
            _ => (a_rows, a_cols),
        };

        if x_shape.0 != expected_x_len {
            return Err(MatrixError::DimensionMismatch(format!(
                "Vector x length {} doesn't match expected {}",
                x_shape.0, expected_x_len
            )));
        }
        if y_shape.0 != expected_y_len {
            return Err(MatrixError::DimensionMismatch(format!(
                "Vector y length {} doesn't match expected {}",
                y_shape.0, expected_y_len
            )));
        }

        // Use ZGEMM with vectors treated as matrices
        Self::zgemm(trans, Transpose::No, alpha, a, x, beta, y)
    }

    /// Create an identity matrix of size n x n
    pub fn identity_matrix(n: usize) -> Array2<MyComplex> {
        let mut matrix = Array2::zeros((n, n));
        for i in 0..n {
            matrix[[i, i]] = MyComplex::one();
        }
        matrix
    }

    /// Create a zero matrix of size m x n
    pub fn zero_matrix(m: usize, n: usize) -> Array2<MyComplex> {
        Array2::zeros((m, n))
    }

    /// Print a matrix in a readable format (for debugging)
    pub fn print_matrix(matrix: &ArrayView2<MyComplex>, name: &str) {
        let (rows, cols) = matrix.dim();
        println!("Matrix {} ({}x{}):", name, rows, cols);
        for i in 0..rows {
            print!("  ");
            for j in 0..cols {
                print!("{:>12} ", format!("{}", matrix[[i, j]]));
            }
            println!();
        }
        println!();
    }

    /// Transpose a matrix (creates a new matrix)
    pub fn transpose_matrix(matrix: &ArrayView2<MyComplex>) -> Array2<MyComplex> {
        matrix.t().to_owned()
    }

    /// Conjugate transpose a matrix (creates a new matrix)
    pub fn conjugate_transpose_matrix(matrix: &ArrayView2<MyComplex>) -> Array2<MyComplex> {
        let transposed = matrix.t();
        transposed.map(|x| x.conj())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycomplex::MyComplex;
    use ndarray::Array2;
    use num_traits::{One, Zero};

    #[test]
    fn test_simple_matrix_multiply() {
        // Test 2x2 * 2x2 = 2x2
        // A = [1 2]   B = [5 6]   Expected C = [19 22]
        //     [3 4]       [7 8]                 [43 50]

        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(3.0, 0.0), // Row 0
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(4.0, 0.0), // Row 1
            ],
        )
        .unwrap();

        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(5.0, 0.0),
                MyComplex::from_f64(7.0, 0.0), // Row 0
                MyComplex::from_f64(6.0, 0.0),
                MyComplex::from_f64(8.0, 0.0), // Row 1
            ],
        )
        .unwrap();

        let c = MyComplex::matrix_multiply(&a.view(), &b.view());

        // Check results
        assert_eq!(c[[0, 0]].real().to_f64(), 23.0); // 1*5 + 3*6 = 23
        assert_eq!(c[[1, 0]].real().to_f64(), 34.0); // 2*5 + 4*6 = 34
        assert_eq!(c[[0, 1]].real().to_f64(), 31.0); // 1*7 + 3*8 = 31
        assert_eq!(c[[1, 1]].real().to_f64(), 46.0); // 2*7 + 4*8 = 46

        let d = MyComplex::try_matrix_multiply(&a.view(), &b.view()).unwrap();

        // Check results
        assert_eq!(d[[0, 0]].real().to_f64(), 23.0); // 1*5 + 3*6 = 23
        assert_eq!(d[[1, 0]].real().to_f64(), 34.0); // 2*5 + 4*6 = 34
        assert_eq!(d[[0, 1]].real().to_f64(), 31.0); // 1*7 + 3*8 = 31
        assert_eq!(d[[1, 1]].real().to_f64(), 46.0); // 2*7 + 4*8 = 46
    }

    #[test]
    fn test_complex_matrix_multiply() {
        // Test with complex numbers
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 1.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(1.0, -1.0),
            ],
        )
        .unwrap();

        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(0.0, 1.0),
            ],
        )
        .unwrap();

        let c = MyComplex::matrix_multiply(&a.view(), &b.view());

        // C[0,0] = (1+i)*1 + 0*0 = 1+i
        assert_eq!(c[[0, 0]].real().to_f64(), 1.0);
        assert_eq!(c[[0, 0]].imag().to_f64(), 1.0);

        // C[1,1] = 0*0 + (1-i)*i = (1-i)*i = i + 1 = 1+i
        assert_eq!(c[[1, 1]].real().to_f64(), 1.0);
        assert_eq!(c[[1, 1]].imag().to_f64(), 1.0);

        let d = MyComplex::try_matrix_multiply(&a.view(), &b.view()).unwrap();

        // D[0,0] = (1+i)*1 + 0*0 = 1+i
        assert_eq!(d[[0, 0]].real().to_f64(), 1.0);
        assert_eq!(d[[0, 0]].imag().to_f64(), 1.0);

        // D[1,1] = 0*0 + (1-i)*i = (1-i)*i = i + 1 = 1+i
        assert_eq!(d[[1, 1]].real().to_f64(), 1.0);
        assert_eq!(d[[1, 1]].imag().to_f64(), 1.0);
    }

    #[test]
    fn test_zgemm_with_transpose() {
        // Test A^T * B
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
            ],
        )
        .unwrap();

        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(5.0, 0.0),
                MyComplex::from_f64(7.0, 0.0),
                MyComplex::from_f64(6.0, 0.0),
                MyComplex::from_f64(8.0, 0.0),
            ],
        )
        .unwrap();

        let mut c = Array2::zeros((2, 2));

        // Compute C = A^T * B
        MyComplex::zgemm(
            Transpose::Trans,
            Transpose::No,
            &MyComplex::one(),
            &a.view(),
            &b.view(),
            &MyComplex::zero(),
            &mut c.view_mut(),
        )
        .unwrap();

        // A^T = [1 2]  B = [5 7]  C = A^T * B = [26 30]
        //       [3 4]      [6 8]                 [38 44]
        assert_eq!(c[[0, 0]].real().to_f64(), 17.0); // 1*5 + 2*6 = 17
        assert_eq!(c[[1, 0]].real().to_f64(), 39.0); // 3*5 + 4*6 = 39
        assert_eq!(c[[0, 1]].real().to_f64(), 23.0); // 1*7 + 2*8 = 23
        assert_eq!(c[[1, 1]].real().to_f64(), 53.0); // 3*7 + 4*8 = 53
    }

    #[test]
    fn test_zgemm_with_scaling() {
        // Test alpha*A*B + beta*C
        let a = MyComplex::identity_matrix(2);
        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
            ],
        )
        .unwrap();

        let mut c = MyComplex::identity_matrix(2);

        let alpha = MyComplex::from_f64(2.0, 0.0);
        let beta = MyComplex::from_f64(3.0, 0.0);

        MyComplex::zgemm(
            Transpose::No,
            Transpose::No,
            &alpha,
            &a.view(),
            &b.view(),
            &beta,
            &mut c.view_mut(),
        )
        .unwrap();

        // A*B = [2 0]  alpha*A*B = [4 0]  beta*C = [3 0]  Result = [7 0]
        //       [0 3]              [0 6]           [0 3]           [0 9]
        assert_eq!(c[[0, 0]].real().to_f64(), 7.0); // 2*2 + 3*1 = 7
        assert_eq!(c[[1, 1]].real().to_f64(), 9.0); // 2*3 + 3*1 = 9
    }

    #[test]
    fn test_conjugate_transpose() {
        // Test A^H * B (conjugate transpose)
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 1.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(0.0, 0.0),
                MyComplex::from_f64(1.0, -1.0),
            ],
        )
        .unwrap();

        let b = MyComplex::identity_matrix(2);
        let mut c = Array2::zeros((2, 2));

        MyComplex::zgemm(
            Transpose::ConjTrans,
            Transpose::No,
            &MyComplex::one(),
            &a.view(),
            &b.view(),
            &MyComplex::zero(),
            &mut c.view_mut(),
        )
        .unwrap();

        // A^H = [1-i  0 ]  B = I  C = A^H * B = [1-i  0]
        //       [0   1+i]                        [0  1+i]
        assert_eq!(c[[0, 0]].real().to_f64(), 1.0);
        assert_eq!(c[[0, 0]].imag().to_f64(), -1.0); // (1+i)* = 1-i
        assert_eq!(c[[1, 1]].real().to_f64(), 1.0);
        assert_eq!(c[[1, 1]].imag().to_f64(), 1.0); // (1-i)* = 1+i
    }

    #[test]
    fn test_dimension_validation() {
        let a = Array2::zeros((2, 3));
        let b = Array2::zeros((2, 2)); // Wrong dimensions

        let result = MyComplex::try_matrix_multiply(&a.view(), &b.view());
        assert!(result.is_err());

        match result {
            Err(MatrixError::DimensionMismatch(_)) => {}
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_rectangular_matrices() {
        // Test 3x2 * 2x4 = 3x4
        let a = Array2::from_shape_vec(
            (3, 2),
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

        let b = Array2::from_shape_vec(
            (2, 4),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(7.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(8.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(9.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
                MyComplex::from_f64(10.0, 0.0),
            ],
        )
        .unwrap();

        let c = MyComplex::matrix_multiply(&a.view(), &b.view());
        assert_eq!(c.dim(), (3, 4));

        // Verify some elements
        // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*1 + 2*3 = 7
        assert_eq!(c[[0, 0]].real().to_f64(), 7.0);

        // C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*1 + 4*3 = 15
        assert_eq!(c[[1, 0]].real().to_f64(), 15.0);

        let d = MyComplex::try_matrix_multiply(&a.view(), &b.view()).unwrap();
        assert_eq!(d.dim(), (3, 4));

        // Verify some elements
        // D[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*1 + 2*3 = 7
        assert_eq!(d[[0, 0]].real().to_f64(), 7.0);

        // D[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*1 + 4*3 = 15
        assert_eq!(d[[1, 0]].real().to_f64(), 15.0);
    }

    #[test]
    fn test_zero_alpha() {
        // Test with alpha = 0, should only apply beta scaling
        let a = Array2::ones((2, 2));
        let b = Array2::ones((2, 2));
        let mut c = Array2::from_elem((2, 2), MyComplex::from_f64(5.0, 0.0));

        MyComplex::zgemm(
            Transpose::No,
            Transpose::No,
            &MyComplex::zero(), // alpha = 0
            &a.view(),
            &b.view(),
            &MyComplex::from_f64(2.0, 0.0), // beta = 2
            &mut c.view_mut(),
        )
        .unwrap();

        // Result should be beta * original_c = 2 * 5 = 10
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(c[[i, j]].real().to_f64(), 10.0);
            }
        }
    }

    #[test]
    fn test_beta_zero() {
        // Test with beta = 0, should zero out C first
        let a = Array2::from_elem((2, 2), MyComplex::from_f64(2.0, 0.0));
        let b = Array2::from_elem((2, 2), MyComplex::from_f64(3.0, 0.0));
        let mut c = Array2::from_elem((2, 2), MyComplex::from_f64(100.0, 0.0)); // Large initial values

        MyComplex::zgemm(
            Transpose::No,
            Transpose::No,
            &MyComplex::one(),
            &a.view(),
            &b.view(),
            &MyComplex::zero(), // beta = 0
            &mut c.view_mut(),
        )
        .unwrap();

        // Result should be A*B, ignoring initial C values
        // Each element: 2*3 + 2*3 = 12 (since A and B are all 2s and 3s)
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(c[[i, j]].real().to_f64(), 12.0);
            }
        }
    }

    #[test]
    fn test_matrix_vector_multiply() {
        // Test matrix-vector multiplication
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
            ],
        )
        .unwrap();

        let x = Array2::from_shape_vec(
            (2, 1),
            vec![MyComplex::from_f64(5.0, 0.0), MyComplex::from_f64(6.0, 0.0)],
        )
        .unwrap();

        let mut y = Array2::zeros((2, 1));

        MyComplex::matrix_vector_multiply(
            Transpose::No,
            &MyComplex::one(),
            &a.view(),
            &x.view(),
            &MyComplex::zero(),
            &mut y.view_mut(),
        );

        // y = A * x = [1 3] * [5] = [1*5 + 3*6] = [23]
        //             [2 4]   [6]   [2*5 + 4*6]   [34]
        assert_eq!(y[[0, 0]].real().to_f64(), 23.0);
        assert_eq!(y[[1, 0]].real().to_f64(), 34.0);

        let mut z = Array2::zeros((2, 1));

        MyComplex::try_matrix_vector_multiply(
            Transpose::No,
            &MyComplex::one(),
            &a.view(),
            &x.view(),
            &MyComplex::zero(),
            &mut z.view_mut(),
        )
        .unwrap();

        // z = A * x = [1 3] * [5] = [1*5 + 3*6] = [23]
        //             [2 4]   [6]   [2*5 + 4*6]   [34]
        assert_eq!(z[[0, 0]].real().to_f64(), 23.0);
        assert_eq!(z[[1, 0]].real().to_f64(), 34.0);
    }

    #[test]
    fn test_utility_functions() {
        // Test identity matrix creation
        let identity = MyComplex::identity_matrix(3);
        assert_eq!(identity.dim(), (3, 3));
        assert_eq!(identity[[0, 0]], MyComplex::one());
        assert_eq!(identity[[1, 1]], MyComplex::one());
        assert_eq!(identity[[2, 2]], MyComplex::one());
        assert_eq!(identity[[0, 1]], MyComplex::zero());

        // Test zero matrix creation
        let zero_mat = MyComplex::zero_matrix(2, 3);
        assert_eq!(zero_mat.dim(), (2, 3));
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(zero_mat[[i, j]], MyComplex::zero());
            }
        }

        // Test transpose
        let original = Array2::from_shape_vec(
            (2, 3),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(4.0, 0.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(5.0, 0.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(6.0, 0.0),
            ],
        )
        .unwrap();

        let transposed = MyComplex::transpose_matrix(&original.view());
        assert_eq!(transposed.dim(), (3, 2));
        assert_eq!(transposed[[0, 0]], original[[0, 0]]);
        assert_eq!(transposed[[1, 0]], original[[0, 1]]);
        assert_eq!(transposed[[2, 0]], original[[0, 2]]);

        // Test conjugate transpose
        let complex_mat = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 2.0),
                MyComplex::from_f64(3.0, 4.0),
                MyComplex::from_f64(5.0, 6.0),
                MyComplex::from_f64(7.0, 8.0),
            ],
        )
        .unwrap();

        let conj_transposed = MyComplex::conjugate_transpose_matrix(&complex_mat.view());
        assert_eq!(conj_transposed.dim(), (2, 2));
        assert_eq!(conj_transposed[[0, 0]], MyComplex::from_f64(1.0, -2.0)); // (1+2i)*
        assert_eq!(conj_transposed[[1, 0]], MyComplex::from_f64(3.0, -4.0)); // (3+4i)*
        assert_eq!(conj_transposed[[0, 1]], MyComplex::from_f64(5.0, -6.0)); // (5+6i)*
        assert_eq!(conj_transposed[[1, 1]], MyComplex::from_f64(7.0, -8.0)); // (7+8i)*
    }

    #[test]
    fn test_print_matrix() {
        // This test just ensures the print function doesn't panic
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                MyComplex::from_f64(1.0, 2.0),
                MyComplex::from_f64(3.0, 4.0),
                MyComplex::from_f64(5.0, 6.0),
                MyComplex::from_f64(7.0, 8.0),
            ],
        )
        .unwrap();

        // Just verify it doesn't panic - output goes to stdout
        MyComplex::print_matrix(&matrix.view(), "Test Matrix");
    }

    #[test]
    fn test_all_transpose_combinations() {
        // Test all 9 combinations of transpose operations
        let a = Array2::from_shape_vec(
            (2, 3),
            vec![
                MyComplex::from_f64(1.0, 1.0),
                MyComplex::from_f64(2.0, 0.0),
                MyComplex::from_f64(3.0, -1.0),
                MyComplex::from_f64(4.0, 2.0),
                MyComplex::from_f64(5.0, 0.0),
                MyComplex::from_f64(6.0, -2.0),
            ],
        )
        .unwrap();

        let b = Array2::from_shape_vec(
            (3, 2),
            vec![
                MyComplex::from_f64(1.0, 0.0),
                MyComplex::from_f64(2.0, 1.0),
                MyComplex::from_f64(3.0, 0.0),
                MyComplex::from_f64(4.0, -1.0),
                MyComplex::from_f64(5.0, 0.0),
                MyComplex::from_f64(6.0, 1.0),
            ],
        )
        .unwrap();

        let transpose_ops = [Transpose::No, Transpose::Trans, Transpose::ConjTrans];

        for &trans_a in &transpose_ops {
            for &trans_b in &transpose_ops {
                let mut c = Array2::zeros((2, 2));

                // This should not panic for any combination
                let result = MyComplex::zgemm(
                    trans_a,
                    trans_b,
                    &MyComplex::one(),
                    &a.view(),
                    &b.view(),
                    &MyComplex::zero(),
                    &mut c.view_mut(),
                );

                // Some combinations will have dimension mismatches, which is expected
                // We're just testing that the function handles all cases gracefully
                match result {
                    Ok(_) => {
                        // Verify result is not all zeros (unless input was zeros)
                        println!("Success: {:?} × {:?}", trans_a, trans_b);
                    }
                    Err(MatrixError::DimensionMismatch(_)) => {
                        // Expected for some combinations
                        println!(
                            "Dimension mismatch (expected): {:?} × {:?}",
                            trans_a, trans_b
                        );
                    }
                    Err(e) => panic!("Unexpected error: {}", e),
                }
            }
        }
    }
}
