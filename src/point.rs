use crate::error::InversionError;
use ndarray::SliceArg;
use ndarray::iter::{IndexedIter, IndexedIterMut};
use ndarray::prelude::*;

pub mod c64;
pub mod f64;
pub mod mycomplex;
pub mod myfloat;

pub struct Point<T>(Array2<T>);

pub trait Pt<T, U> {
    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: (usize, usize)) -> Self;
    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: (usize, usize)) -> Self;
    /// Create an identity matrix of given size
    fn eye(size: usize) -> Self;

    /// Create a matrix from a 2D vector
    fn from_vec(data: Vec<Vec<T>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_f64(data: Vec<Vec<f64>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 2D vector of U values
    fn from_vec_float(data: Vec<Vec<U>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_c64(data: Vec<Vec<(f64, f64)>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<(U, U)>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(data: Vec<U>, rows: usize, cols: usize) -> Result<Self, &'static str>
    where
        Self: Sized;
    fn from_shape_fn<F>(shape: (usize, usize), f: F) -> Self
    where
        F: Fn((usize, usize)) -> T;
    fn from_shape_vec(shape: (usize, usize), v: Vec<T>) -> Result<Self, &'static str>
    where
        Self: Sized;

    fn indexed_iter(&self) -> IndexedIter<'_, T, Ix2>;
    fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, T, Ix2>;
    fn slice<I>(&self, info: I) -> ArrayView<'_, T, I::OutDim>
    where
        I: SliceArg<Ix2>;
    fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut<'_, T, I::OutDim>
    where
        I: SliceArg<Ix2>;

    /// Get the number of rows
    fn nrows(&self) -> usize;
    /// Get the number of columns
    fn ncols(&self) -> usize;
    /// Get the shape as (rows, cols)
    fn dim(&self) -> (usize, usize);
    /// Get the shape as (rows, cols)
    fn shape(&self) -> (usize, usize);

    /// Check if the matrix is square
    fn is_square(&self) -> bool;

    /// Get a view of the matrix
    fn view(&self) -> ArrayView2<'_, T>;
    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut2<'_, T>;

    /// Transpose the matrix
    fn t(&self) -> Self;
    /// Transpose the matrix
    fn transpose(&self) -> Self;
    /// Get the conjugate transpose (Hermitian transpose)
    fn h(&self) -> Self;
    /// Get the conjugate transpose (Hermitian transpose)
    fn conj_transpose(&self) -> Self;

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> T;
    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> T;
    /// Element-wise conjugate
    fn conj(&self) -> Self;
    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> U;

    /// Point multiplication
    fn dot(&self, other: &Self) -> Self;
    // fn not(&self) -> Self;

    /// Get a row as a new matrix (1 x ncols)
    fn row(&self, index: usize) -> Self;
    /// Get a column as a new matrix (nrows x 1)
    fn col(&self, index: usize) -> Self;

    /// Set a row from another matrix
    fn set_row(&mut self, index: usize, row: &Self);
    /// Set a column from another matrix
    fn set_col(&mut self, index: usize, col: &Self);

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array2<T>;
    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array2<T>;

    /// Create a matrix filled with the given value
    fn fill(rows: usize, cols: usize, value: T) -> Self;

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T;
    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T;

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    ///
    /// This function computes A^(-1) such that A * A^(-1) = I, where I is the identity matrix.
    ///
    /// # Arguments
    /// * `matrix` - A square matrix to invert
    ///
    /// # Returns
    /// * `Ok(Array2<Complex64>)` - The inverted matrix
    /// * `Err(InversionError)` - If the matrix is not square, singular, or other errors
    ///
    /// # Examples
    /// ```rust
    /// use num::complex::Complex64;
    /// use ndarray::prelude::*;
    /// use rfkit::prelude::*;
    ///
    /// // Create a 2x2 matrix
    /// let matrix = Point::from_shape_vec((2, 2), vec![
    ///     Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
    ///     Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let inv_matrix = matrix.inv();
    /// ```
    fn inv(&self) -> Self;
    fn try_inv(&self) -> Result<Self, InversionError>
    where
        Self: Sized;

    /// Solve the linear system Ax = b using LU decomposition
    ///
    /// # Arguments
    /// * `a` - Coefficient matrix A (n×n)
    /// * `b` - Right-hand side vector b (n×1)
    ///
    /// # Returns
    /// * `Ok(Array2<MyComplex>)` - Solution vector x
    /// * `Err(InversionError)` - If the system cannot be solved
    fn solve_linear_system(&self, b: &ArrayView2<T>) -> Result<Array2<T>, InversionError>;

    /// Check if a matrix is approximately equal to another matrix within a tolerance
    ///
    /// # Arguments
    /// * `a` - First matrix
    /// * `b` - Second matrix  
    /// * `tol` - Tolerance for comparison
    ///
    /// # Returns
    /// * `true` if matrices are approximately equal, `false` otherwise
    fn approx_eq(a: &ArrayView2<T>, b: &ArrayView2<T>, tol: f64) -> bool;
}
