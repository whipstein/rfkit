use crate::error::InversionError;
use ndarray::iter::{AxisIter, AxisIterMut, IndexedIter, IndexedIterMut};
use ndarray::{ShapeError, SliceArg, prelude::*};

pub mod c64;
pub mod f64;
pub mod mycomplex;
pub mod myfloat;

pub struct Points<T>(Array3<T>);

pub trait Pts<T, U> {
    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: (usize, usize, usize)) -> Self;
    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: (usize, usize, usize)) -> Self;
    /// Create an identity matrix of given (length, size)
    fn eye(size: (usize, usize)) -> Self;

    /// Create a matrix from a 3D vector
    fn from_vec(data: Vec<Vec<Vec<T>>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 3D vector of f64 values
    fn from_vec_f64(data: Vec<Vec<Vec<f64>>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 3D vector of float values
    fn from_vec_float(data: Vec<Vec<Vec<U>>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 3D vector of complex tuples (real, imag)
    fn from_vec_c64(data: Vec<Vec<Vec<(f64, f64)>>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 3D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<Vec<(U, U)>>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        len: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, &'static str>
    where
        Self: Sized;
    fn from_shape_fn<F>(shape: (usize, usize, usize), f: F) -> Self
    where
        F: Fn((usize, usize, usize)) -> T;
    fn from_shape_vec(shape: (usize, usize, usize), v: Vec<T>) -> Result<Self, &'static str>
    where
        Self: Sized;

    fn axis_iter(&self, axis: Axis) -> AxisIter<'_, T, Ix2>;
    fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, T, Ix2>;
    fn indexed_iter(&self) -> IndexedIter<'_, T, Ix3>;
    fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, T, Ix3>;
    fn outer_iter(&self) -> AxisIter<'_, T, Ix2>;
    fn outer_iter_mut(&mut self) -> AxisIterMut<'_, T, Ix2>;

    fn slice<I>(&self, info: I) -> ArrayView<'_, T, I::OutDim>
    where
        I: SliceArg<Ix3>;
    fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut<'_, T, I::OutDim>
    where
        I: SliceArg<Ix3>;

    fn assign<E: Dimension>(&mut self, rhs: &Array<T, E>);
    fn push(&mut self, axis: Axis, array: ArrayView<'_, T, Ix2>) -> Result<(), ShapeError>
    where
        T: Clone;

    /// Get the len of axis
    fn len_of(&self, axis: Axis) -> usize;
    /// Get the number of points
    fn npts(&self) -> usize;
    /// Get the number of rows
    fn nrows(&self) -> usize;
    /// Get the number of columns
    fn ncols(&self) -> usize;
    /// Get the shape as (len, rows, cols)
    fn dim(&self) -> (usize, usize, usize);
    /// Get the shape as (len, rows, cols)
    fn shape(&self) -> (usize, usize, usize);

    /// Check if the matrix is square
    fn is_square(&self) -> bool;

    /// Get a view of the matrix
    fn view(&self) -> ArrayView3<'_, T>;
    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut3<'_, T>;

    /// Transpose the matrix
    fn t(&self) -> Self;
    /// Transpose the matrix
    fn transpose(&self) -> Self;
    /// Get the conjugate transpose (Hermitian transpose)
    fn h(&self) -> Self;
    /// Get the conjugate transpose (Hermitian transpose)
    fn conj_transpose(&self) -> Self;

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> Array1<T>;
    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Array1<T>;
    /// Element-wise conjugate
    fn conj(&self) -> Self;
    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> Array1<U>;

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
    fn inner(&self) -> &Array3<T>;
    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array3<T>;
    fn into_raw_vec_and_offset(self) -> (Vec<T>, Option<usize>);

    /// Create a matrix filled with the given value
    fn fill(len: usize, rows: usize, cols: usize, value: T) -> Self;

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T;
    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T;

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    fn inv(&self) -> Self;
    fn try_inv(&self) -> Result<Self, InversionError>
    where
        Self: Sized;

    /// Solve the linear system Ax = b using LU decomposition
    fn solve_linear_system(&self, b: &ArrayView3<T>) -> Result<Array3<T>, InversionError>;

    /// Check if a matrix is approximately equal to another matrix within a tolerance
    fn approx_eq(a: &ArrayView3<T>, b: &ArrayView3<T>, tol: f64) -> bool;
}
