use crate::{error::InversionError, num::RFNum};
use ndarray::{
    Dimension, IntoDimension, ShapeError, SliceArg,
    iter::{AxisIter, AxisIterMut, IndexedIter, IndexedIterMut},
    prelude::*,
};
use ndarray_linalg::error::LinalgError;

pub mod ix2;
pub mod ix3;

pub use ix3::c64;
pub use ix3::f64;
pub use ix3::mycomplex;
pub use ix3::myfloat;

#[derive(Clone)]
pub struct Points<T, D>(pub Array<T, D>);

pub trait Pts<T, D>
where
    T: RFNum,
    D: Dimension,
{
    type Dim;
    type Tuple<'a>
    where
        Self: 'a;

    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: impl IntoDimension<Dim = D>) -> Self;
    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: impl IntoDimension<Dim = D>) -> Self;
    /// Create an identity matrix of given (length, size)
    fn eye(size: impl IntoDimension<Dim = D>) -> Self;

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        shape: impl IntoDimension<Dim = D>,
    ) -> Result<Self, &'static str>
    where
        Self: Sized;
    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = D>, f: F) -> Self
    where
        F: Fn(D::Pattern) -> T;
    fn from_shape_vec(shape: impl IntoDimension<Dim = D>, v: Vec<T>) -> Result<Self, &'static str>
    where
        Self: Sized;

    fn axis_iter(&self, axis: Axis) -> AxisIter<'_, T, D::Smaller>;
    fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, T, D::Smaller>;
    fn indexed_iter(&self) -> IndexedIter<'_, T, D>;
    fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, T, D>;
    fn outer_iter(&self) -> AxisIter<'_, T, D::Smaller>;
    fn outer_iter_mut(&mut self) -> AxisIterMut<'_, T, D::Smaller>;

    fn slice<I>(&self, info: I) -> ArrayView<'_, T, I::OutDim>
    where
        I: SliceArg<D>;
    fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut<'_, T, I::OutDim>
    where
        I: SliceArg<D>;

    fn assign<E: Dimension>(&mut self, rhs: &Array<T, E>);
    fn push(&mut self, axis: Axis, array: ArrayView<'_, T, D::Smaller>) -> Result<(), ShapeError>
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
    fn dim(&self) -> Self::Dim;
    /// Get the dimension
    fn raw_dim(&self) -> D;
    /// Get the shape as (len, rows, cols)
    fn shape(&self) -> Self::Dim;

    /// Check if the matrix is square
    fn is_square(&self) -> bool;

    /// Get a view of the matrix
    fn view(&self) -> ArrayView<'_, T, D>;
    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut<'_, T, D>;

    /// Transpose the matrix
    fn t(&self) -> Self;
    /// Transpose the matrix
    fn transpose(&self) -> Self;
    /// Get the conjugate transpose (Hermitian transpose)
    fn h(&self) -> Self;
    /// Get the conjugate transpose (Hermitian transpose)
    fn conj_transpose(&self) -> Self;

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> Result<Array<T, <D::Smaller as Dimension>::Smaller>, LinalgError>;
    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Result<Array<T, <D::Smaller as Dimension>::Smaller>, LinalgError>;
    /// Element-wise conjugate
    fn conj(&self) -> Self;
    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> Array<T::Real, <D::Smaller as Dimension>::Smaller>;

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
    fn inner(&self) -> &Array<T, D>;
    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array<T, D>;
    fn into_raw_vec_and_offset(self) -> (Vec<T>, Option<usize>);

    /// Create a matrix filled with the given value
    fn fill(shape: impl IntoDimension<Dim = D>, value: T) -> Self;

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
    fn solve_linear_system(&self, b: &ArrayView<T, D>) -> Result<Array<T, D>, InversionError>;

    /// Check if a matrix is approximately equal to another matrix within a tolerance
    fn approx_eq(a: &ArrayView<T, D>, b: &ArrayView<T, D>, tol: f64) -> bool;
}
