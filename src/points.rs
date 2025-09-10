use ndarray::iter::{AxisIter, AxisIterMut, IndexedIter, IndexedIterMut};
use ndarray::{ShapeError, SliceArg, prelude::*};

mod complex;
mod float;

pub use self::complex::Points;
pub use self::float::Pointsf64;

pub trait Pts<T, U> {
    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: (usize, usize, usize)) -> Self;
    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: (usize, usize, usize)) -> Self;
    /// Create an identity matrix of given (length, size)
    fn eye(size: (usize, usize)) -> Self;

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_f64(data: Vec<Vec<Vec<f64>>>) -> Result<Self, &'static str>
    where
        Self: Sized;
    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<Vec<(f64, f64)>>>) -> Result<Self, &'static str>
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
    /// Get the shape as (len, rows, cols)
    fn dim(&self) -> (usize, usize, usize);
    /// Get the shape as (len, rows, cols)
    fn shape(&self) -> (usize, usize, usize);

    /// Get a view of the matrix
    fn view(&self) -> ArrayView3<T>;
    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut3<T>;

    /// Element-wise conjugate
    fn conj(&self) -> Self;

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
}
