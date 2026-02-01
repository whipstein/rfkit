use crate::{
    error::InversionError,
    num::{MyComplex, MyFloat, RFNum},
};
use ndarray::{
    Dimension, IntoDimension, NdIndex, ShapeError, SliceArg,
    iter::{AxisIter, AxisIterMut, IndexedIter, IndexedIterMut},
    prelude::*,
};
use ndarray_linalg::{Norm, error::LinalgError};
use num::complex::Complex64;
use num_traits::{One, Zero};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub mod ix1;
pub mod ix2;
pub mod ix3;

#[derive(Clone)]
pub struct Points<T, D>(pub Array<T, D>);

pub type Points1<T> = Points<T, Ix1>;
pub type Points2<T> = Points<T, Ix2>;
pub type Points3<T> = Points<T, Ix3>;

impl<T, D> Points<T, D>
where
    T: RFNum,
    D: Dimension,
{
    pub fn new(array: Array<T, D>) -> Self {
        Points(array)
    }

    pub fn inner(&self) -> &Array<T, D> {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut Array<T, D> {
        &mut self.0
    }

    pub fn swap<I>(&mut self, index1: I, index2: I)
    where
        I: NdIndex<D>,
    {
        self.0.swap(index1, index2);
    }
}

pub trait Pts<T, D>
where
    T: RFNum,
    D: Dimension,
    Points<T, D>: Add
        + Sub
        + Mul
        + Div
        + Neg
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Default
        + PartialEq
        + Zero
        + One
        + fmt::Display
        + fmt::Debug
        + Index<Self::Idx>
        + IndexMut<Self::Idx>,
{
    type Idx;
    type Tuple<'a>
    where
        Self: 'a;

    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: impl IntoDimension<Dim = D>) -> Self;
    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: impl IntoDimension<Dim = D>) -> Self;

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        shape: impl IntoDimension<Dim = D>,
    ) -> Result<Self, &'static str>
    where
        Self: Sized;
    fn from_elem(shape: impl IntoDimension<Dim = D>, elem: T) -> Self;
    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = D>, f: F) -> Self
    where
        F: Fn(D::Pattern) -> T;
    fn from_shape_vec(shape: impl IntoDimension<Dim = D>, v: Vec<T>) -> Result<Self, &'static str>
    where
        Self: Sized;

    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
    fn iter(&self) -> ndarray::iter::Iter<'_, T, D>;
    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, D>;
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

    /// Get and set outer axis point
    fn pt<I>(&self, index: usize) -> Points<T, I::OutDim>
    where
        I: SliceArg<D, OutDim = D::Smaller>;
    fn pt_mut<I>(&mut self, index: usize) -> Points<T, I::OutDim>
    where
        I: SliceArg<D, OutDim = D::Smaller>;
    fn set_pt(&mut self, index: usize, pt: Points<T, D::Smaller>);

    /// Add dimension
    fn insert_axis(self, axis: Axis) -> Points<T, D::Larger>;

    /// Get the len of axis
    fn len_of(&self, axis: Axis) -> usize;
    /// Get the lenght
    fn len(&self) -> usize;
    /// Get the number of points
    fn npts(&self) -> usize;
    /// Get the shape as (len, rows, cols)
    fn dim(&self) -> Self::Idx;
    /// Get the dimension
    fn raw_dim(&self) -> D;
    /// Get the shape as (len, rows, cols)
    fn shape(&self) -> Self::Idx;
    fn is_empty(&self) -> bool;

    /// Get a view of the matrix
    fn view(&self) -> ArrayView<'_, T, D>;
    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut<'_, T, D>;

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

    /// Check if a matrix is approximately equal to another matrix within a tolerance
    fn approx_eq(a: &ArrayView<T, D>, b: &ArrayView<T, D>, tol: f64) -> bool
    where
        <T as RFNum>::Real: PartialOrd<f64>,
        for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>;
}

pub trait Matrix<T, D>: Pts<T, D>
where
    T: RFNum,
    D: Dimension,
    Points<T, D>: Add
        + Sub
        + Mul
        + Div
        + Neg
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Default
        + PartialEq
        + Zero
        + One
        + fmt::Display
        + fmt::Debug
        + Index<Self::Idx>
        + IndexMut<Self::Idx>,
{
    /// Create an identity matrix of given (length, size)
    fn eye(size: impl IntoDimension<Dim = D::Smaller>) -> Self;

    /// Get the number of rows
    fn nrows(&self) -> usize;
    /// Get the number of columns
    fn ncols(&self) -> usize;

    /// Check if the matrix is square
    fn is_square(&self) -> bool;

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
    fn det(&self) -> Result<Array<T, <D::Smaller as Dimension>::Smaller>, LinalgError>
    where
        for<'a> &'a T: std::ops::Mul<T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>;
    /// Element-wise conjugate
    fn conj(&self) -> Self;
    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> Array<T::Real, <D::Smaller as Dimension>::Smaller>
    where
        T::Real: RFNum;

    /// Get a row as a new matrix (1 x ncols)
    fn row(&self, index: usize) -> Points<T, D::Smaller>;
    /// Get a column as a new matrix (nrows x 1)
    fn col(&self, index: usize) -> Points<T, D::Smaller>;

    /// Set a row from another matrix
    fn set_row(&mut self, index: usize, row: &Points<T, D::Smaller>);
    /// Set a column from another matrix
    fn set_col(&mut self, index: usize, col: &Points<T, D::Smaller>);

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    fn inv(&self) -> Self
    where
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
        <T as RFNum>::Real: PartialOrd,
        <T as RFNum>::Real: PartialOrd<f64>;
    fn try_inv(&self) -> Result<Self, InversionError>
    where
        Self: Sized,
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
        <T as RFNum>::Real: PartialOrd,
        <T as RFNum>::Real: PartialOrd<f64>;

    /// Solve the linear system Ax = b using LU decomposition
    fn solve_linear_system(&self, b: &ArrayView<T, D>) -> Result<Array<T, D>, InversionError>
    where
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
        <T as RFNum>::Real: PartialOrd,
        <T as RFNum>::Real: PartialOrd<f64>;
}

// Indexing
impl<T, D, I> Index<I> for Points<T, D>
where
    T: RFNum,
    D: Dimension,
    I: NdIndex<D>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, D, I> IndexMut<I> for Points<T, D>
where
    T: RFNum,
    D: Dimension,
    I: NdIndex<D>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Norm product implementations
impl<T, D> Norm for Points<T, D>
where
    T: RFNum,
    D: Dimension,
    <T as RFNum>::Real: RFNum + PartialOrd<T::Real>,
{
    type Output = T::Real;

    fn norm_l1(&self) -> Self::Output {
        self.0.iter().map(|x| x.norm()).sum()
    }

    fn norm_l2(&self) -> Self::Output {
        self.0.iter().map(|x| x.square()).sum::<T::Real>().sqrt()
    }

    fn norm_max(&self) -> Self::Output {
        self.0.iter().fold(T::Real::zero(), |f, val| {
            let v = val.norm();
            if f > v { f } else { v }
        })
    }
}

macro_rules! impl_self_math_op(
    ($trt:ident, $operator:tt, $mth:ident, $doc:expr) => (
        impl<T, D> $trt for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
        {
            type Output = Self;

            #[track_caller]
            fn $mth(self, other: Self) -> Self {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                Points(&self.0 $operator &other.0)
            }
        }

        impl<T, D> $trt<&Points<T, D>> for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
        {
            type Output = Self;

            #[track_caller]
            fn $mth(self, other: &Self) -> Self {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                Points(&self.0 $operator &other.0)
            }
        }

        impl<T, D> $trt<Points<T, D>> for &Points<T, D>
        where
            T: RFNum,
            D: Dimension,
        {
            type Output = Points<T, D>;

            #[track_caller]
            fn $mth(self, other: Points<T, D>) -> Self::Output
            {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                Points(&self.0 $operator &other.0)
            }
        }

        impl<T, D> $trt<&Points<T, D>> for &Points<T, D>
        where
            T: RFNum,
            D: Dimension,
        {
            type Output = Points<T, D>;

            #[track_caller]
            fn $mth(self, other: &Points<T, D>) -> Self::Output
            {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                Points(&self.0 $operator &other.0)
            }
        }

        impl<T, D> $trt<T> for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
            for<'a, 'b> &'a T: $trt<&'b T, Output = T>,
        {
            type Output = Self;

            #[track_caller]
            fn $mth(self, scalar: T) -> Self::Output {
                Points(self.0.map(|x| x $operator &scalar))
            }
        }

        impl<T, D> $trt<T> for &Points<T, D>
        where
            T: RFNum,
            D: Dimension,
            for<'a, 'b> &'a T: $trt<&'b T, Output = T>,
        {
            type Output = Points<T, D>;

            #[track_caller]
            fn $mth(self, scalar: T) -> Self::Output {
                Points(self.0.map(|x| x $operator &scalar))
            }
        }

        impl<T, D> $trt<&T> for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
            for<'a, 'b> &'a T: $trt<&'b T, Output = T>,
        {
            type Output = Self;

            #[track_caller]
            fn $mth(self, scalar: &T) -> Self::Output {
                Points(self.0.map(|x| x $operator scalar))
            }
        }

        impl<T, D> $trt<&T> for &Points<T, D>
        where
            T: RFNum,
            D: Dimension,
            for<'a, 'b> &'a T: $trt<&'b T, Output = T>,
        {
            type Output = Points<T, D>;

            #[track_caller]
            fn $mth(self, scalar: &T) -> Self::Output {
                Points(self.0.map(|x| x $operator scalar))
            }
        }
    );
);

macro_rules! impl_math_op(
    ($trt:ident, $operator:tt, $mth:ident, $lhs:ident, $rhs:ident, $output:ident, $doc:expr) => (
        impl<D> $trt<&Points<$rhs, D>> for Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, other: &Points<$rhs, D>) -> Self::Output {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                let mut pts = Array::<$output, D>::zeros(self.0.dim());
                azip!((pts in &mut pts, x in &self.0, y in &other.0) *pts = x $operator y);
                Points(pts)
            }
        }

        impl<D> $trt<Points<$rhs, D>> for &Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, other: Points<$rhs, D>) -> Self::Output
            {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                let mut pts = Array::<$output, D>::zeros(self.0.dim());
                azip!((pts in &mut pts, x in &self.0, y in &other.0) *pts = x $operator y);
                Points(pts)
            }
        }

        impl<D> $trt<&Points<$rhs, D>> for &Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, other: &Points<$rhs, D>) -> Self::Output
            {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                let mut pts = Array::<$output, D>::zeros(self.0.dim());
                azip!((pts in &mut pts, x in &self.0, y in &other.0) *pts = x $operator y);
                Points(pts)
            }
        }

        impl<D> $trt<$rhs> for Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: $rhs) -> Self::Output {
                Points(self.0.map(|x| x $operator &scalar))
            }
        }

        impl<D> $trt<$rhs> for &Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: $rhs) -> Self::Output {
                Points(self.0.map(|x| x $operator &scalar))
            }
        }

        impl<D> $trt<&$rhs> for Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: &$rhs) -> Self::Output {
                Points(self.0.map(|x| x $operator scalar))
            }
        }

        impl<D> $trt<&$rhs> for &Points<$lhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: &$rhs) -> Self::Output {
                Points(self.0.map(|x| x $operator scalar))
            }
        }

        impl<D> $trt<&Points<$lhs, D>> for Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, other: &Points<$lhs, D>) -> Self::Output {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                let mut pts = Array::<$output, D>::zeros(self.0.dim());
                azip!((pts in &mut pts, x in &self.0, y in &other.0) *pts = x $operator y);
                Points(pts)
            }
        }

        impl<D> $trt<Points<$lhs, D>> for &Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, other: Points<$lhs, D>) -> Self::Output
            {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                let mut pts = Array::<$output, D>::zeros(self.0.dim());
                azip!((pts in &mut pts, x in &self.0, y in &other.0) *pts = x $operator y);
                Points(pts)
            }
        }

        impl<D> $trt<&Points<$lhs, D>> for &Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, other: &Points<$lhs, D>) -> Self::Output
            {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                let mut pts = Array::<$output, D>::zeros(self.0.dim());
                azip!((pts in &mut pts, x in &self.0, y in &other.0) *pts = x $operator y);
                Points(pts)
            }
        }

        impl<D> $trt<$lhs> for Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: $lhs) -> Self::Output {
                Points(self.0.map(|x| x $operator &scalar))
            }
        }

        impl<D> $trt<$lhs> for &Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: $lhs) -> Self::Output {
                Points(self.0.map(|x| x $operator &scalar))
            }
        }

        impl<D> $trt<&$lhs> for Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: &$lhs) -> Self::Output {
                Points(self.0.map(|x| x $operator scalar))
            }
        }

        impl<D> $trt<&$lhs> for &Points<$rhs, D>
        where
            D: Dimension,
        {
            type Output = Points<$output, D>;

            #[track_caller]
            fn $mth(self, scalar: &$lhs) -> Self::Output {
                Points(self.0.map(|x| x $operator scalar))
            }
        }
    );
);

macro_rules! impl_math_assign_op(
    ($trt:ident, $trt_alt:ident, $operator:tt, $op_alt:tt, $mth:ident, $doc:expr) => (
        impl<T, D> $trt for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
        {
            fn $mth(&mut self, other: Self) {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                self.0 $operator &other.0;
            }
        }

        impl<T, D> $trt<&Points<T, D>> for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
        {
            fn $mth(&mut self, other: &Points<T, D>) {
                if self.0.shape() != other.0.shape() {
                    panic!(
                        "Point dimensions must match for {}: {:?} vs {:?}",
                        $doc,
                        self.0.shape(),
                        other.0.shape()
                    );
                }
                self.0 $operator &other.0;
            }
        }

        impl<T, D> $trt<T> for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
            for<'a, 'b> &'a T: $trt_alt<&'b T, Output = T>,
        {
            fn $mth(&mut self, scalar: T) {
                self.0.map_inplace(|x| *x = &*x $op_alt &scalar);
            }
        }

        impl<T, D> $trt<&T> for Points<T, D>
        where
            T: RFNum,
            D: Dimension,
            for<'a, 'b> &'a T: $trt_alt<&'b T, Output = T>,
        {
            fn $mth(&mut self, scalar: &T) {
                self.0.map_inplace(|x| *x = &*x $op_alt scalar);
            }
        }
    );
);

mod arithmetic_ops {
    use super::*;
    use std::ops::*;

    fn clone_opf<A: Clone, B: Clone, C>(f: impl Fn(A, B) -> C) -> impl FnMut(&A, &B) -> C {
        move |x, y| f(x.clone(), y.clone())
    }

    fn clone_iopf<A: Clone, B: Clone>(f: impl Fn(A, B) -> A) -> impl FnMut(&mut A, &B) {
        move |x, y| *x = f(x.clone(), y.clone())
    }

    fn clone_iopf_rev<A: Clone, B: Clone>(f: impl Fn(A, B) -> B) -> impl FnMut(&mut B, &A) {
        move |x, y| *x = f(y.clone(), x.clone())
    }

    impl_self_math_op!(Add, +, add, "addition");
    impl_self_math_op!(Sub, -, sub, "subtraction");
    impl_self_math_op!(Mul, *, mul, "multiplication");
    impl_self_math_op!(Div, /, div, "division");
    impl_math_assign_op!(AddAssign, Add, +=, +, add_assign, "addition assign");
    impl_math_assign_op!(SubAssign, Sub, -=, -, sub_assign, "subtraction assign");
    impl_math_assign_op!(MulAssign, Mul, *=, *, mul_assign, "multiplication assign");
    impl_math_assign_op!(DivAssign, Div, /=, /, div_assign, "division assign");
    impl_math_op!(Add, +, add, f64, MyFloat, MyFloat, "addition");
    impl_math_op!(Sub, -, sub, f64, MyFloat, MyFloat, "subtraction");
    impl_math_op!(Mul, *, mul, f64, MyFloat, MyFloat, "multiplication");
    impl_math_op!(Div, /, div, f64, MyFloat, MyFloat, "division");
    impl_math_op!(Add, +, add, f64, Complex64, Complex64, "addition");
    impl_math_op!(Sub, -, sub, f64, Complex64, Complex64, "subtraction");
    impl_math_op!(Mul, *, mul, f64, Complex64, Complex64, "multiplication");
    impl_math_op!(Div, /, div, f64, Complex64, Complex64, "division");
    impl_math_op!(Add, +, add, f64, MyComplex, MyComplex, "addition");
    impl_math_op!(Sub, -, sub, f64, MyComplex, MyComplex, "subtraction");
    impl_math_op!(Mul, *, mul, f64, MyComplex, MyComplex, "multiplication");
    impl_math_op!(Div, /, div, f64, MyComplex, MyComplex, "division");
    impl_math_op!(Add, +, add, MyFloat, Complex64, MyComplex, "addition");
    impl_math_op!(Sub, -, sub, MyFloat, Complex64, MyComplex, "subtraction");
    impl_math_op!(Mul, *, mul, MyFloat, Complex64, MyComplex, "multiplication");
    impl_math_op!(Div, /, div, MyFloat, Complex64, MyComplex, "division");
    impl_math_op!(Add, +, add, MyFloat, MyComplex, MyComplex, "addition");
    impl_math_op!(Sub, -, sub, MyFloat, MyComplex, MyComplex, "subtraction");
    impl_math_op!(Mul, *, mul, MyFloat, MyComplex, MyComplex, "multiplication");
    impl_math_op!(Div, /, div, MyFloat, MyComplex, MyComplex, "division");
    impl_math_op!(Add, +, add, Complex64, MyComplex, MyComplex, "addition");
    impl_math_op!(Sub, -, sub, Complex64, MyComplex, MyComplex, "subtraction");
    impl_math_op!(Mul, *, mul, Complex64, MyComplex, MyComplex, "multiplication");
    impl_math_op!(Div, /, div, Complex64, MyComplex, MyComplex, "division");

    // Negation
    impl<T, D> Neg for Points<T, D>
    where
        T: RFNum,
        D: Dimension,
    {
        type Output = Self;

        fn neg(self) -> Self {
            Points(-self.0)
        }
    }

    impl<T, D> Neg for &Points<T, D>
    where
        T: RFNum,
        D: Dimension,
        for<'a> &'a T: std::ops::Neg<Output = T>,
    {
        type Output = Points<T, D>;

        fn neg(self) -> Self::Output {
            Points(-&self.0)
        }
    }
}

// Traits
impl<T, D> PartialEq for Points<T, D>
where
    T: RFNum,
    D: Dimension,
{
    fn eq(&self, other: &Self) -> bool {
        if self.0.shape() != other.0.shape() {
            return false;
        }

        self.0.iter().zip(other.0.iter()).all(|(a, b)| *a == *b)
    }
}
