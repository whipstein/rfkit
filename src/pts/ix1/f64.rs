use crate::{
    myfloat::MyFloat,
    pts::{Points, Pts},
};
use ndarray::{IntoDimension, SliceArg, linalg::Dot, prelude::*};
use num_traits::{One, Zero};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A matrix wrapper around ndarray::Array1 with f64 elements
// pub struct Points<f64, Ix1>(Array1<f64>);

impl Points<f64, Ix1> {
    pub fn new(array: Array1<f64>) -> Self {
        Points(array)
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec(data: Vec<f64>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let len = data.len();

        let mut matrix = Self::zeros(len);
        for (i, &val) in data.iter().enumerate() {
            matrix[i] = val;
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_f64(data: Vec<f64>) -> Result<Self, &'static str> {
        Self::from_vec(data)
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_float(data: Vec<f64>) -> Result<Self, &'static str> {
        Self::from_vec(data)
    }
}

impl Pts<f64, Ix1> for Points<f64, Ix1> {
    type Idx = usize;
    type Tuple<'a> = &'a [(f64, f64)];

    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>) -> Self {
        Points(Array1::from_elem(shape, f64::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>) -> Self {
        Points(Array1::from_elem(shape, f64::one()))
    }

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        shape: impl IntoDimension<Dim = Dim<[usize; 1]>>,
    ) -> Result<Self, &'static str> {
        let len = shape.into_dimension().into_pattern();
        if data.len() != len {
            return Err("Data length does not match matrix dimensions");
        }

        let mut matrix = Self::zeros(len);
        for (idx, _) in data.iter().enumerate() {
            matrix[idx] = 0.0;
        }

        Ok(matrix)
    }

    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>, f: F) -> Self
    where
        F: Fn(Self::Idx) -> f64,
    {
        Points(Array1::<f64>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(
        shape: impl IntoDimension<Dim = Dim<[usize; 1]>>,
        v: Vec<f64>,
    ) -> Result<Self, &'static str> {
        match Array1::<f64>::from_shape_vec(shape, v) {
            Ok(x) => Ok(Points(x)),
            Err(_) => Err("Cannot create vector"),
        }
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, f64, ndarray::Ix0> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(&mut self, axis: Axis) -> ndarray::iter::AxisIterMut<'_, f64, ndarray::Ix0> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, f64, ndarray::Ix1> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, f64, ndarray::Ix1> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, f64, ndarray::Ix0> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, f64, ndarray::Ix0> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, f64, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix1>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, f64, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix1>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<f64, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, f64, ndarray::Ix0>,
    ) -> Result<(), ndarray::ShapeError>
    where
        f64: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get and set outer axis point
    fn pt<I>(&self, index: usize) -> ndarray::ArrayView<'_, f64, I::OutDim>
    where
        I: SliceArg<Ix1, OutDim = Dim<[usize; 0]>>,
    {
        if index >= self.len() {
            panic!(
                "Length index {} out of bounds for matrix with {} points",
                index,
                self.len()
            );
        }

        self.slice(s![index])
    }

    fn pt_mut<I>(&mut self, index: usize) -> ndarray::ArrayViewMut<'_, f64, I::OutDim>
    where
        I: SliceArg<Ix1, OutDim = Dim<[usize; 0]>>,
    {
        if index >= self.len() {
            panic!(
                "Length index {} out of bounds for matrix with {} points",
                index,
                self.len()
            );
        }

        self.slice_mut(s![index])
    }

    fn set_pt(&mut self, index: usize, pt: Points<f64, Ix0>) {
        if index >= self.len() {
            panic!(
                "Length index {} out of bounds for matrix with {} points",
                index,
                self.len()
            );
        }

        self[index] = pt.0.into_scalar().clone();
    }

    /// Get the number of rows
    fn len_of(&self, axis: Axis) -> usize {
        self.0.len_of(axis)
    }

    /// Get the length
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Get the number of points
    fn npts(&self) -> usize {
        1
    }

    /// Get the shape as (rows, cols)
    fn dim(&self) -> Self::Idx {
        self.0.dim()
    }

    /// Get the dimension
    fn raw_dim(&self) -> Ix1 {
        self.0.raw_dim()
    }

    /// Get the shape as (rows, cols)
    fn shape(&self) -> Self::Idx {
        self.0.dim()
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView1<'_, f64> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut1<'_, f64> {
        self.0.view_mut()
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array1<f64> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array1<f64> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<f64>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>, value: f64) -> Self {
        Points(Array1::from_elem(shape, value))
    }

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&f64) -> f64,
    {
        Points(self.0.map(&f))
    }

    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&f64) -> f64,
    {
        self.0.map_inplace(|x| *x = f(x));
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
    fn approx_eq(a: &ArrayView1<f64>, b: &ArrayView1<f64>, tol: f64) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        let len = a.dim();

        for i in 0..len {
            let diff = &a[i] - &b[i];
            if diff.abs() > tol {
                return false;
            }
        }

        true
    }
}

// Indexing
impl Index<usize> for Points<f64, Ix1> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<[usize; 1]> for Points<f64, Ix1> {
    type Output = f64;

    fn index(&self, index: [usize; 1]) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Points<f64, Ix1> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IndexMut<[usize; 1]> for Points<f64, Ix1> {
    fn index_mut(&mut self, index: [usize; 1]) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Addition implementations
impl Add for Points<f64, Ix1> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

impl Add<&Points<f64, Ix1>> for Points<f64, Ix1> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

impl Add<Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn add(self, other: Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

impl Add<&Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn add(self, other: &Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

// Scalar addition with f64
impl Add<f64> for Points<f64, Ix1> {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn add(self, scalar: f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<&f64> for Points<f64, Ix1> {
    type Output = Self;

    fn add(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<&f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn add(self, scalar: &f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x + scalar))
    }
}

// Subtraction implementations
impl Sub for Points<f64, Ix1> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

impl Sub<&Points<f64, Ix1>> for Points<f64, Ix1> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

impl Sub<Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn sub(self, other: Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

impl Sub<&Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn sub(self, other: &Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

// Scalar subtraction
impl Sub<f64> for Points<f64, Ix1> {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn sub(self, scalar: f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<&f64> for Points<f64, Ix1> {
    type Output = Self;

    fn sub(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x - scalar))
    }
}

impl Sub<&f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn sub(self, scalar: &f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x - scalar))
    }
}

// Element-wise multiplication
impl Mul for Points<f64, Ix1> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

impl Mul<&Points<f64, Ix1>> for Points<f64, Ix1> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

impl Mul<Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn mul(self, other: Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

impl Mul<&Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn mul(self, other: &Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

// Scalar multiplication
impl Mul<f64> for Points<f64, Ix1> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn mul(self, scalar: f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<&f64> for Points<f64, Ix1> {
    type Output = Self;

    fn mul(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<&f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn mul(self, scalar: &f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x * scalar))
    }
}

// Division implementations
impl Div for Points<f64, Ix1> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

impl Div<&Points<f64, Ix1>> for Points<f64, Ix1> {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

impl Div<Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn div(self, other: Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

impl Div<&Points<f64, Ix1>> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn div(self, other: &Points<f64, Ix1>) -> Points<f64, Ix1> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

// Scalar division
impl Div<f64> for Points<f64, Ix1> {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn div(self, scalar: f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<&f64> for Points<f64, Ix1> {
    type Output = Self;

    fn div(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x / scalar))
    }
}

impl Div<&f64> for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn div(self, scalar: &f64) -> Points<f64, Ix1> {
        Points(self.0.map(|x| x / scalar))
    }
}

// Negation
impl Neg for Points<f64, Ix1> {
    type Output = Self;

    fn neg(self) -> Self {
        Points(-self.0)
    }
}

impl Neg for &Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn neg(self) -> Points<f64, Ix1> {
        Points(-&self.0)
    }
}

// Assignment operators
impl AddAssign for Points<f64, Ix1> {
    fn add_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 += &other.0;
    }
}

impl AddAssign<&Points<f64, Ix1>> for Points<f64, Ix1> {
    fn add_assign(&mut self, other: &Points<f64, Ix1>) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 += &other.0;
    }
}

impl SubAssign for Points<f64, Ix1> {
    fn sub_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 -= &other.0;
    }
}

impl SubAssign<&Points<f64, Ix1>> for Points<f64, Ix1> {
    fn sub_assign(&mut self, other: &Points<f64, Ix1>) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 -= &other.0;
    }
}

impl MulAssign for Points<f64, Ix1> {
    fn mul_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 *= &other.0;
    }
}

impl MulAssign<&Points<f64, Ix1>> for Points<f64, Ix1> {
    fn mul_assign(&mut self, other: &Points<f64, Ix1>) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 *= &other.0;
    }
}

impl DivAssign for Points<f64, Ix1> {
    fn div_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 /= &other.0;
    }
}

impl DivAssign<&Points<f64, Ix1>> for Points<f64, Ix1> {
    fn div_assign(&mut self, other: &Points<f64, Ix1>) {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 /= &other.0;
    }
}

// Scalar assignment operators
impl AddAssign<f64> for Points<f64, Ix1> {
    fn add_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x + &scalar);
    }
}

impl AddAssign<&f64> for Points<f64, Ix1> {
    fn add_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x + scalar);
    }
}

impl SubAssign<f64> for Points<f64, Ix1> {
    fn sub_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x - &scalar);
    }
}

impl SubAssign<&f64> for Points<f64, Ix1> {
    fn sub_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x - scalar);
    }
}

impl MulAssign<f64> for Points<f64, Ix1> {
    fn mul_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x * &scalar);
    }
}

impl MulAssign<&f64> for Points<f64, Ix1> {
    fn mul_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x * scalar);
    }
}

impl DivAssign<f64> for Points<f64, Ix1> {
    fn div_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x / &scalar);
    }
}

impl DivAssign<&f64> for Points<f64, Ix1> {
    fn div_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x / scalar);
    }
}

// Dot product implementations
impl Dot<Points<f64, Ix1>> for Points<f64, Ix1> {
    type Output = f64;

    fn dot(&self, rhs: &Points<f64, Ix1>) -> Self::Output {
        self.inner().dot(rhs.inner())
    }
}

impl Dot<Points<f64, Ix2>> for Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn dot(&self, rhs: &Points<f64, Ix2>) -> Self::Output {
        Points(self.inner().dot(rhs.inner()))
    }
}

impl Dot<Points<MyFloat, Ix1>> for Points<f64, Ix1> {
    type Output = f64;

    fn dot(&self, rhs: &Points<MyFloat, Ix1>) -> Self::Output {
        let b: Points<f64, Ix1> = rhs.into();
        self.inner().dot(b.inner())
    }
}

impl Dot<Points<MyFloat, Ix2>> for Points<f64, Ix1> {
    type Output = Points<f64, Ix1>;

    fn dot(&self, rhs: &Points<MyFloat, Ix2>) -> Self::Output {
        let b: Points<f64, Ix2> = rhs.into();
        Points(self.inner().dot(b.inner()))
    }
}

// Traits
impl Default for Points<f64, Ix1> {
    fn default() -> Self {
        Points::<f64, Ix1>::zeros(0)
    }
}

impl PartialEq for Points<f64, Ix1> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl Zero for Points<f64, Ix1> {
    fn zero() -> Self {
        Points::<f64, Ix1>::zeros(0)
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl One for Points<f64, Ix1> {
    fn one() -> Self {
        Points::<f64, Ix1>::ones(0)
    }

    fn is_one(&self) -> bool {
        self.0.iter().all(|x| x.is_one())
    }
}

// Display implementation
impl fmt::Display for Points<f64, Ix1> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() == 0 {
            return write!(f, "[]");
        }

        writeln!(f, "[")?;
        for j in 0..self.len() {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self[j])?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Points<f64, Ix1> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Points({}) {}", self.len(), self)
    }
}

// Conversion traits
impl From<Array1<f64>> for Points<f64, Ix1> {
    fn from(array: Array1<f64>) -> Self {
        Points(array)
    }
}

impl From<Array1<MyFloat>> for Points<f64, Ix1> {
    fn from(array: Array1<MyFloat>) -> Self {
        Points(array).into()
    }
}

impl From<ArrayView1<'_, f64>> for Points<f64, Ix1> {
    fn from(array: ArrayView1<f64>) -> Self {
        Points(array.to_owned())
    }
}

impl From<ArrayView1<'_, MyFloat>> for Points<f64, Ix1> {
    fn from(array: ArrayView1<MyFloat>) -> Self {
        Points(array.to_owned()).into()
    }
}

impl From<Points<f64, Ix1>> for Array1<f64> {
    fn from(matrix: Points<f64, Ix1>) -> Self {
        matrix.0
    }
}

impl From<Vec<f64>> for Points<f64, Ix1> {
    fn from(data: Vec<f64>) -> Self {
        Points::<f64, Ix1>::from_vec_float(data).expect("Invalid matrix data")
    }
}

impl From<&Points<f64, Ix1>> for Points<f64, Ix1> {
    fn from(point: &Points<f64, Ix1>) -> Self {
        point.clone()
    }
}

impl From<Points<MyFloat, Ix1>> for Points<f64, Ix1> {
    fn from(point: Points<MyFloat, Ix1>) -> Self {
        Points::from_shape_fn(point.0.dim(), |j| point[j].to_f64())
    }
}

impl From<&Points<MyFloat, Ix1>> for Points<f64, Ix1> {
    fn from(point: &Points<MyFloat, Ix1>) -> Self {
        Points::from_shape_fn(point.0.dim(), |j| point[j].to_f64())
    }
}

#[cfg(test)]
mod points_ix1_f64_tests {
    use super::*;

    #[test]
    fn test_creation() {
        let zeros = Points::<f64, Ix1>::zeros(2);
        assert_eq!(zeros.shape(), 2);
        assert!(zeros[0].is_zero());
        assert!(zeros[1].is_zero());

        let ones = Points::<f64, Ix1>::ones(3);
        assert_eq!(ones.shape(), 3);
        assert!(ones[0].is_one());
        assert!(ones[2].is_one());
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = Points::<f64, Ix1>::from_vec(data).unwrap();

        assert_eq!(matrix.shape(), 3);
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        assert_eq!(matrix[2], 3.0);
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = Points::<f64, Ix1>::from_vec_f64(data).unwrap();

        assert_eq!(matrix.shape(), 3);
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        assert_eq!(matrix[2], 3.0);
    }

    #[test]
    fn test_from_vec_float() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = Points::<f64, Ix1>::from_vec_float(data).unwrap();

        assert_eq!(matrix.shape(), 3);
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        assert_eq!(matrix[2], 3.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        let b = Points::<f64, Ix1>::from_vec_float(vec![5.0, 6.0]).unwrap();

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[0], 6.0);
        assert_eq!(sum[1], 8.0);

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[0], 4.0);
        assert_eq!(diff[1], 4.0);

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[0], 5.0);
        assert_eq!(prod[1], 12.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        let b = Points::<f64, Ix1>::from_vec_float(vec![5.0, 6.0]).unwrap();

        let result = a.dot(&b);

        // Expected: [1*5+2*6] = 17
        assert_eq!(result, 17.0);
    }

    #[test]
    fn test_scalar_operations() {
        let matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[0], 6.0);
        assert_eq!(added[1], 7.0);

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[0], 2.0);
        assert_eq!(multiplied[1], 4.0);
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        let other = Points::<f64, Ix1>::from_vec_float(vec![5.0, 6.0]).unwrap();

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[0], 6.0);
        assert_eq!(matrix[1], 8.0);

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[0], 12.0);
        assert_eq!(matrix[1], 16.0);
    }

    #[test]
    fn test_map_functions() {
        let matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[0], 1.0);
        assert_eq!(squared[1], 4.0);

        // Original should be unchanged
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * 2.0);
        assert_eq!(matrix_mut[0], 2.0);
        assert_eq!(matrix_mut[1], 4.0);
    }

    #[test]
    fn test_display() {
        let matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        let display_str = format!("{}", matrix);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
    }

    #[test]
    fn test_equality() {
        let matrix1 = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        let matrix2 = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]).unwrap();

        let matrix3 = Points::<f64, Ix1>::from_vec_float(vec![1.0, 3.0]).unwrap();

        assert_eq!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
    }

    #[test]
    fn test_error_cases() {
        // Test mismatched dimensions for addition
        let matrix1 = Points::<f64, Ix1>::zeros(2);
        let matrix2 = Points::<f64, Ix1>::zeros(3);

        std::panic::catch_unwind(|| {
            let _ = &matrix1 + &matrix2;
        })
        .expect_err("Should panic on dimension mismatch");

        // Test incompatible matrix multiplication
        std::panic::catch_unwind(|| {
            let _ = matrix1.dot(&matrix2);
        })
        .expect_err("Should panic on incompatible multiplication");
    }

    #[test]
    fn test_conversions() {
        let data = vec![1.0, 2.0];

        // Test From<Vec<f64>>
        let matrix: Points<f64, Ix1> = data.into();
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);

        // Test conversion to Array1
        let array: Array1<f64> = matrix.into();
        assert_eq!(array[0], 1.0);
        assert_eq!(array[1], 2.0);
    }
}
