use crate::{
    num::{RFComplex, RFFloat, RFNum},
    pts::{Matrix, Points, Pts},
};
use ndarray::{IntoDimension, SliceArg, linalg::Dot, prelude::*};
use num::complex::Complex64;
use num_traits::{One, Zero};
use std::fmt;

// pub mod c64;
// pub mod f64;
// pub mod mycomplex;
// pub mod myfloat;

/// A matrix wrapper around ndarray::Array1 with RFNum elements
impl<T> Points<T, Ix1>
where
    T: RFNum,
{
    fn check_vec<U>(vec: &Vec<U>) -> usize {
        vec.len()
    }

    /// Create a matrix from a 1D vector of f64 values
    pub fn from_vec(data: Vec<T>) -> Self {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, value) in data.iter().enumerate() {
            matrix[i] = value.clone();
        }

        matrix
    }

    /// Create a matrix from a 1D vector of f64 values
    pub fn from_real_vec(data: Vec<T::Real>) -> Self {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, value) in data.iter().enumerate() {
            matrix[i] = <<T as RFNum>::Real as Into<T>>::into(value.clone());
        }

        matrix
    }

    /// Create a one-dimensional Points from an iterator or iterable.
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    ///
    /// ```rust
    /// use rfkit::prelude::*;
    ///
    /// let points = Points::from_iter((0..10).map(|x| x as f64));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Self {
        Self::from_vec(iterable.into_iter().collect())
    }
}

/// A matrix wrapper around ndarray::Array1 with RFNum elements
impl<T> Points<T, Ix1>
where
    T: RFFloat,
{
    /// Create a matrix from a 1D vector of f64 values
    pub fn from_vec_f64(data: Vec<f64>) -> Self {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, &value) in data.iter().enumerate() {
            matrix[i] = T::from(value);
        }

        matrix
    }

    /// Create a matrix from a 1D vector of f64 values
    pub fn from_vec_float<U>(data: Vec<U>) -> Self
    where
        T: From<U>,
        U: RFFloat,
    {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, value) in data.iter().enumerate() {
            matrix[i] = T::from(value.clone());
        }

        matrix
    }
}

/// A matrix wrapper around ndarray::Array1 with RFNum elements
impl<T> Points<T, Ix1>
where
    T: RFComplex,
{
    /// Create a matrix from a 1D vector of complex tuples (real, imag)
    pub fn from_vec_c64(data: Vec<Complex64>) -> Self {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, &value) in data.iter().enumerate() {
            matrix[i] = T::from(value);
        }

        matrix
    }

    /// Create a matrix from a 1D vector of complex tuples (real, imag)
    pub fn from_vec_complex<U>(data: Vec<U>) -> Self
    where
        for<'a> T: From<&'a U>,
        U: RFComplex,
    {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, value) in data.iter().enumerate() {
            matrix[i] = T::from(value);
        }

        matrix
    }

    /// Create a matrix from a 1D vector of float tuples (real, imag)
    pub fn from_vec_tuple<U>(data: Vec<(U, U)>) -> Self
    where
        T::Real: From<U>,
        U: RFFloat,
    {
        if data.is_empty() {
            return Self::new(Array1::from_elem(0, T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, (real, imag)) in data.iter().enumerate() {
            matrix[i] = T::new(
                &<T as RFNum>::Real::from(real.clone()),
                &<T as RFNum>::Real::from(imag.clone()),
            );
        }

        matrix
    }
}

impl<T> Pts<T, Ix1> for Points<T, Ix1>
where
    T: RFNum,
{
    type Idx = usize;
    type Tuple<'a>
        = (T::Real, T::Real)
    where
        T: 'a;

    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>) -> Self {
        Points(Array1::from_elem(shape, T::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>) -> Self {
        Points(Array1::from_elem(shape, T::one()))
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
        for (idx, &value) in data.iter().enumerate() {
            matrix[idx] = T::from(value);
        }

        Ok(matrix)
    }

    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>, f: F) -> Self
    where
        F: Fn(usize) -> T,
    {
        Points(Array1::<T>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(
        shape: impl IntoDimension<Dim = Dim<[usize; 1]>>,
        v: Vec<T>,
    ) -> Result<Self, &'static str> {
        match Array1::<T>::from_shape_vec(shape, v) {
            Ok(x) => Ok(Points(x)),
            Err(_) => Err("Cannot create vector"),
        }
    }

    fn first(&self) -> Option<&T> {
        self.0.first()
    }

    fn last(&self) -> Option<&T> {
        self.0.last()
    }

    fn iter(&self) -> ndarray::iter::Iter<'_, T, Ix1> {
        self.0.iter()
    }

    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix1> {
        self.0.iter_mut()
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix0> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(&mut self, axis: Axis) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix0> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, T, ndarray::Ix1> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, T, ndarray::Ix1> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix0> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix0> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, T, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix1>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, T, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix1>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<T, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, T, ndarray::Ix0>,
    ) -> Result<(), ndarray::ShapeError>
    where
        T: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get and set outer axis point
    fn pt<I>(&self, index: usize) -> ndarray::ArrayView<'_, T, I::OutDim>
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

    fn pt_mut<I>(&mut self, index: usize) -> ndarray::ArrayViewMut<'_, T, I::OutDim>
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

    fn set_pt(&mut self, index: usize, pt: Points<T, Ix0>) {
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

    /// Get the shape as len
    fn dim(&self) -> usize {
        self.0.dim()
    }

    /// Get the dimension
    fn raw_dim(&self) -> Ix1 {
        self.0.raw_dim()
    }

    /// Get the shape as len
    fn shape(&self) -> usize {
        self.dim()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView1<'_, T> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut1<'_, T> {
        self.0.view_mut()
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array1<T> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array1<T> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<T>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(shape: impl IntoDimension<Dim = Dim<[usize; 1]>>, value: T) -> Self {
        Points(Array1::from_elem(shape, value))
    }

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        Points(self.0.map(&f))
    }

    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
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
    fn approx_eq(a: &ArrayView1<T>, b: &ArrayView1<T>, tol: f64) -> bool
    where
        <T as RFNum>::Real: PartialOrd<f64>,
        for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    {
        if a.dim() != b.dim() {
            return false;
        }

        let len = a.dim();

        for i in 0..len {
            let diff = &a[i] - &b[i];
            if diff.norm() > tol {
                return false;
            }
        }

        true
    }

    // /// Calculate the L1 norm (Manhattan norm)
    // fn norm_l1(&self) -> Array0<T::Real>
    // where
    //     T::Real: RFFloat,
    // {
    //     let mut sum = T::Real::zero();
    //     for i in 0..self.len() {
    //         sum += self[i].norm();
    //     }
    //     Array0::from_elem((), sum)
    // }

    // /// Calculate the L2 norm (Euclidean/Frobenius norm)
    // fn norm_l2(&self) -> Array0<T::Real>
    // where
    //     T::Real: RFFloat + AddAssign<T>,
    //     for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    // {
    //     let mut sum = T::Real::zero();
    //     for i in 0..self.len() {
    //         sum += &self[i] * &self[i];
    //     }
    //     Array0::from_elem((), sum.sqrt())
    // }

    // /// Calculate the infinite norm
    // fn norm_inf(&self) -> Array0<T::Real>
    // where
    //     T::Real: RFFloat,
    // {
    //     let mut max = T::Real::zero();
    //     for i in 0..self.len() {
    //         let abs_val = self[i].norm();
    //         if abs_val > max {
    //             max = abs_val;
    //         }
    //     }
    //     Array0::from_elem((), max)
    // }
}

// Dot product implementations
impl<T, U> Dot<Points<U, Ix1>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = T;

    fn dot(&self, rhs: &Points<U, Ix1>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}",
                self.len(),
                rhs.len(),
            );
        }

        let mut result = Self::Output::zero();

        for i in 0..self.len() {
            result += &self[i] * &rhs[i];
        }

        result
    }
}

impl<T, U> Dot<&Points<U, Ix1>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = T;

    fn dot(&self, rhs: &&Points<U, Ix1>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}",
                self.len(),
                rhs.len(),
            );
        }

        let mut result = Self::Output::zero();

        for i in 0..self.len() {
            result += &self[i] * &rhs[i];
        }

        result
    }
}

impl<T, U> Dot<Points<U, Ix2>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = Points<T, Ix1>;

    fn dot(&self, rhs: &Points<U, Ix2>) -> Self::Output {
        if self.len() != rhs.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}x{}",
                self.len(),
                rhs.nrows(),
                rhs.ncols()
            );
        }

        let mut result = Points::zeros(rhs.ncols());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                result[j] += &self[i] * &rhs[[i, j]];
            }
        }

        result
    }
}

impl<T, U> Dot<Array1<U>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = T;

    fn dot(&self, rhs: &Array1<U>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}",
                self.len(),
                rhs.len(),
            );
        }

        let mut result = Self::Output::zero();

        for i in 0..self.len() {
            result += &self[i] * &rhs[i];
        }

        result
    }
}

impl<T, U> Dot<Array2<U>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = Points<T, Ix1>;

    fn dot(&self, rhs: &Array2<U>) -> Self::Output {
        if self.len() != rhs.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}x{}",
                self.len(),
                rhs.nrows(),
                rhs.ncols()
            );
        }

        let mut result = Points::zeros(rhs.ncols());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                result[j] += &self[i] * &rhs[[i, j]];
            }
        }

        result
    }
}

impl<T, U> Dot<ArrayView1<'_, U>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = T;

    fn dot(&self, rhs: &ArrayView1<'_, U>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}",
                self.len(),
                rhs.len(),
            );
        }

        let mut result = Self::Output::zero();

        for i in 0..self.len() {
            result += &self[i] * &rhs[i];
        }

        result
    }
}

impl<T, U> Dot<ArrayView2<'_, U>> for Points<T, Ix1>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = Points<T, Ix1>;

    fn dot(&self, rhs: &ArrayView2<'_, U>) -> Self::Output {
        if self.len() != rhs.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}x{}",
                self.len(),
                rhs.nrows(),
                rhs.ncols()
            );
        }

        let mut result = Points::zeros(rhs.ncols());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                result[j] += &self[i] * &rhs[[i, j]];
            }
        }

        result
    }
}

impl<T, U> Dot<Points<U, Ix1>> for Array1<T>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = T;

    fn dot(&self, rhs: &Points<U, Ix1>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}",
                self.len(),
                rhs.len(),
            );
        }

        let mut result = Self::Output::zero();

        for i in 0..self.len() {
            result += &self[i] * &rhs[i];
        }

        result
    }
}

impl<T, U> Dot<Points<U, Ix2>> for Array1<T>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = Array1<T>;

    fn dot(&self, rhs: &Points<U, Ix2>) -> Self::Output {
        if self.len() != rhs.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}x{}",
                self.len(),
                rhs.nrows(),
                rhs.ncols()
            );
        }

        let mut result = Array1::zeros(rhs.ncols());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                result[j] += &self[i] * &rhs[[i, j]];
            }
        }

        result
    }
}

impl<T, U> Dot<Points<U, Ix1>> for ArrayView1<'_, T>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = T;

    fn dot(&self, rhs: &Points<U, Ix1>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}",
                self.len(),
                rhs.len(),
            );
        }

        let mut result = Self::Output::zero();

        for i in 0..self.len() {
            result += &self[i] * &rhs[i];
        }

        result
    }
}

impl<T, U> Dot<Points<U, Ix2>> for ArrayView1<'_, T>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = Array1<T>;

    fn dot(&self, rhs: &Points<U, Ix2>) -> Self::Output {
        if self.len() != rhs.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {} * {}x{}",
                self.len(),
                rhs.nrows(),
                rhs.ncols()
            );
        }

        let mut result = Array1::zeros(rhs.ncols());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                result[j] += &self[i] * &rhs[[i, j]];
            }
        }

        result
    }
}

// Traits
impl<T> Default for Points<T, Ix1>
where
    T: RFNum,
{
    fn default() -> Self {
        Points::zeros(0)
    }
}

impl<T> Zero for Points<T, Ix1>
where
    T: RFNum,
{
    fn zero() -> Self {
        Points::zeros(0)
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl<T> One for Points<T, Ix1>
where
    T: RFNum,
{
    fn one() -> Self {
        Points::ones(0)
    }

    fn is_one(&self) -> bool {
        self.0.iter().all(|x| x.is_one())
    }
}

// Display implementation
impl<T> fmt::Display for Points<T, Ix1>
where
    T: RFNum,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() == 0 {
            return write!(f, "[]");
        }

        writeln!(f, "[")?;
        for i in 0..self.len() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self[i])?;
        }
        write!(f, "]")
    }
}

impl<T> fmt::Debug for Points<T, Ix1>
where
    T: RFNum,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Points({}) {}", self.len(), self)
    }
}

// Conversion traits
impl<T> From<Array1<T>> for Points<T, Ix1>
where
    T: RFNum,
{
    fn from(array: Array1<T>) -> Self {
        Points(array)
    }
}

impl<T> From<ArrayRef1<T>> for Points<T, Ix1>
where
    T: RFNum,
    ArrayRef1<T>: Sized,
{
    fn from(array: ArrayRef1<T>) -> Self {
        Points(array.to_owned())
    }
}

impl<T> From<ArrayView1<'_, T>> for Points<T, Ix1>
where
    T: RFNum,
{
    fn from(array: ArrayView1<T>) -> Self {
        Points(array.to_owned())
    }
}

impl<T> From<Points<T, Ix1>> for Array1<T> {
    fn from(matrix: Points<T, Ix1>) -> Self {
        matrix.0
    }
}

impl<T> From<Vec<T>> for Points<T, Ix1>
where
    T: RFNum,
{
    fn from(data: Vec<T>) -> Self {
        Points::<T, Ix1>::from_vec(data)
    }
}

impl<T> From<Vec<(T::Real, T::Real)>> for Points<T, Ix1>
where
    T: RFComplex,
    <T as RFNum>::Real: RFFloat,
{
    fn from(data: Vec<(T::Real, T::Real)>) -> Self {
        Points::<T, Ix1>::from_vec_tuple(data)
    }
}

// impl<T> From<&Points<T, Ix1>> for Points<T, Ix1>
// where
//     T: RFNum,
// {
//     fn from(point: &Points<T, Ix1>) -> Self {
//         point.clone()
//     }
// }

impl<T, U> From<&Points<U, Ix1>> for Points<T, Ix1>
where
    T: RFNum + std::convert::From<U>,
    U: RFNum,
{
    fn from(point: &Points<U, Ix1>) -> Self {
        Points::<T, Ix1>::from_shape_fn(point.shape(), |i| point[i].clone().into())
    }
}

impl<T> FromIterator<T> for Points<T, Ix1>
where
    T: RFNum,
{
    fn from_iter<I>(iterable: I) -> Points<T, Ix1>
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_iter(iterable)
    }
}

impl<T> IntoIterator for Points<T, Ix1>
where
    T: RFNum,
{
    type Item = T;
    type IntoIter = ndarray::iter::IntoIter<T, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// For borrowing iteration
impl<'a, T> IntoIterator for &'a Points<T, Ix1>
where
    T: RFNum,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// For mutable borrowing iteration
impl<'a, T> IntoIterator for &'a mut Points<T, Ix1>
where
    T: RFNum,
{
    type Item = &'a mut T;
    type IntoIter = ndarray::iter::IterMut<'a, T, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

#[cfg(test)]
mod points_ix1_f64_tests {
    use super::*;
    use crate::util::{comp_num, comp_pts_ix1};
    use float_cmp::F64Margin;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn test_creation() {
        let zeros = Points::<f64, Ix1>::zeros(2);
        assert_eq!(zeros.shape(), 2);
        assert!(zeros[0].is_zero());
        assert!(zeros[1].is_zero());
        comp_pts_ix1(&array![0.0, 0.0].into(), &zeros, MARGIN, "zeros()");

        let ones = Points::<f64, Ix1>::ones(3);
        assert_eq!(ones.shape(), 3);
        assert!(ones[0].is_one());
        assert!(ones[2].is_one());
        comp_pts_ix1(&array![1.0, 1.0, 1.0].into(), &ones, MARGIN, "ones()");
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = Points::<f64, Ix1>::from_vec(data);

        assert_eq!(matrix.shape(), 3);
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        assert_eq!(matrix[2], 3.0);
        comp_pts_ix1(&array![1.0, 2.0, 3.0].into(), &matrix, MARGIN, "from_vec()");
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = Points::<f64, Ix1>::from_vec_f64(data);

        assert_eq!(matrix.shape(), 3);
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        assert_eq!(matrix[2], 3.0);
        comp_pts_ix1(
            &array![1.0, 2.0, 3.0].into(),
            &matrix,
            MARGIN,
            "from_vec_f64()",
        );
    }

    #[test]
    fn test_from_vec_float() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = Points::<f64, Ix1>::from_vec_float(data);

        assert_eq!(matrix.shape(), 3);
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        assert_eq!(matrix[2], 3.0);
        comp_pts_ix1(
            &array![1.0, 2.0, 3.0].into(),
            &matrix,
            MARGIN,
            "from_vec_float()",
        );
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        let b = Points::<f64, Ix1>::from_vec_float(vec![5.0, 6.0]);

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[0], 6.0);
        assert_eq!(sum[1], 8.0);
        comp_pts_ix1(&array![6.0, 8.0].into(), &sum, MARGIN, "&a + &b");

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[0], 4.0);
        assert_eq!(diff[1], 4.0);
        comp_pts_ix1(&array![4.0, 4.0].into(), &diff, MARGIN, "&a - &b");

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[0], 5.0);
        assert_eq!(prod[1], 12.0);
        comp_pts_ix1(&array![5.0, 12.0].into(), &prod, MARGIN, "&a * &b");
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        let b = Points::<f64, Ix1>::from_vec_float(vec![5.0, 6.0]);

        let result = a.dot(&b);

        // Expected: [1*5+2*6] = 17
        assert_eq!(result, 17.0);
        comp_num(&17.0, &result, MARGIN, "a.dot(b)", "");
    }

    #[test]
    fn test_scalar_operations() {
        let matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[0], 6.0);
        assert_eq!(added[1], 7.0);
        comp_pts_ix1(&array![6.0, 7.0].into(), &added, MARGIN, "&matrix + 5.0");

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[0], 2.0);
        assert_eq!(multiplied[1], 4.0);
        comp_pts_ix1(
            &array![2.0, 4.0].into(),
            &multiplied,
            MARGIN,
            "&matrix + 2.0",
        );
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        let other = Points::<f64, Ix1>::from_vec_float(vec![5.0, 6.0]);

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[0], 6.0);
        assert_eq!(matrix[1], 8.0);
        comp_pts_ix1(
            &array![6.0, 8.0].into(),
            &matrix,
            MARGIN,
            "&matrix += &other",
        );

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[0], 12.0);
        assert_eq!(matrix[1], 16.0);
        comp_pts_ix1(
            &array![12.0, 16.0].into(),
            &matrix,
            MARGIN,
            "&matrix *= 2.0",
        );
    }

    #[test]
    fn test_map_functions() {
        let matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[0], 1.0);
        assert_eq!(squared[1], 4.0);
        comp_pts_ix1(&array![1.0, 4.0].into(), &squared, MARGIN, "map()");

        // Original should be unchanged
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 2.0);
        comp_pts_ix1(&array![1.0, 2.0].into(), &matrix, MARGIN, "map(orig)");

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * 2.0);
        assert_eq!(matrix_mut[0], 2.0);
        assert_eq!(matrix_mut[1], 4.0);
        comp_pts_ix1(
            &array![2.0, 4.0].into(),
            &matrix_mut,
            MARGIN,
            "map_inplace()",
        );
    }

    #[test]
    fn test_display() {
        let matrix = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        let display_str = format!("{}", matrix);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
    }

    #[test]
    fn test_equality() {
        let matrix1 = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        let matrix2 = Points::<f64, Ix1>::from_vec_float(vec![1.0, 2.0]);

        let matrix3 = Points::<f64, Ix1>::from_vec_float(vec![1.0, 3.0]);

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
        comp_pts_ix1(&array![1.0, 2.0].into(), &matrix, MARGIN, "into()");

        // Test conversion to Array1
        let array: Array1<f64> = matrix.into();
        assert_eq!(array[0], 1.0);
        assert_eq!(array[1], 2.0);
    }
}
