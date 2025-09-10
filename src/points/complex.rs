use crate::points::Pts;
use ndarray::{Array, Array3, ArrayView3, ArrayViewMut3, Axis, Ix3, SliceArg};
use num::complex::Complex64;
use num_traits::{One, Zero};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A matrix wrapper around ndarray::Array3 with Complex64 elements
pub struct Points(Array3<Complex64>);

impl Points {
    pub fn new(array: Array3<Complex64>) -> Self {
        Points(array)
    }
}

impl Pts<Complex64, f64> for Points {
    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: (usize, usize, usize)) -> Self {
        Points(Array3::from_elem(shape, Complex64::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: (usize, usize, usize)) -> Self {
        Points(Array3::from_elem(shape, Complex64::one()))
    }

    /// Create an identity matrix of given size
    fn eye(size: (usize, usize)) -> Self {
        Points(Array3::from_shape_fn(
            (size.0, size.1, size.1),
            |(_, j, k)| {
                if j == k {
                    Complex64::ONE
                } else {
                    Complex64::ZERO
                }
            },
        ))
    }

    /// Create a matrix from a 3D vector of f64 values
    fn from_vec_f64(data: Vec<Vec<Vec<f64>>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let len = data.len();
        let rows = data[0].len();
        let cols = data[0][0].len();

        if cols == 0 {
            return Err("Cannot create matrix with zero columns");
        }

        // // Check all rows have the same length
        // if !data.iter().all(|row| row.len() == cols) {
        //     return Err("All rows must have the same length");
        // }

        let mut matrix = Self::zeros((len, rows, cols));
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, &value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = Complex64::new(value, 0.0);
                }
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<Vec<(f64, f64)>>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let len = data.len();
        let rows = data[0].len();
        let cols = data[0][0].len();

        if cols == 0 {
            return Err("Cannot create matrix with zero columns");
        }

        // if !data.iter().all(|row| row.len() == cols) {
        //     return Err("All rows must have the same length");
        // }

        let mut matrix = Self::zeros((len, rows, cols));
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, &(real, imag)) in row.iter().enumerate() {
                    matrix[(i, j, k)] = Complex64::new(real, imag);
                }
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        len: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, &'static str> {
        if data.len() != len * rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        let mut matrix = Self::zeros((len, rows, cols));
        for (idx, &value) in data.iter().enumerate() {
            let i = idx / (rows * cols);
            let j = idx / cols;
            let k = idx % cols;
            matrix[(i, j, k)] = Complex64::new(value, 0.0);
        }

        Ok(matrix)
    }

    fn from_shape_fn<F>(shape: (usize, usize, usize), f: F) -> Self
    where
        F: Fn((usize, usize, usize)) -> Complex64,
    {
        Points(Array3::<Complex64>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(
        shape: (usize, usize, usize),
        v: Vec<Complex64>,
    ) -> Result<Self, &'static str> {
        match Array3::<Complex64>::from_shape_vec(shape, v) {
            Ok(x) => Ok(Points(x)),
            Err(_) => Err("Cannot create vector"),
        }
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, Complex64, ndarray::Ix2> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(
        &mut self,
        axis: Axis,
    ) -> ndarray::iter::AxisIterMut<'_, Complex64, ndarray::Ix2> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, Complex64, ndarray::Ix3> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, Complex64, ndarray::Ix3> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, Complex64, ndarray::Ix2> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, Complex64, ndarray::Ix2> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, Complex64, I::OutDim>
    where
        I: SliceArg<Ix3>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, Complex64, I::OutDim>
    where
        I: SliceArg<Ix3>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<Complex64, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, Complex64, ndarray::Ix2>,
    ) -> Result<(), ndarray::ShapeError>
    where
        Complex64: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get the number of rows
    fn len_of(&self, axis: Axis) -> usize {
        self.0.len_of(axis)
    }

    /// Get the shape as (rows, cols)
    fn dim(&self) -> (usize, usize, usize) {
        self.0.dim()
    }

    /// Get the shape as (rows, cols)
    fn shape(&self) -> (usize, usize, usize) {
        let shape = self.0.dim();
        (shape.0, shape.1, shape.2)
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView3<Complex64> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut3<Complex64> {
        self.0.view_mut()
    }

    /// Element-wise conjugate
    fn conj(&self) -> Self {
        let mut result = self.clone();
        for element in result.0.iter_mut() {
            *element = element.conj();
        }
        result
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array3<Complex64> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array3<Complex64> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<Complex64>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(len: usize, rows: usize, cols: usize, value: Complex64) -> Self {
        Points(Array3::from_elem((len, rows, cols), value))
    }

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&Complex64) -> Complex64,
    {
        Points(self.0.map(&f))
    }

    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&Complex64) -> Complex64,
    {
        self.0.map_inplace(|x| *x = f(x));
    }
}

// Indexing
impl Index<(usize, usize, usize)> for Points {
    type Output = Complex64;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<[usize; 3]> for Points {
    type Output = Complex64;

    fn index(&self, index: [usize; 3]) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<(usize, usize, usize)> for Points {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IndexMut<[usize; 3]> for Points {
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Addition implementations
impl Add for Points {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

impl Add<&Points> for Points {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

impl Add<Points> for &Points {
    type Output = Points;

    fn add(self, other: Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

impl Add<&Points> for &Points {
    type Output = Points;

    fn add(self, other: &Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 + &other.0)
    }
}

// Scalar addition with Complex64
impl Add<Complex64> for Points {
    type Output = Self;

    fn add(self, scalar: Complex64) -> Self {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<Complex64> for &Points {
    type Output = Points;

    fn add(self, scalar: Complex64) -> Points {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<&Complex64> for Points {
    type Output = Self;

    fn add(self, scalar: &Complex64) -> Self {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<&Complex64> for &Points {
    type Output = Points;

    fn add(self, scalar: &Complex64) -> Points {
        Points(self.0.map(|x| x + scalar))
    }
}

// Scalar addition with f64
impl Add<f64> for Points {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self + scalar_complex
    }
}

impl Add<f64> for &Points {
    type Output = Points;

    fn add(self, scalar: f64) -> Points {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self + scalar_complex
    }
}

// Subtraction implementations
impl Sub for Points {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

impl Sub<&Points> for Points {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

impl Sub<Points> for &Points {
    type Output = Points;

    fn sub(self, other: Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

impl Sub<&Points> for &Points {
    type Output = Points;

    fn sub(self, other: &Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 - &other.0)
    }
}

// Scalar subtraction
impl Sub<Complex64> for Points {
    type Output = Self;

    fn sub(self, scalar: Complex64) -> Self {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<f64> for Points {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self - scalar_complex
    }
}

// Element-wise multiplication
impl Mul for Points {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

impl Mul<&Points> for Points {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

impl Mul<Points> for &Points {
    type Output = Points;

    fn mul(self, other: Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

impl Mul<&Points> for &Points {
    type Output = Points;

    fn mul(self, other: &Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 * &other.0)
    }
}

// Scalar multiplication
impl Mul<Complex64> for Points {
    type Output = Self;

    fn mul(self, scalar: Complex64) -> Self {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<Complex64> for &Points {
    type Output = Points;

    fn mul(self, scalar: Complex64) -> Points {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<&Complex64> for Points {
    type Output = Self;

    fn mul(self, scalar: &Complex64) -> Self {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<&Complex64> for &Points {
    type Output = Points;

    fn mul(self, scalar: &Complex64) -> Points {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<f64> for Points {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self * scalar_complex
    }
}

impl Mul<f64> for &Points {
    type Output = Points;

    fn mul(self, scalar: f64) -> Points {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self * scalar_complex
    }
}

// Division implementations
impl Div for Points {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

impl Div<&Points> for Points {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

impl Div<Points> for &Points {
    type Output = Points;

    fn div(self, other: Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

impl Div<&Points> for &Points {
    type Output = Points;

    fn div(self, other: &Points) -> Points {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Points(&self.0 / &other.0)
    }
}

// Scalar division
impl Div<Complex64> for Points {
    type Output = Self;

    fn div(self, scalar: Complex64) -> Self {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<f64> for Points {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self / scalar_complex
    }
}

// Negation
impl Neg for Points {
    type Output = Self;

    fn neg(self) -> Self {
        Points(-self.0)
    }
}

impl Neg for &Points {
    type Output = Points;

    fn neg(self) -> Points {
        Points(-&self.0)
    }
}

// Assignment operators
impl AddAssign for Points {
    fn add_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 += &other.0;
    }
}

impl AddAssign<&Points> for Points {
    fn add_assign(&mut self, other: &Points) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 += &other.0;
    }
}

impl SubAssign for Points {
    fn sub_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 -= &other.0;
    }
}

impl SubAssign<&Points> for Points {
    fn sub_assign(&mut self, other: &Points) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 -= &other.0;
    }
}

impl MulAssign for Points {
    fn mul_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 *= &other.0;
    }
}

impl MulAssign<&Points> for Points {
    fn mul_assign(&mut self, other: &Points) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 *= &other.0;
    }
}

impl DivAssign for Points {
    fn div_assign(&mut self, other: Self) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 /= &other.0;
    }
}

impl DivAssign<&Points> for Points {
    fn div_assign(&mut self, other: &Points) {
        if self.shape() != other.shape() {
            panic!(
                "Points dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.0 /= &other.0;
    }
}

// Scalar assignment operators
impl AddAssign<Complex64> for Points {
    fn add_assign(&mut self, scalar: Complex64) {
        self.0.map_inplace(|x| *x = &*x + &scalar);
    }
}

impl AddAssign<f64> for Points {
    fn add_assign(&mut self, scalar: f64) {
        let scalar_complex = Complex64::new(scalar, 0.0);
        *self += scalar_complex;
    }
}

impl MulAssign<Complex64> for Points {
    fn mul_assign(&mut self, scalar: Complex64) {
        self.0.map_inplace(|x| *x = &*x * &scalar);
    }
}

impl MulAssign<f64> for Points {
    fn mul_assign(&mut self, scalar: f64) {
        let scalar_complex = Complex64::new(scalar, 0.0);
        *self *= scalar_complex;
    }
}

// Traits
impl Clone for Points {
    fn clone(&self) -> Self {
        Points(self.0.clone())
    }
}

impl Default for Points {
    fn default() -> Self {
        Points::zeros((0, 0, 0))
    }
}

impl PartialEq for Points {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl Zero for Points {
    fn zero() -> Self {
        Points::zeros((0, 0, 0))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

// Display implementation
impl fmt::Display for Points {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len_of(Axis(0)) == 0 || self.len_of(Axis(1)) == 0 || self.len_of(Axis(2)) == 0 {
            return write!(f, "[]");
        }

        writeln!(f, "[")?;
        for i in 0..self.len_of(Axis(0)) {
            write!(f, "  [")?;
            for j in 0..self.len_of(Axis(1)) {
                write!(f, "  [")?;
                for k in 0..self.len_of(Axis(2)) {
                    if k > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self[(i, j, k)])?;
                }
                if j < self.len_of(Axis(1)) - 1 {
                    writeln!(f, "],")?;
                } else {
                    writeln!(f, "]")?;
                }
            }
            if i < self.len_of(Axis(0)) - 1 {
                writeln!(f, "],")?;
            } else {
                writeln!(f, "]")?;
            }
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Points {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Points({}x{}x{}) {}",
            self.len_of(Axis(0)),
            self.len_of(Axis(1)),
            self.len_of(Axis(2)),
            self
        )
    }
}

// Conversion traits
impl From<Array3<Complex64>> for Points {
    fn from(array: Array3<Complex64>) -> Self {
        Points(array)
    }
}

impl From<ArrayView3<'_, Complex64>> for Points {
    fn from(array: ArrayView3<Complex64>) -> Self {
        Points(array.to_owned())
    }
}

impl From<Points> for Array3<Complex64> {
    fn from(matrix: Points) -> Self {
        matrix.0
    }
}

impl From<Vec<Vec<Vec<f64>>>> for Points {
    fn from(data: Vec<Vec<Vec<f64>>>) -> Self {
        Points::from_vec_f64(data).expect("Invalid matrix data")
    }
}

impl From<Vec<Vec<Vec<(f64, f64)>>>> for Points {
    fn from(data: Vec<Vec<Vec<(f64, f64)>>>) -> Self {
        Points::from_vec_complex(data).expect("Invalid matrix data")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let zeros = Points::zeros((2, 3, 4));
        assert_eq!(zeros.shape(), (2, 3, 4));
        assert!(zeros[(0, 0, 0)].is_zero());
        assert!(zeros[(1, 2, 3)].is_zero());

        let ones = Points::ones((3, 2, 1));
        assert_eq!(ones.shape(), (3, 2, 1));
        assert!(ones[(0, 0, 0)].is_one());
        assert!(ones[(2, 1, 0)].is_one());
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
            ],
            vec![
                vec![13.0, 14.0, 15.0, 16.0],
                vec![17.0, 18.0, 19.0, 20.0],
                vec![21.0, 22.0, 23.0, 24.0],
            ],
        ];
        let matrix = Points::from_vec_f64(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3, 4));
        assert_eq!(matrix[(0, 0, 0)].re, 1.0);
        assert_eq!(matrix[(0, 1, 2)].re, 7.0);
        assert_eq!(matrix[(0, 2, 3)].re, 12.0);
        assert_eq!(matrix[(1, 0, 0)].re, 13.0);
        assert_eq!(matrix[(1, 1, 1)].re, 18.0);
        assert_eq!(matrix[(1, 2, 3)].re, 24.0);
    }

    #[test]
    fn test_from_vec_complex() {
        let data = vec![
            vec![vec![(1.0, 2.0), (3.0, 4.0)], vec![(5.0, 6.0), (7.0, 8.0)]],
            vec![
                vec![(9.0, 10.0), (11.0, 12.0)],
                vec![(13.0, 14.0), (15.0, 16.0)],
            ],
        ];
        let matrix = Points::from_vec_complex(data).unwrap();

        assert_eq!(matrix.shape(), (2, 2, 2));
        assert_eq!(matrix[(0, 0, 0)].re, 1.0);
        assert_eq!(matrix[(0, 0, 0)].im, 2.0);
        assert_eq!(matrix[(1, 1, 1)].re, 15.0);
        assert_eq!(matrix[(1, 1, 1)].im, 16.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let b = Points::from_vec_f64(vec![
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
        ])
        .unwrap();

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[(0, 0, 0)].re, 6.0);
        assert_eq!(sum[(0, 1, 1)].re, 12.0);
        assert_eq!(sum[(1, 0, 0)].re, 14.0);
        assert_eq!(sum[(1, 1, 1)].re, 20.0);

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[(0, 0, 0)].re, 4.0);
        assert_eq!(diff[(0, 1, 1)].re, 4.0);
        assert_eq!(diff[(1, 0, 0)].re, 4.0);
        assert_eq!(diff[(1, 1, 1)].re, 4.0);

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[(0, 0, 0)].re, 5.0);
        assert_eq!(prod[(0, 1, 1)].re, 32.0);
        assert_eq!(prod[(1, 0, 0)].re, 45.0);
        assert_eq!(prod[(1, 1, 1)].re, 96.0);
    }

    #[test]
    fn test_scalar_operations() {
        let matrix = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[(0, 0, 0)].re, 6.0);
        assert_eq!(added[(1, 1, 1)].re, 13.0);

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[(0, 0, 0)].re, 2.0);
        assert_eq!(multiplied[(0, 1, 1)].re, 8.0);
        assert_eq!(multiplied[(1, 0, 0)].re, 10.0);
        assert_eq!(multiplied[(1, 1, 1)].re, 16.0);
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let other = Points::from_vec_f64(vec![
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
        ])
        .unwrap();

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[(0, 0, 0)].re, 6.0);
        assert_eq!(matrix[(1, 1, 1)].re, 20.0);

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[(0, 0, 0)].re, 12.0);
        assert_eq!(matrix[(1, 1, 1)].re, 40.0);
    }

    #[test]
    fn test_map_functions() {
        let matrix = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[(0, 0, 0)].re, 1.0);
        assert_eq!(squared[(0, 1, 1)].re, 16.0);
        assert_eq!(squared[(1, 0, 0)].re, 25.0);
        assert_eq!(squared[(1, 1, 1)].re, 64.0);

        // Original should be unchanged
        assert_eq!(matrix[(0, 0, 0)].re, 1.0);
        assert_eq!(matrix[(1, 1, 1)].re, 8.0);

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * &Complex64::new(2.0, 0.0));
        assert_eq!(matrix_mut[(0, 0, 0)].re, 2.0);
        assert_eq!(matrix_mut[(0, 1, 1)].re, 8.0);
        assert_eq!(matrix_mut[(1, 0, 0)].re, 10.0);
        assert_eq!(matrix_mut[(1, 1, 1)].re, 16.0);
    }

    #[test]
    fn test_display() {
        let matrix = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let display_str = format!("{}", matrix);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
        assert!(display_str.contains("5"));
        assert!(display_str.contains("6"));
        assert!(display_str.contains("7"));
        assert!(display_str.contains("8"));
    }

    #[test]
    fn test_equality() {
        let matrix1 = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let matrix2 = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let matrix3 = Points::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 5.0]],
            vec![vec![5.0, 6.0], vec![7.0, 9.0]],
        ])
        .unwrap();

        assert_eq!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
    }

    #[test]
    fn test_error_cases() {
        // Test mismatched dimensions for addition
        let matrix1 = Points::zeros((2, 4, 5));
        let matrix2 = Points::zeros((3, 2, 1));

        std::panic::catch_unwind(|| {
            let _ = &matrix1 + &matrix2;
        })
        .expect_err("Should panic on dimension mismatch");
    }

    #[test]
    fn test_conversions() {
        let data = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];

        // Test From<Vec<Vec<f64>>>
        let matrix: Points = data.into();
        assert_eq!(matrix[(0, 0, 0)].re, 1.0);
        assert_eq!(matrix[(1, 1, 1)].re, 8.0);

        // Test conversion to Array3
        let array: Array3<Complex64> = matrix.into();
        assert_eq!(array[(0, 0, 0)].re, 1.0);
        assert_eq!(array[(1, 1, 1)].re, 8.0);
    }
}
