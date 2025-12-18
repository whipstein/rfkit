use crate::{
    error::InversionError,
    myfloat::MyFloat,
    pts::{Points, Pts},
};
use ndarray::{IntoDimension, prelude::*};
use ndarray_linalg::error::LinalgError;
use num_traits::{One, Zero};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A matrix wrapper around ndarray::Array2 with f64 elements
// pub struct Points<MyFloat, Ix2>(Array2<MyFloat>);

impl Points<MyFloat, Ix2> {
    pub fn new(array: Array2<MyFloat>) -> Self {
        Points(array)
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_f64(data: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let rows = data.len();
        let cols = data[0].len();

        if cols == 0 {
            return Err("Cannot create matrix with zero columns");
        }

        // Check all rows have the same length
        if !data.iter().all(|row| row.len() == cols) {
            return Err("All rows must have the same length");
        }

        let mut matrix = Self::zeros((rows, cols));
        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.clone().iter().enumerate() {
                matrix[[i, j]] = val.into();
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_c64(data: Vec<Vec<(f64, f64)>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let rows = data.len();
        let cols = data[0].len();

        if cols == 0 {
            return Err("Cannot create matrix with zero columns");
        }

        if !data.iter().all(|row| row.len() == cols) {
            return Err("All rows must have the same length");
        }

        let mut matrix = Self::zeros((rows, cols));
        for (i, row) in data.iter().enumerate() {
            for (j, &(real, _)) in row.clone().iter().enumerate() {
                matrix[[i, j]] = real.into();
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector
    fn from_vec(data: Vec<Vec<MyFloat>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let rows = data.len();
        let cols = data[0].len();

        if cols == 0 {
            return Err("Cannot create matrix with zero columns");
        }

        // Check all rows have the same length
        if !data.iter().all(|row| row.len() == cols) {
            return Err("All rows must have the same length");
        }

        let mut matrix = Self::zeros((rows, cols));
        for (i, row) in data.iter().enumerate() {
            for (j, val) in row.clone().iter().enumerate() {
                matrix[[i, j]] = val.clone();
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_float(data: Vec<Vec<MyFloat>>) -> Result<Self, &'static str> {
        Self::from_vec(data)
    }

    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<(MyFloat, MyFloat)>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Cannot create matrix from empty data");
        }

        let rows = data.len();
        let cols = data[0].len();

        if cols == 0 {
            return Err("Cannot create matrix with zero columns");
        }

        if !data.iter().all(|row| row.len() == cols) {
            return Err("All rows must have the same length");
        }

        let mut matrix = Self::zeros((rows, cols));
        for (i, row) in data.iter().enumerate() {
            for (j, (real, _)) in row.clone().iter().enumerate() {
                matrix[[i, j]] = real.clone();
            }
        }

        Ok(matrix)
    }
}

impl Pts<MyFloat, Ix2> for Points<MyFloat, Ix2> {
    type Dim = (usize, usize);
    type Tuple<'a> = &'a [(f64, f64)];

    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>) -> Self {
        Points(Array2::from_elem(shape, MyFloat::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>) -> Self {
        Points(Array2::from_elem(shape, MyFloat::one()))
    }

    /// Create an identity matrix of given size
    fn eye(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>) -> Self {
        Points(Array2::from_shape_fn(shape, |(j, k)| {
            if j == k {
                MyFloat::one()
            } else {
                MyFloat::zero()
            }
        }))
    }

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        shape: impl IntoDimension<Dim = Dim<[usize; 2]>>,
    ) -> Result<Self, &'static str> {
        let (rows, cols) = shape.into_dimension().into_pattern();
        if data.len() != rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        let mut matrix = Self::zeros((rows, cols));
        for (idx, _) in data.iter().enumerate() {
            let i = idx / cols;
            let j = idx % cols;
            matrix[[i, j]] = MyFloat::zero();
        }

        Ok(matrix)
    }

    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>, f: F) -> Self
    where
        F: Fn(Self::Dim) -> MyFloat,
    {
        Points(Array2::<MyFloat>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(
        shape: impl IntoDimension<Dim = Dim<[usize; 2]>>,
        v: Vec<MyFloat>,
    ) -> Result<Self, &'static str> {
        match Array2::<MyFloat>::from_shape_vec(shape, v) {
            Ok(x) => Ok(Points(x)),
            Err(_) => Err("Cannot create vector"),
        }
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, MyFloat, ndarray::Ix1> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(
        &mut self,
        axis: Axis,
    ) -> ndarray::iter::AxisIterMut<'_, MyFloat, ndarray::Ix1> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, MyFloat, ndarray::Ix2> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, MyFloat, ndarray::Ix2> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, MyFloat, ndarray::Ix1> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, MyFloat, ndarray::Ix1> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, MyFloat, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix2>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, MyFloat, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix2>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<MyFloat, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, MyFloat, ndarray::Ix1>,
    ) -> Result<(), ndarray::ShapeError>
    where
        MyFloat: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get the number of rows
    fn len_of(&self, _axis: Axis) -> usize {
        1
    }

    /// Get the number of points
    fn npts(&self) -> usize {
        self.0.shape()[0]
    }

    /// Get the number of rows
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    /// Get the number of columns
    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    /// Get the shape as (rows, cols)
    fn dim(&self) -> Self::Dim {
        self.0.dim()
    }

    /// Get the dimension
    fn raw_dim(&self) -> Ix2 {
        self.0.raw_dim()
    }

    /// Get the shape as (rows, cols)
    fn shape(&self) -> Self::Dim {
        let shape = self.0.dim();
        (shape.0, shape.1)
    }

    /// Check if the matrix is square
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView2<'_, MyFloat> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut2<'_, MyFloat> {
        self.0.view_mut()
    }

    /// Transpose the matrix
    fn t(&self) -> Self {
        Points(self.0.t().to_owned())
    }

    /// Transpose the matrix
    fn transpose(&self) -> Self {
        self.t()
    }

    /// Get the conjugate transpose (Hermitian transpose)
    fn h(&self) -> Self {
        // let transposed = self.transpose();
        // let mut result = transposed;
        // for element in result.0.iter_mut() {
        //     *element = element.conj();
        // }
        // result
        self.t()
    }

    /// Get the conjugate transpose (Hermitian transpose)
    fn conj_transpose(&self) -> Self {
        self.h()
    }

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> Result<Array0<MyFloat>, LinalgError> {
        let n = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        }?;
        let sum = (0..n as usize).map(|i| &self[(i, i)]).sum();
        Ok(Array0::from_elem((), sum))
        // if !self.is_square() {
        //     panic!("Trace is only defined for square matrices");
        // }

        // let mut sum = MyFloat::zero();
        // for i in 0..self.nrows() {
        //     sum += &self[[i, i]];
        // }
        // sum
    }

    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Result<Array0<MyFloat>, LinalgError> {
        _ = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        };

        match self.nrows() {
            0 => Ok(Array0::from_elem((), MyFloat::one())),
            1 => Ok(Array0::from_elem((), self[[0, 0]].clone())),
            2 => Ok(Array0::from_elem(
                (),
                &self[[0, 0]] * &self[[1, 1]] - &self[[0, 1]] * &self[[1, 0]],
            )),
            3 => {
                let a00 = &self[[0, 0]];
                let a01 = &self[[0, 1]];
                let a02 = &self[[0, 2]];
                let a10 = &self[[1, 0]];
                let a11 = &self[[1, 1]];
                let a12 = &self[[1, 2]];
                let a20 = &self[[2, 0]];
                let a21 = &self[[2, 1]];
                let a22 = &self[[2, 2]];

                Ok(Array0::from_elem(
                    (),
                    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
                        + a02 * (a10 * a21 - a11 * a20),
                ))
            }
            _ => Err(LinalgError::NotStandardShape {
                obj: "Determinant calculation for matrices larger than 3x3 not implemented",
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        }
    }

    /// Element-wise conjugate
    fn conj(&self) -> Self {
        // let mut result = self.clone();
        // for element in result.0.iter_mut() {
        //     *element = element.conj();
        // }
        // result
        self.clone()
    }

    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> Array0<MyFloat> {
        let mut sum = MyFloat::zero();
        for element in self.0.iter() {
            sum += element.abs().square();
        }
        Array0::from_elem((), sum.sqrt())
    }

    /// Point multiplication
    fn dot(&self, other: &Self) -> Self {
        if self.ncols() != other.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {}x{} * {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            );
        }

        let mut result = Self::zeros((self.nrows(), other.ncols()));

        for i in 0..self.nrows() {
            for j in 0..other.ncols() {
                let mut sum = MyFloat::zero();
                for k in 0..self.ncols() {
                    sum += &self[[i, k]] * &other[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    /// Get a row as a new matrix (1 x ncols)
    fn row(&self, index: usize) -> Self {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} rows",
                index,
                self.nrows()
            );
        }

        let mut result = Self::zeros((1, self.ncols()));
        for j in 0..self.ncols() {
            result[[0, j]] = self[[index, j]].clone();
        }
        result
    }

    /// Get a column as a new matrix (nrows x 1)
    fn col(&self, index: usize) -> Self {
        if index >= self.ncols() {
            panic!(
                "Column index {} out of bounds for matrix with {} columns",
                index,
                self.ncols()
            );
        }

        let mut result = Self::zeros((self.nrows(), 1));
        for i in 0..self.nrows() {
            result[[i, 0]] = self[[i, index]].clone();
        }
        result
    }

    /// Set a row from another matrix
    fn set_row(&mut self, index: usize, row: &Self) {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} rows",
                index,
                self.nrows()
            );
        }
        if row.nrows() != 1 || row.ncols() != self.ncols() {
            panic!(
                "Row dimensions incompatible: expected 1x{}, got {}x{}",
                self.ncols(),
                row.nrows(),
                row.ncols()
            );
        }

        for j in 0..self.ncols() {
            self[[index, j]] = row[[0, j]].clone();
        }
    }

    /// Set a column from another matrix
    fn set_col(&mut self, index: usize, col: &Self) {
        if index >= self.ncols() {
            panic!(
                "Column index {} out of bounds for matrix with {} columns",
                index,
                self.ncols()
            );
        }
        if col.nrows() != self.nrows() || col.ncols() != 1 {
            panic!(
                "Column dimensions incompatible: expected {}x1, got {}x{}",
                self.nrows(),
                col.nrows(),
                col.ncols()
            );
        }

        for i in 0..self.nrows() {
            self[[i, index]] = col[[i, 0]].clone();
        }
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array2<MyFloat> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array2<MyFloat> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<MyFloat>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>, value: MyFloat) -> Self {
        Points(Array2::from_elem(shape, value))
    }

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&MyFloat) -> MyFloat,
    {
        Points(self.0.map(&f))
    }

    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&MyFloat) -> MyFloat,
    {
        self.0.map_inplace(|x| *x = f(x));
    }

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    ///
    /// This function computes A^(-1) such that A * A^(-1) = I, where I is the identity matrix.
    ///
    /// # Arguments
    /// * `matrix` - A square matrix to invert
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - The inverted matrix
    /// * `Err(InversionError)` - If the matrix is not square, singular, or other errors
    ///
    /// # Examples
    /// ```rust
    /// use ndarray::prelude::*;
    /// use rfkit::prelude::*;
    ///
    /// // Create a 2x2 matrix
    /// let matrix = Points::from_shape_vec((2, 2), vec![
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]).unwrap();
    ///
    /// let inv_matrix = matrix.inv();
    /// ```
    fn inv(&self) -> Points<MyFloat, Ix2> {
        self.try_inv().unwrap()
    }

    fn try_inv(&self) -> Result<Points<MyFloat, Ix2>, InversionError> {
        let (rows, cols) = self.dim();

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
                augmented[[i, j]] = self.0[[i, j]].clone();
            }
        }

        // Create identity matrix in right half
        for i in 0..n {
            augmented[[i, i + n]] = MyFloat::one();
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
            if max_abs < 1e-12 {
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

        Ok(Points(inverse))
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
    fn solve_linear_system(
        &self,
        b: &ArrayView2<MyFloat>,
    ) -> Result<Array2<MyFloat>, InversionError> {
        let (a_rows, a_cols) = self.dim();
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
                augmented[[i, j]] = self[[i, j]].clone();
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
            if max_abs < 1e-12 {
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
    fn approx_eq(a: &ArrayView2<MyFloat>, b: &ArrayView2<MyFloat>, tol: f64) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        let (rows, cols) = a.dim();

        for i in 0..rows {
            for j in 0..cols {
                let diff = &a[[i, j]] - &b[[i, j]];
                if diff.abs() > tol {
                    return false;
                }
            }
        }

        true
    }
}

// Indexing
impl Index<(usize, usize)> for Points<MyFloat, Ix2> {
    type Output = MyFloat;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<[usize; 2]> for Points<MyFloat, Ix2> {
    type Output = MyFloat;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<(usize, usize)> for Points<MyFloat, Ix2> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IndexMut<[usize; 2]> for Points<MyFloat, Ix2> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Addition implementations
impl Add for Points<MyFloat, Ix2> {
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

impl Add<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
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

impl Add<Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn add(self, other: Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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

impl Add<&Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn add(self, other: &Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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
impl Add<MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn add(self, scalar: MyFloat) -> Self {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn add(self, scalar: MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<&MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn add(self, scalar: &MyFloat) -> Self {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<&MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn add(self, scalar: &MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<&f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn add(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x + *scalar))
    }
}

impl Add<f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn add(self, scalar: f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<&f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn add(self, scalar: &f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x + *scalar))
    }
}

// Subtraction implementations
impl Sub for Points<MyFloat, Ix2> {
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

impl Sub<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
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

impl Sub<Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn sub(self, other: Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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

impl Sub<&Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn sub(self, other: &Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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
impl Sub<MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn sub(self, scalar: MyFloat) -> Self {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn sub(self, scalar: MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<&MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn sub(self, scalar: &MyFloat) -> Self {
        Points(self.0.map(|x| x - scalar))
    }
}

impl Sub<&MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn sub(self, scalar: &MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x - scalar))
    }
}

impl Sub<f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x - scalar))
    }
}

impl Sub<&f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn sub(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x - *scalar))
    }
}

impl Sub<f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn sub(self, scalar: f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x - scalar))
    }
}

impl Sub<&f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn sub(self, scalar: &f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x - *scalar))
    }
}

// Element-wise multiplication
impl Mul for Points<MyFloat, Ix2> {
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

impl Mul<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
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

impl Mul<Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn mul(self, other: Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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

impl Mul<&Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn mul(self, other: &Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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
impl Mul<MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn mul(self, scalar: MyFloat) -> Self {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn mul(self, scalar: MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<&MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn mul(self, scalar: &MyFloat) -> Self {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<&MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn mul(self, scalar: &MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<&f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn mul(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x * *scalar))
    }
}

impl Mul<f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn mul(self, scalar: f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<&f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn mul(self, scalar: &f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x * *scalar))
    }
}

// Division implementations
impl Div for Points<MyFloat, Ix2> {
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

impl Div<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
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

impl Div<Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn div(self, other: Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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

impl Div<&Points<MyFloat, Ix2>> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn div(self, other: &Points<MyFloat, Ix2>) -> Points<MyFloat, Ix2> {
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
impl Div<MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn div(self, scalar: MyFloat) -> Self {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn div(self, scalar: MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<&MyFloat> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn div(self, scalar: &MyFloat) -> Self {
        Points(self.0.map(|x| x / scalar))
    }
}

impl Div<&MyFloat> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn div(self, scalar: &MyFloat) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x / scalar))
    }
}

impl Div<f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x / scalar))
    }
}

impl Div<&f64> for Points<MyFloat, Ix2> {
    type Output = Self;

    fn div(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x / *scalar))
    }
}

impl Div<f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn div(self, scalar: f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x / scalar))
    }
}

impl Div<&f64> for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn div(self, scalar: &f64) -> Points<MyFloat, Ix2> {
        Points(self.0.map(|x| x / *scalar))
    }
}

// Negation
impl Neg for Points<MyFloat, Ix2> {
    type Output = Self;

    fn neg(self) -> Self {
        Points(-self.0)
    }
}

impl Neg for &Points<MyFloat, Ix2> {
    type Output = Points<MyFloat, Ix2>;

    fn neg(self) -> Points<MyFloat, Ix2> {
        Points(-&self.0)
    }
}

// Assignment operators
impl AddAssign for Points<MyFloat, Ix2> {
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

impl AddAssign<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
    fn add_assign(&mut self, other: &Points<MyFloat, Ix2>) {
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

impl SubAssign for Points<MyFloat, Ix2> {
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

impl SubAssign<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
    fn sub_assign(&mut self, other: &Points<MyFloat, Ix2>) {
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

impl MulAssign for Points<MyFloat, Ix2> {
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

impl MulAssign<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
    fn mul_assign(&mut self, other: &Points<MyFloat, Ix2>) {
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

impl DivAssign for Points<MyFloat, Ix2> {
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

impl DivAssign<&Points<MyFloat, Ix2>> for Points<MyFloat, Ix2> {
    fn div_assign(&mut self, other: &Points<MyFloat, Ix2>) {
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
impl AddAssign<MyFloat> for Points<MyFloat, Ix2> {
    fn add_assign(&mut self, scalar: MyFloat) {
        self.0.map_inplace(|x| *x = &*x + &scalar);
    }
}

impl AddAssign<&MyFloat> for Points<MyFloat, Ix2> {
    fn add_assign(&mut self, scalar: &MyFloat) {
        self.0.map_inplace(|x| *x = &*x + scalar);
    }
}

impl AddAssign<f64> for Points<MyFloat, Ix2> {
    fn add_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x + scalar);
    }
}

impl AddAssign<&f64> for Points<MyFloat, Ix2> {
    fn add_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x + *scalar);
    }
}

impl SubAssign<MyFloat> for Points<MyFloat, Ix2> {
    fn sub_assign(&mut self, scalar: MyFloat) {
        self.0.map_inplace(|x| *x = &*x - &scalar);
    }
}

impl SubAssign<&MyFloat> for Points<MyFloat, Ix2> {
    fn sub_assign(&mut self, scalar: &MyFloat) {
        self.0.map_inplace(|x| *x = &*x - scalar);
    }
}

impl SubAssign<f64> for Points<MyFloat, Ix2> {
    fn sub_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x - scalar);
    }
}

impl SubAssign<&f64> for Points<MyFloat, Ix2> {
    fn sub_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x - *scalar);
    }
}

impl MulAssign<MyFloat> for Points<MyFloat, Ix2> {
    fn mul_assign(&mut self, scalar: MyFloat) {
        self.0.map_inplace(|x| *x = &*x * &scalar);
    }
}

impl MulAssign<&MyFloat> for Points<MyFloat, Ix2> {
    fn mul_assign(&mut self, scalar: &MyFloat) {
        self.0.map_inplace(|x| *x = &*x * scalar);
    }
}

impl MulAssign<f64> for Points<MyFloat, Ix2> {
    fn mul_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x * scalar);
    }
}

impl MulAssign<&f64> for Points<MyFloat, Ix2> {
    fn mul_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x * *scalar);
    }
}

impl DivAssign<MyFloat> for Points<MyFloat, Ix2> {
    fn div_assign(&mut self, scalar: MyFloat) {
        self.0.map_inplace(|x| *x = &*x / &scalar);
    }
}

impl DivAssign<&MyFloat> for Points<MyFloat, Ix2> {
    fn div_assign(&mut self, scalar: &MyFloat) {
        self.0.map_inplace(|x| *x = &*x / scalar);
    }
}

impl DivAssign<f64> for Points<MyFloat, Ix2> {
    fn div_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x / scalar);
    }
}

impl DivAssign<&f64> for Points<MyFloat, Ix2> {
    fn div_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x / *scalar);
    }
}

// Traits
// impl Clone for Points<MyFloat, Ix2> {
//     fn clone(&self) -> Self {
//         Points(self.0.clone())
//     }
// }

impl Default for Points<MyFloat, Ix2> {
    fn default() -> Self {
        Points::<MyFloat, Ix2>::zeros((0, 0))
    }
}

impl PartialEq for Points<MyFloat, Ix2> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl Zero for Points<MyFloat, Ix2> {
    fn zero() -> Self {
        Points::<MyFloat, Ix2>::zeros((0, 0))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

// Display implementation
impl fmt::Display for Points<MyFloat, Ix2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nrows() == 0 || self.ncols() == 0 {
            return write!(f, "[]");
        }

        writeln!(f, "[")?;
        for i in 0..self.nrows() {
            write!(f, "  [")?;
            for j in 0..self.ncols() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self[[i, j]])?;
            }
            if i < self.nrows() - 1 {
                writeln!(f, "],")?;
            } else {
                writeln!(f, "]")?;
            }
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Points<MyFloat, Ix2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Points({}x{}) {}", self.nrows(), self.ncols(), self)
    }
}

// Conversion traits
impl From<Array2<MyFloat>> for Points<MyFloat, Ix2> {
    fn from(array: Array2<MyFloat>) -> Self {
        Points(array)
    }
}

impl From<ArrayView2<'_, MyFloat>> for Points<MyFloat, Ix2> {
    fn from(array: ArrayView2<MyFloat>) -> Self {
        Points(array.to_owned())
    }
}

impl From<Points<MyFloat, Ix2>> for Array2<MyFloat> {
    fn from(matrix: Points<MyFloat, Ix2>) -> Self {
        matrix.0
    }
}

impl From<Vec<Vec<f64>>> for Points<MyFloat, Ix2> {
    fn from(data: Vec<Vec<f64>>) -> Self {
        Points::<MyFloat, Ix2>::from_vec_f64(data).expect("Invalid matrix data")
    }
}

impl From<Vec<Vec<MyFloat>>> for Points<MyFloat, Ix2> {
    fn from(data: Vec<Vec<MyFloat>>) -> Self {
        Points::<MyFloat, Ix2>::from_vec_float(data).expect("Invalid matrix data")
    }
}

impl From<Vec<Vec<(f64, f64)>>> for Points<MyFloat, Ix2> {
    fn from(data: Vec<Vec<(f64, f64)>>) -> Self {
        Points::<MyFloat, Ix2>::from_vec_c64(data).expect("Invalid matrix data")
    }
}

impl From<Vec<Vec<(MyFloat, MyFloat)>>> for Points<MyFloat, Ix2> {
    fn from(data: Vec<Vec<(MyFloat, MyFloat)>>) -> Self {
        Points::<MyFloat, Ix2>::from_vec_complex(data).expect("Invalid matrix data")
    }
}

impl From<Points<f64, Ix2>> for Points<MyFloat, Ix2> {
    fn from(point: Points<f64, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(j, k)| point[[j, k]].into())
    }
}

impl From<&Points<f64, Ix2>> for Points<MyFloat, Ix2> {
    fn from(point: &Points<f64, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(j, k)| point[[j, k]].into())
    }
}

#[cfg(test)]
mod points_ix2_myfloat_tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn test_creation() {
        let zeros = Points::<MyFloat, Ix2>::zeros((2, 3));
        assert_eq!(zeros.shape(), (2, 3));
        assert!(zeros[[0, 0]].is_zero());
        assert!(zeros[[1, 2]].is_zero());

        let ones = Points::<MyFloat, Ix2>::ones((3, 2));
        assert_eq!(ones.shape(), (3, 2));
        assert!(ones[[0, 0]].is_one());
        assert!(ones[[2, 1]].is_one());

        let eye = Points::<MyFloat, Ix2>::eye((3, 3));
        assert_eq!(eye.shape(), (3, 3));
        assert!(eye[[0, 0]].is_one());
        assert!(eye[[1, 1]].is_one());
        assert!(eye[[2, 2]].is_one());
        assert!(eye[[0, 1]].is_zero());
        assert!(eye[[1, 0]].is_zero());
    }

    #[test]
    fn test_from_vec() {
        let data: Vec<Vec<MyFloat>> = vec![
            vec![1.0.into(), 2.0.into(), 3.0.into()],
            vec![4.0.into(), 5.0.into(), 6.0.into()],
        ];
        let matrix = Points::<MyFloat, Ix2>::from_vec(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
        assert_eq!(matrix[[1, 0]], 4.0);
        assert_eq!(matrix[[1, 1]], 5.0);
        assert_eq!(matrix[[1, 2]], 6.0);
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Points::<MyFloat, Ix2>::from_vec_f64(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
        assert_eq!(matrix[[1, 0]], 4.0);
        assert_eq!(matrix[[1, 1]], 5.0);
        assert_eq!(matrix[[1, 2]], 6.0);
    }

    #[test]
    fn test_from_vec_float() {
        let data: Vec<Vec<MyFloat>> = vec![
            vec![1.0.into(), 2.0.into(), 3.0.into()],
            vec![4.0.into(), 5.0.into(), 6.0.into()],
        ];
        let matrix = Points::<MyFloat, Ix2>::from_vec_float(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
        assert_eq!(matrix[[1, 0]], 4.0);
        assert_eq!(matrix[[1, 1]], 5.0);
        assert_eq!(matrix[[1, 2]], 6.0);
    }

    #[test]
    fn test_from_vec_c64() {
        let data = vec![vec![(1.0, 2.0), (3.0, 4.0)], vec![(5.0, 6.0), (7.0, 8.0)]];
        let matrix = Points::<MyFloat, Ix2>::from_vec_c64(data).unwrap();

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 7.0);
    }

    #[test]
    fn test_from_vec_complex() {
        let data = vec![
            vec![
                (MyFloat::new(1.0), MyFloat::new(2.0)),
                (MyFloat::new(3.0), MyFloat::new(4.0)),
            ],
            vec![
                (MyFloat::new(5.0), MyFloat::new(6.0)),
                (MyFloat::new(7.0), MyFloat::new(8.0)),
            ],
        ];
        let matrix = Points::<MyFloat, Ix2>::from_vec_complex(data).unwrap();

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 7.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[[0, 0]], 6.0);
        assert_eq!(sum[[0, 1]], 8.0);
        assert_eq!(sum[[1, 0]], 10.0);
        assert_eq!(sum[[1, 1]], 12.0);

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[[0, 0]], 4.0);
        assert_eq!(diff[[0, 1]], 4.0);
        assert_eq!(diff[[1, 0]], 4.0);
        assert_eq!(diff[[1, 1]], 4.0);

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[[0, 0]], 5.0);
        assert_eq!(prod[[0, 1]], 12.0);
        assert_eq!(prod[[1, 0]], 21.0);
        assert_eq!(prod[[1, 1]], 32.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        let result = a.dot(&b);

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result[[0, 0]], 19.0);
        assert_eq!(result[[0, 1]], 22.0);
        assert_eq!(result[[1, 0]], 43.0);
        assert_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn test_transpose() {
        let matrix =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
                .unwrap();

        let transposed = matrix.transpose();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed[[0, 0]], 1.0);
        assert_eq!(transposed[[0, 1]], 4.0);
        assert_eq!(transposed[[1, 0]], 2.0);
        assert_eq!(transposed[[1, 1]], 5.0);
        assert_eq!(transposed[[2, 0]], 3.0);
        assert_eq!(transposed[[2, 1]], 6.0);
    }

    #[test]
    fn test_conjugate_transpose() {
        let matrix = Points::<MyFloat, Ix2>::from_vec_c64(vec![
            vec![(1.0, 2.0), (3.0, 4.0)],
            vec![(5.0, 6.0), (7.0, 8.0)],
        ])
        .unwrap();

        let conj_t = matrix.conj_transpose();
        assert_eq!(conj_t.shape(), (2, 2));

        // Original: [(1+2i, 3+4i), (5+6i, 7+8i)]
        // Conj transpose: [(1-2i, 5-6i), (3-4i, 7-8i)]
        assert_eq!(conj_t[[0, 0]], 1.0);
        assert_eq!(conj_t[[0, 1]], 5.0);
        assert_eq!(conj_t[[1, 0]], 3.0);
        assert_eq!(conj_t[[1, 1]], 7.0);
    }

    #[test]
    fn test_trace() {
        let matrix = Points::<MyFloat, Ix2>::from_vec_f64(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        let trace = matrix.trace().unwrap().into_scalar();
        assert_eq!(trace, 15.0); // 1 + 5 + 9 = 15
    }

    #[test]
    fn test_determinant() {
        // Test 2x2 determinant
        let matrix2x2 =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let det2 = matrix2x2.det().unwrap().into_scalar();
        assert_eq!(det2, -2.0); // 1*4 - 2*3 = -2

        // Test 3x3 determinant
        let matrix3x3 = Points::<MyFloat, Ix2>::from_vec_f64(vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ])
        .unwrap();

        let det3 = matrix3x3.det().unwrap().into_scalar();
        assert_eq!(det3, 1.0); // Should be 1
    }

    #[test]
    fn test_scalar_operations() {
        let matrix =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[[0, 0]], 6.0);
        assert_eq!(added[[1, 1]], 9.0);

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[[0, 0]], 2.0);
        assert_eq!(multiplied[[0, 1]], 4.0);
        assert_eq!(multiplied[[1, 0]], 6.0);
        assert_eq!(multiplied[[1, 1]], 8.0);
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let other =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[[0, 0]], 6.0);
        assert_eq!(matrix[[1, 1]], 12.0);

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[[0, 0]], 12.0);
        assert_eq!(matrix[[1, 1]], 24.0);
    }

    #[test]
    fn test_row_col_operations() {
        let matrix = Points::<MyFloat, Ix2>::from_vec_f64(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        // Test getting a row
        let row1 = matrix.row(1);
        assert_eq!(row1.shape(), (1, 3));
        assert_eq!(row1[[0, 0]], 4.0);
        assert_eq!(row1[[0, 1]], 5.0);
        assert_eq!(row1[[0, 2]], 6.0);

        // Test getting a column
        let col2 = matrix.col(2);
        assert_eq!(col2.shape(), (3, 1));
        assert_eq!(col2[[0, 0]], 3.0);
        assert_eq!(col2[[1, 0]], 6.0);
        assert_eq!(col2[[2, 0]], 9.0);

        // Test setting a row
        let mut matrix_mut = matrix.clone();
        let new_row = Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![10.0, 11.0, 12.0]]).unwrap();
        matrix_mut.set_row(0, &new_row);
        assert_eq!(matrix_mut[[0, 0]], 10.0);
        assert_eq!(matrix_mut[[0, 1]], 11.0);
        assert_eq!(matrix_mut[[0, 2]], 12.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![3.0, 4.0], vec![0.0, 0.0]]).unwrap();

        let norm = matrix.frobenius_norm().into_scalar();
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_map_functions() {
        let matrix =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[[0, 0]], 1.0);
        assert_eq!(squared[[0, 1]], 4.0);
        assert_eq!(squared[[1, 0]], 9.0);
        assert_eq!(squared[[1, 1]], 16.0);

        // Original should be unchanged
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 4.0);

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * 2.0);
        assert_eq!(matrix_mut[[0, 0]], 2.0);
        assert_eq!(matrix_mut[[0, 1]], 4.0);
        assert_eq!(matrix_mut[[1, 0]], 6.0);
        assert_eq!(matrix_mut[[1, 1]], 8.0);
    }

    #[test]
    fn test_display() {
        let matrix =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let display_str = format!("{}", matrix);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
    }

    #[test]
    fn test_equality() {
        let matrix1 =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let matrix2 =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let matrix3 =
            Points::<MyFloat, Ix2>::from_vec_f64(vec![vec![1.0, 2.0], vec![3.0, 5.0]]).unwrap();

        assert_eq!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
    }

    #[test]
    fn test_error_cases() {
        // Test mismatched dimensions for addition
        let matrix1 = Points::<MyFloat, Ix2>::zeros((2, 4));
        let matrix2 = Points::<MyFloat, Ix2>::zeros((3, 2));

        std::panic::catch_unwind(|| {
            let _ = &matrix1 + &matrix2;
        })
        .expect_err("Should panic on dimension mismatch");

        // Test incompatible matrix multiplication
        std::panic::catch_unwind(|| {
            let _ = matrix1.dot(&matrix2);
        })
        .expect_err("Should panic on incompatible multiplication");

        // Test trace on non-square matrix
        let result = matrix1.trace();
        match result {
            Err(LinalgError::NotSquare { rows: 2, cols: 4 }) => (),
            _ => panic!("Should panic on trace of non-square matrix"),
        }
    }

    #[test]
    fn test_conversions() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        // Test From<Vec<Vec<f64>>>
        let matrix: Points<MyFloat, Ix2> = data.into();
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 4.0);

        // Test conversion to Array2
        let array: Array2<MyFloat> = matrix.into();
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 1]], 4.0);
    }

    #[test]
    fn test_2x2_matrix_inversion() {
        // Test inverting a simple 2x2 matrix
        // A = [1 2]  =>  A^(-1) = [-2  1]
        //     [3 4]               [1.5 -0.5]

        let matrix = Points::<MyFloat, Ix2>::from_shape_vec(
            (2, 2),
            vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Check specific values
        approx_eq!(f64, inverse[[0, 0]].to_f64(), -2.0, F64Margin::default());
        approx_eq!(f64, inverse[[0, 1]].to_f64(), 1.0, F64Margin::default());
        approx_eq!(f64, inverse[[1, 0]].to_f64(), 1.5, F64Margin::default());
        approx_eq!(f64, inverse[[1, 1]].to_f64(), -0.5, F64Margin::default());

        // Verify A * A^(-1) = I
        let product = &matrix.dot(&inverse);
        let identity = Points::<MyFloat, Ix2>::eye((2, 2));

        assert!(Points::approx_eq(&product.view(), &identity.view(), 1e-10));
    }

    #[test]
    fn test_complex_matrix_inversion() {
        // Test with complex numbers
        let matrix = Points::<MyFloat, Ix2>::from_shape_vec(
            (2, 2),
            vec![1.0.into(), 0.0.into(), 1.0.into(), 1.0.into()],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Verify A * A^(-1) = I
        let product = matrix.dot(&inverse);
        let identity = Points::<MyFloat, Ix2>::eye((2, 2));

        assert!(Points::approx_eq(&product.view(), &identity.view(), 1e-10));
    }

    #[test]
    fn test_3x3_matrix_inversion() {
        // Test with a 3x3 matrix
        let matrix = Points::<MyFloat, Ix2>::from_shape_vec(
            (3, 3),
            vec![
                2.0.into(),
                MyFloat::new(-1.0),
                0.0.into(),
                MyFloat::new(-1.0),
                2.0.into(),
                MyFloat::new(-1.0),
                0.0.into(),
                MyFloat::new(-1.0),
                2.0.into(),
            ],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Verify A * A^(-1) = I
        let product = matrix.dot(&inverse);
        let identity = Points::<MyFloat, Ix2>::eye((3, 3));

        assert!(Points::approx_eq(&product.view(), &identity.view(), 1e-10));
    }

    #[test]
    fn test_singular_matrix() {
        // Test with a singular matrix (determinant = 0)
        let matrix = Points::<MyFloat, Ix2>::from_shape_vec(
            (2, 2),
            vec![
                1.0.into(),
                2.0.into(),
                2.0.into(),
                4.0.into(), // Second row is 2x first row
            ],
        )
        .unwrap();

        let result = matrix.try_inv();
        assert!(result.is_err());

        match result {
            Err(InversionError::Singular(_)) => {}
            _ => panic!("Expected Singular error"),
        }
    }

    #[test]
    fn test_non_square_matrix() {
        // Test with a non-square matrix
        let matrix = Points::<MyFloat, Ix2>::from_shape_vec(
            (2, 3),
            vec![
                1.0.into(),
                2.0.into(),
                3.0.into(),
                4.0.into(),
                5.0.into(),
                6.0.into(),
            ],
        )
        .unwrap();

        let result = matrix.try_inv();
        assert!(result.is_err());

        match result {
            Err(InversionError::NotSquare(_)) => {}
            _ => panic!("Expected NotSquare error"),
        }
    }

    #[test]
    fn test_identity_matrix_inversion() {
        // Identity matrix should be its own inverse
        let identity = Points::<MyFloat, Ix2>::eye((4, 4));
        let inverse = identity.inv();

        assert!(Points::approx_eq(&identity.view(), &inverse.view(), 1e-10));
    }
}
