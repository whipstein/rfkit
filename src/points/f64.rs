use crate::error::InversionError;
use crate::points::{Points, Pts};
use ndarray::SliceArg;
use ndarray::prelude::*;
use num_traits::{One, Zero};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A matrix wrapper around ndarray::Array3 with f64 elements
// pub struct Points(Array3<f64>);

impl Points<f64> {
    pub fn new(array: Array3<f64>) -> Self {
        Points(array)
    }
}

impl Pts<f64, f64> for Points<f64> {
    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: (usize, usize, usize)) -> Self {
        Points(Array3::from_elem(shape, f64::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: (usize, usize, usize)) -> Self {
        Points(Array3::from_elem(shape, f64::one()))
    }

    /// Create an identity matrix of given size
    fn eye(size: (usize, usize)) -> Self {
        Points(Array3::from_shape_fn(
            (size.0, size.1, size.1),
            |(_, j, k)| {
                if j == k { 1.0 } else { 0.0 }
            },
        ))
    }

    /// Create a matrix from a 3D vector of f64 values
    fn from_vec(data: Vec<Vec<Vec<f64>>>) -> Result<Self, &'static str> {
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
                    matrix[(i, j, k)] = value;
                }
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 3D vector of f64 values
    fn from_vec_f64(data: Vec<Vec<Vec<f64>>>) -> Result<Self, &'static str> {
        Self::from_vec(data)
    }

    /// Create a matrix from a 3D vector of f64 values
    fn from_vec_float(data: Vec<Vec<Vec<f64>>>) -> Result<Self, &'static str> {
        Self::from_vec_f64(data)
    }

    /// Create a matrix from a 3D vector of complex tuples (real, imag)
    fn from_vec_c64(data: Vec<Vec<Vec<(f64, f64)>>>) -> Result<Self, &'static str> {
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
                for (k, &(real, _)) in row.iter().enumerate() {
                    matrix[(i, j, k)] = real;
                }
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 3D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<Vec<(f64, f64)>>>) -> Result<Self, &'static str> {
        Self::from_vec_c64(data)
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
            matrix[(i, j, k)] = value;
        }

        Ok(matrix)
    }

    fn from_shape_fn<F>(shape: (usize, usize, usize), f: F) -> Self
    where
        F: Fn((usize, usize, usize)) -> f64,
    {
        Points(Array3::<f64>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(shape: (usize, usize, usize), v: Vec<f64>) -> Result<Self, &'static str> {
        match Array3::<f64>::from_shape_vec(shape, v) {
            Ok(x) => Ok(Points(x)),
            Err(_) => Err("Cannot create vector"),
        }
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, f64, ndarray::Ix2> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(&mut self, axis: Axis) -> ndarray::iter::AxisIterMut<'_, f64, ndarray::Ix2> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, f64, ndarray::Ix3> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, f64, ndarray::Ix3> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, f64, ndarray::Ix2> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, f64, ndarray::Ix2> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, f64, I::OutDim>
    where
        I: SliceArg<Ix3>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, f64, I::OutDim>
    where
        I: SliceArg<Ix3>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<f64, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, f64, ndarray::Ix2>,
    ) -> Result<(), ndarray::ShapeError>
    where
        f64: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get the number of rows
    fn len_of(&self, axis: Axis) -> usize {
        self.0.len_of(axis)
    }

    /// Get the number of points
    fn npts(&self) -> usize {
        self.0.shape()[0]
    }

    /// Get the number of rows
    fn nrows(&self) -> usize {
        self.0.shape()[1]
    }

    /// Get the number of cols
    fn ncols(&self) -> usize {
        self.0.shape()[2]
    }

    /// Get the shape as (len, rows, cols)
    fn dim(&self) -> (usize, usize, usize) {
        self.0.dim()
    }

    /// Get the shape as (len, rows, cols)
    fn shape(&self) -> (usize, usize, usize) {
        let shape = self.0.dim();
        (shape.0, shape.1, shape.2)
    }

    /// Check if the matrix is square
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView3<'_, f64> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut3<'_, f64> {
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
        self.transpose()
    }

    /// Get the conjugate transpose (Hermitian transpose)
    fn conj_transpose(&self) -> Self {
        self.h()
    }

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> Array1<f64> {
        if !self.is_square() {
            panic!("Trace is only defined for square matrices");
        }

        let mut trace = Array1::zeros(self.npts());
        for (i, pt) in self.outer_iter().enumerate() {
            let mut sum = 0.0;
            for j in 0..pt.nrows() {
                sum += &pt[[j, j]];
            }
            trace[i] = sum;
        }
        trace
    }

    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Array1<f64> {
        if !self.is_square() {
            panic!("Determinant is only defined for square matrices");
        }

        let mut det = Array1::zeros(self.npts());
        for (i, pt) in self.outer_iter().enumerate() {
            det[i] = match pt.nrows() {
                0 => 1.0,
                1 => pt[[0, 0]].clone(),
                2 => &pt[[0, 0]] * &pt[[1, 1]] - &pt[[0, 1]] * &pt[[1, 0]],
                3 => {
                    let a00 = &pt[[0, 0]];
                    let a01 = &pt[[0, 1]];
                    let a02 = &pt[[0, 2]];
                    let a10 = &pt[[1, 0]];
                    let a11 = &pt[[1, 1]];
                    let a12 = &pt[[1, 2]];
                    let a20 = &pt[[2, 0]];
                    let a21 = &pt[[2, 1]];
                    let a22 = &pt[[2, 2]];

                    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
                        + a02 * (a10 * a21 - a11 * a20)
                }
                _ => panic!("Determinant calculation for matrices larger than 3x3 not implemented"),
            }
        }
        det
    }

    /// Element-wise conjugate
    fn conj(&self) -> Self {
        self.clone()
    }

    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> Array1<f64> {
        let mut norm = Array1::zeros(self.npts());
        for (i, pt) in self.outer_iter().enumerate() {
            let mut sum = 0.0;
            for element in pt.iter() {
                sum += element.powi(2);
            }
            norm[i] = sum.sqrt()
        }
        norm
    }

    /// Point multiplication
    fn dot(&self, other: &Self) -> Self {
        if self.ncols() != other.nrows() {
            panic!(
                "Points dimensions incompatible for multiplication: {}x{} * {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            );
        }

        let mut result = Self::zeros((self.npts(), self.nrows(), other.ncols()));

        for i in 0..self.npts() {
            for j in 0..self.nrows() {
                for k in 0..other.ncols() {
                    let mut sum = 0.0;
                    for l in 0..self.ncols() {
                        sum += &self[[i, j, l]] * &other[[i, l, k]];
                    }
                    result[[i, j, k]] = sum;
                }
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

        let mut result = Self::zeros((self.npts(), 1, self.ncols()));
        for i in 0..self.npts() {
            for k in 0..self.ncols() {
                result[[i, 0, k]] = self[[i, index, k]].clone();
            }
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

        let mut result = Self::zeros((self.npts(), self.nrows(), 1));
        for i in 0..self.npts() {
            for j in 0..self.nrows() {
                result[[i, j, 0]] = self[[i, j, index]].clone();
            }
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

        for i in 0..self.npts() {
            for k in 0..self.ncols() {
                self[[i, index, k]] = row[[i, 0, k]].clone();
            }
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

        for i in 0..self.npts() {
            for j in 0..self.nrows() {
                self[[i, j, index]] = col[[i, j, 0]].clone();
            }
        }
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array3<f64> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array3<f64> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<f64>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(len: usize, rows: usize, cols: usize, value: f64) -> Self {
        Points(Array3::from_elem((len, rows, cols), value))
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

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    fn inv(&self) -> Points<f64> {
        self.try_inv().unwrap()
    }

    fn try_inv(&self) -> Result<Points<f64>, InversionError> {
        let (npts, rows, cols) = self.dim();

        // Check if matrix is square
        if rows != cols {
            return Err(InversionError::NotSquare(format!(
                "Matrix dimensions are {}x{}, expected square matrix",
                rows, cols
            )));
        }

        let n = rows;

        let mut inverse = Array3::zeros((npts, n, n));
        for i in 0..npts {
            // Create augmented matrix [A | I]
            let mut augmented = Array2::zeros((n, 2 * n));

            // Copy original matrix to left half
            for j in 0..n {
                for k in 0..n {
                    augmented[[j, k]] = self.0[[i, j, k]].clone();
                }
            }

            // Create identity matrix in right half
            for j in 0..n {
                augmented[[j, j + n]] = 1.0;
            }

            // Perform Gauss-Jordan elimination with partial pivoting
            for j in 0..n {
                // Find pivot row (row with largest absolute value in column j)
                let mut pivot_row = j;
                let mut max_abs = augmented[[j, j]].abs();

                for l in (j + 1)..n {
                    let abs_val = augmented[[l, j]].abs();
                    if abs_val > max_abs {
                        max_abs = abs_val;
                        pivot_row = l;
                    }
                }

                // Check for singularity
                if max_abs < 1e-12 {
                    return Err(InversionError::Singular(format!(
                        "Matrix is singular or nearly singular at pivot {}",
                        j
                    )));
                }

                // Swap rows if necessary
                if pivot_row != j {
                    for k in 0..(2 * n) {
                        let temp = augmented[[j, k]].clone();
                        augmented[[j, k]] = augmented[[pivot_row, k]].clone();
                        augmented[[pivot_row, k]] = temp;
                    }
                }

                // Scale pivot row to make diagonal element 1
                let pivot = augmented[[j, j]].clone();
                for k in 0..(2 * n) {
                    augmented[[j, k]] /= &pivot;
                }

                // Eliminate column i in all other rows
                for l in 0..n {
                    if l != j {
                        let factor = augmented[[l, j]].clone();
                        for k in 0..(2 * n) {
                            let temp = &augmented[[j, k]] * &factor;
                            augmented[[l, k]] -= &temp;
                        }
                    }
                }
            }

            // Extract the inverse matrix from the right half
            for j in 0..n {
                for k in 0..n {
                    inverse[[i, j, k]] = augmented[[j, k + n]].clone();
                }
            }
        }

        Ok(Points(inverse))
    }

    /// Solve the linear system Ax = b using LU decomposition
    fn solve_linear_system(&self, b: &ArrayView3<f64>) -> Result<Array3<f64>, InversionError> {
        let (a_npts, a_rows, a_cols) = self.dim();
        let (b_npts, b_rows, b_cols) = b.dim();

        if a_npts != b_npts {
            return Err(InversionError::DimensionMismatch(format!(
                "Matrix A has {} points but vector b has {} points",
                a_npts, b_npts
            )));
        }

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

        let mut x = Array3::zeros((a_npts, n, 1));
        for i in 0..a_npts {
            // Create augmented matrix [A | b]
            let mut augmented = Array2::zeros((n, n + 1));

            // Copy A
            for j in 0..n {
                for k in 0..n {
                    augmented[[j, k]] = self[[i, j, k]].clone();
                }
            }

            // Copy b
            for j in 0..n {
                augmented[[j, n]] = b[[i, j, 0]].clone();
            }

            // Forward elimination with partial pivoting
            for j in 0..n {
                // Find pivot
                let mut pivot_row = j;
                let mut max_abs = augmented[[j, j]].abs();

                for l in (j + 1)..n {
                    let abs_val = augmented[[l, j]].abs();
                    if abs_val > max_abs {
                        max_abs = abs_val;
                        pivot_row = l;
                    }
                }

                // Check for singularity
                if max_abs < 1e-12 {
                    return Err(InversionError::Singular(format!(
                        "Matrix is singular at pivot {}",
                        j
                    )));
                }

                // Swap rows if necessary
                if pivot_row != j {
                    for k in 0..(n + 1) {
                        let temp = augmented[[j, k]].clone();
                        augmented[[j, k]] = augmented[[pivot_row, k]].clone();
                        augmented[[pivot_row, k]] = temp;
                    }
                }

                // Eliminate below pivot
                for l in (j + 1)..n {
                    let factor = &augmented[[l, j]] / &augmented[[j, j]];
                    for k in j..(n + 1) {
                        let temp = &augmented[[j, k]] * &factor;
                        augmented[[l, k]] -= &temp;
                    }
                }
            }

            // Back substitution
            for j in (0..n).rev() {
                let mut sum = augmented[[j, n]].clone();
                for k in (j + 1)..n {
                    sum -= &augmented[[j, k]] * &x[[i, k, 0]];
                }
                x[[i, j, 0]] = sum / &augmented[[j, j]];
            }
        }

        Ok(x)
    }

    /// Check if a matrix is approximately equal to another matrix within a tolerance
    fn approx_eq(a: &ArrayView3<f64>, b: &ArrayView3<f64>, tol: f64) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        let (npts, rows, cols) = a.dim();

        for i in 0..npts {
            for j in 0..rows {
                for k in 0..cols {
                    let diff = &a[[i, j, k]] - &b[[i, j, k]];
                    if diff.abs() > tol {
                        return false;
                    }
                }
            }
        }

        true
    }
}

// Indexing
impl Index<(usize, usize, usize)> for Points<f64> {
    type Output = f64;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<[usize; 3]> for Points<f64> {
    type Output = f64;

    fn index(&self, index: [usize; 3]) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<(usize, usize, usize)> for Points<f64> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IndexMut<[usize; 3]> for Points<f64> {
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Addition implementations
impl Add for Points<f64> {
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

impl Add<&Points<f64>> for Points<f64> {
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

impl Add<Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn add(self, other: Points<f64>) -> Points<f64> {
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

impl Add<&Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn add(self, other: &Points<f64>) -> Points<f64> {
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

// Scalar addition with f64
impl Add<f64> for Points<f64> {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<f64> for &Points<f64> {
    type Output = Points<f64>;

    fn add(self, scalar: f64) -> Points<f64> {
        Points(self.0.map(|x| x + &scalar))
    }
}

impl Add<&f64> for Points<f64> {
    type Output = Self;

    fn add(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x + scalar))
    }
}

impl Add<&f64> for &Points<f64> {
    type Output = Points<f64>;

    fn add(self, scalar: &f64) -> Points<f64> {
        Points(self.0.map(|x| x + scalar))
    }
}

// Subtraction implementations
impl Sub for Points<f64> {
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

impl Sub<&Points<f64>> for Points<f64> {
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

impl Sub<Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn sub(self, other: Points<f64>) -> Points<f64> {
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

impl Sub<&Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn sub(self, other: &Points<f64>) -> Points<f64> {
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
impl Sub<f64> for Points<f64> {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<f64> for &Points<f64> {
    type Output = Points<f64>;

    fn sub(self, scalar: f64) -> Points<f64> {
        Points(self.0.map(|x| x - &scalar))
    }
}

impl Sub<&f64> for Points<f64> {
    type Output = Self;

    fn sub(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x - scalar))
    }
}

impl Sub<&f64> for &Points<f64> {
    type Output = Points<f64>;

    fn sub(self, scalar: &f64) -> Points<f64> {
        Points(self.0.map(|x| x - scalar))
    }
}

// Element-wise multiplication
impl Mul for Points<f64> {
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

impl Mul<&Points<f64>> for Points<f64> {
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

impl Mul<Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn mul(self, other: Points<f64>) -> Points<f64> {
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

impl Mul<&Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn mul(self, other: &Points<f64>) -> Points<f64> {
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
impl Mul<f64> for Points<f64> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<f64> for &Points<f64> {
    type Output = Points<f64>;

    fn mul(self, scalar: f64) -> Points<f64> {
        Points(self.0.map(|x| x * &scalar))
    }
}

impl Mul<&f64> for Points<f64> {
    type Output = Self;

    fn mul(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x * scalar))
    }
}

impl Mul<&f64> for &Points<f64> {
    type Output = Points<f64>;

    fn mul(self, scalar: &f64) -> Points<f64> {
        Points(self.0.map(|x| x * scalar))
    }
}

// Division implementations
impl Div for Points<f64> {
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

impl Div<&Points<f64>> for Points<f64> {
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

impl Div<Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn div(self, other: Points<f64>) -> Points<f64> {
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

impl Div<&Points<f64>> for &Points<f64> {
    type Output = Points<f64>;

    fn div(self, other: &Points<f64>) -> Points<f64> {
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
impl Div<f64> for Points<f64> {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<f64> for &Points<f64> {
    type Output = Points<f64>;

    fn div(self, scalar: f64) -> Points<f64> {
        Points(self.0.map(|x| x / &scalar))
    }
}

impl Div<&f64> for Points<f64> {
    type Output = Self;

    fn div(self, scalar: &f64) -> Self {
        Points(self.0.map(|x| x / scalar))
    }
}

impl Div<&f64> for &Points<f64> {
    type Output = Points<f64>;

    fn div(self, scalar: &f64) -> Points<f64> {
        Points(self.0.map(|x| x / scalar))
    }
}

// Negation
impl Neg for Points<f64> {
    type Output = Self;

    fn neg(self) -> Self {
        Points(-self.0)
    }
}

impl Neg for &Points<f64> {
    type Output = Points<f64>;

    fn neg(self) -> Points<f64> {
        Points(-&self.0)
    }
}

// Assignment operators
impl AddAssign for Points<f64> {
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

impl AddAssign<&Points<f64>> for Points<f64> {
    fn add_assign(&mut self, other: &Points<f64>) {
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

impl SubAssign for Points<f64> {
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

impl SubAssign<&Points<f64>> for Points<f64> {
    fn sub_assign(&mut self, other: &Points<f64>) {
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

impl MulAssign for Points<f64> {
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

impl MulAssign<&Points<f64>> for Points<f64> {
    fn mul_assign(&mut self, other: &Points<f64>) {
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

impl DivAssign for Points<f64> {
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

impl DivAssign<&Points<f64>> for Points<f64> {
    fn div_assign(&mut self, other: &Points<f64>) {
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
impl AddAssign<f64> for Points<f64> {
    fn add_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x + &scalar);
    }
}

impl AddAssign<&f64> for Points<f64> {
    fn add_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x + scalar);
    }
}

impl SubAssign<f64> for Points<f64> {
    fn sub_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x - &scalar);
    }
}

impl SubAssign<&f64> for Points<f64> {
    fn sub_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x - scalar);
    }
}

impl MulAssign<f64> for Points<f64> {
    fn mul_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x * &scalar);
    }
}

impl MulAssign<&f64> for Points<f64> {
    fn mul_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x * scalar);
    }
}

impl DivAssign<f64> for Points<f64> {
    fn div_assign(&mut self, scalar: f64) {
        self.0.map_inplace(|x| *x = &*x / &scalar);
    }
}

impl DivAssign<&f64> for Points<f64> {
    fn div_assign(&mut self, scalar: &f64) {
        self.0.map_inplace(|x| *x = &*x / scalar);
    }
}

// Traits
impl Clone for Points<f64> {
    fn clone(&self) -> Self {
        Points(self.0.clone())
    }
}

impl Default for Points<f64> {
    fn default() -> Self {
        Points::zeros((0, 0, 0))
    }
}

impl PartialEq for Points<f64> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl Zero for Points<f64> {
    fn zero() -> Self {
        Points::zeros((0, 0, 0))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

// Display implementation
impl fmt::Display for Points<f64> {
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

impl fmt::Debug for Points<f64> {
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
impl From<Array3<f64>> for Points<f64> {
    fn from(array: Array3<f64>) -> Self {
        Points(array)
    }
}

impl From<ArrayView3<'_, f64>> for Points<f64> {
    fn from(array: ArrayView3<f64>) -> Self {
        Points(array.to_owned())
    }
}

impl From<Points<f64>> for Array3<f64> {
    fn from(matrix: Points<f64>) -> Self {
        matrix.0
    }
}

impl From<Vec<Vec<Vec<f64>>>> for Points<f64> {
    fn from(data: Vec<Vec<Vec<f64>>>) -> Self {
        Points::from_vec_f64(data).expect("Invalid matrix data")
    }
}

impl From<Vec<Vec<Vec<(f64, f64)>>>> for Points<f64> {
    fn from(data: Vec<Vec<Vec<(f64, f64)>>>) -> Self {
        Points::from_vec_complex(data).expect("Invalid matrix data")
    }
}

#[cfg(test)]
mod points_f64_tests {
    use super::*;

    #[test]
    fn test_creation() {
        let zeros = Points::<f64>::zeros((2, 3, 4));
        assert_eq!(zeros.shape(), (2, 3, 4));
        assert!(zeros[(0, 0, 0)].is_zero());
        assert!(zeros[(1, 2, 3)].is_zero());

        let ones = Points::<f64>::ones((3, 2, 1));
        assert_eq!(ones.shape(), (3, 2, 1));
        assert!(ones[(0, 0, 0)].is_one());
        assert!(ones[(2, 1, 0)].is_one());
    }

    #[test]
    fn test_from_vec() {
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
        let matrix = Points::<f64>::from_vec(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3, 4));
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(0, 1, 2)], 7.0);
        assert_eq!(matrix[(0, 2, 3)], 12.0);
        assert_eq!(matrix[(1, 0, 0)], 13.0);
        assert_eq!(matrix[(1, 1, 1)], 18.0);
        assert_eq!(matrix[(1, 2, 3)], 24.0);
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
        let matrix = Points::<f64>::from_vec_f64(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3, 4));
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(0, 1, 2)], 7.0);
        assert_eq!(matrix[(0, 2, 3)], 12.0);
        assert_eq!(matrix[(1, 0, 0)], 13.0);
        assert_eq!(matrix[(1, 1, 1)], 18.0);
        assert_eq!(matrix[(1, 2, 3)], 24.0);
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
        let matrix = Points::<f64>::from_vec_complex(data).unwrap();

        assert_eq!(matrix.shape(), (2, 2, 2));
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(1, 1, 1)], 15.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::<f64>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let b = Points::<f64>::from_vec_f64(vec![
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
        ])
        .unwrap();

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[(0, 0, 0)], 6.0);
        assert_eq!(sum[(0, 1, 1)], 12.0);
        assert_eq!(sum[(1, 0, 0)], 14.0);
        assert_eq!(sum[(1, 1, 1)], 20.0);

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[(0, 0, 0)], 4.0);
        assert_eq!(diff[(0, 1, 1)], 4.0);
        assert_eq!(diff[(1, 0, 0)], 4.0);
        assert_eq!(diff[(1, 1, 1)], 4.0);

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[(0, 0, 0)], 5.0);
        assert_eq!(prod[(0, 1, 1)], 32.0);
        assert_eq!(prod[(1, 0, 0)], 45.0);
        assert_eq!(prod[(1, 1, 1)], 96.0);
    }

    #[test]
    fn test_scalar_operations() {
        let matrix = Points::<f64>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[(0, 0, 0)], 6.0);
        assert_eq!(added[(1, 1, 1)], 13.0);

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[(0, 0, 0)], 2.0);
        assert_eq!(multiplied[(0, 1, 1)], 8.0);
        assert_eq!(multiplied[(1, 0, 0)], 10.0);
        assert_eq!(multiplied[(1, 1, 1)], 16.0);
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix = Points::<f64>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let other = Points::<f64>::from_vec_f64(vec![
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
        ])
        .unwrap();

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[(0, 0, 0)], 6.0);
        assert_eq!(matrix[(1, 1, 1)], 20.0);

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[(0, 0, 0)], 12.0);
        assert_eq!(matrix[(1, 1, 1)], 40.0);
    }

    #[test]
    fn test_map_functions() {
        let matrix = Points::<f64>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[(0, 0, 0)], 1.0);
        assert_eq!(squared[(0, 1, 1)], 16.0);
        assert_eq!(squared[(1, 0, 0)], 25.0);
        assert_eq!(squared[(1, 1, 1)], 64.0);

        // Original should be unchanged
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(1, 1, 1)], 8.0);

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * 2.0);
        assert_eq!(matrix_mut[(0, 0, 0)], 2.0);
        assert_eq!(matrix_mut[(0, 1, 1)], 8.0);
        assert_eq!(matrix_mut[(1, 0, 0)], 10.0);
        assert_eq!(matrix_mut[(1, 1, 1)], 16.0);
    }

    #[test]
    fn test_display() {
        let matrix = Points::<f64>::from_vec_f64(vec![
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
        let matrix1 = Points::<f64>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let matrix2 = Points::<f64>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ])
        .unwrap();

        let matrix3 = Points::<f64>::from_vec_f64(vec![
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
        let matrix1 = Points::<f64>::zeros((2, 4, 5));
        let matrix2 = Points::<f64>::zeros((3, 2, 1));

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
        let matrix: Points<f64> = data.into();
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(1, 1, 1)], 8.0);

        // Test conversion to Array3
        let array: Array3<f64> = matrix.into();
        assert_eq!(array[(0, 0, 0)], 1.0);
        assert_eq!(array[(1, 1, 1)], 8.0);
    }
}
