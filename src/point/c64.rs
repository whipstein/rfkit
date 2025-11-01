use crate::error::InversionError;
use crate::mycomplex::MyComplex;
use crate::point::{Point, Pt};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num::complex::Complex64;
use num_traits::{One, Zero};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A matrix wrapper around ndarray::Array2 with Complex64 elements
impl Point<Complex64> {
    pub fn new(array: Array2<Complex64>) -> Self {
        Point(array)
    }
}

impl Pt<Complex64, f64> for Point<Complex64> {
    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: (usize, usize)) -> Self {
        Point(Array2::from_elem(shape, Complex64::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: (usize, usize)) -> Self {
        Point(Array2::from_elem(shape, Complex64::one()))
    }

    /// Create an identity matrix of given size
    fn eye(size: usize) -> Self {
        let mut matrix = Self::zeros((size, size));
        for i in 0..size {
            matrix[[i, i]] = Complex64::one();
        }
        matrix
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec(data: Vec<Vec<Complex64>>) -> Result<Self, &'static str> {
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
            for (j, &value) in row.iter().enumerate() {
                matrix[[i, j]] = value;
            }
        }

        Ok(matrix)
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
            for (j, &value) in row.iter().enumerate() {
                matrix[[i, j]] = Complex64::new(value, 0.0);
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector of f64 values
    fn from_vec_float(data: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        Self::from_vec_f64(data)
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
            for (j, &(real, imag)) in row.iter().enumerate() {
                matrix[[i, j]] = Complex64::new(real, imag);
            }
        }

        Ok(matrix)
    }

    /// Create a matrix from a 2D vector of complex tuples (real, imag)
    fn from_vec_complex(data: Vec<Vec<(f64, f64)>>) -> Result<Self, &'static str> {
        Self::from_vec_c64(data)
    }

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, &'static str> {
        if data.len() != rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        let mut matrix = Self::zeros((rows, cols));
        for (idx, &value) in data.iter().enumerate() {
            let i = idx / cols;
            let j = idx % cols;
            matrix[[i, j]] = Complex64::new(value, 0.0);
        }

        Ok(matrix)
    }

    fn from_shape_fn<F>(shape: (usize, usize), f: F) -> Self
    where
        F: Fn((usize, usize)) -> Complex64,
    {
        Point(Array2::<Complex64>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(shape: (usize, usize), v: Vec<Complex64>) -> Result<Self, &'static str> {
        match Array2::<Complex64>::from_shape_vec(shape, v) {
            Ok(x) => Ok(Point(x)),
            Err(_) => Err("Cannot create vector"),
        }
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, Complex64, ndarray::Ix2> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, Complex64, ndarray::Ix2> {
        self.0.indexed_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, Complex64, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix2>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, Complex64, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix2>,
    {
        self.0.slice_mut(info)
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
    fn dim(&self) -> (usize, usize) {
        self.0.dim()
    }

    /// Get the shape as (rows, cols)
    fn shape(&self) -> (usize, usize) {
        let shape = self.0.dim();
        (shape.0, shape.1)
    }

    /// Check if the matrix is square
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView2<'_, Complex64> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.0.view_mut()
    }

    /// Transpose the matrix
    fn t(&self) -> Self {
        Point(self.0.t().to_owned())
    }

    /// Transpose the matrix
    fn transpose(&self) -> Self {
        self.t()
    }

    /// Get the conjugate transpose (Hermitian transpose)
    fn h(&self) -> Self {
        let transposed = self.transpose();
        let mut result = transposed;
        for element in result.0.iter_mut() {
            *element = element.conj();
        }
        result
    }

    /// Get the conjugate transpose (Hermitian transpose)
    fn conj_transpose(&self) -> Self {
        self.h()
    }

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> Complex64 {
        if !self.is_square() {
            panic!("Trace is only defined for square matrices");
        }

        let mut sum = Complex64::zero();
        for i in 0..self.nrows() {
            sum += &self[[i, i]];
        }
        sum
    }

    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Complex64 {
        if !self.is_square() {
            panic!("Determinant is only defined for square matrices");
        }

        match self.nrows() {
            0 => Complex64::one(),
            1 => self[[0, 0]].clone(),
            2 => &self[[0, 0]] * &self[[1, 1]] - &self[[0, 1]] * &self[[1, 0]],
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

                a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
                    + a02 * (a10 * a21 - a11 * a20)
            }
            _ => panic!("Determinant calculation for matrices larger than 3x3 not implemented"),
        }
    }

    /// Element-wise conjugate
    fn conj(&self) -> Self {
        let mut result = self.clone();
        for element in result.0.iter_mut() {
            *element = element.conj();
        }
        result
    }

    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for element in self.0.iter() {
            sum += element.norm_sqr();
        }
        sum.sqrt()
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
                let mut sum = Complex64::zero();
                for k in 0..self.ncols() {
                    sum += &self[[i, k]] * &other[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    // fn not(&self) -> Self {
    //     let mut result = self.clone();
    //     for element in result.0.iter_mut() {
    //         *element = element.conj();
    //     }
    //     result
    // }

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
    fn inner(&self) -> &Array2<Complex64> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array2<Complex64> {
        self.0
    }

    /// Create a matrix filled with the given value
    fn fill(rows: usize, cols: usize, value: Complex64) -> Self {
        Point(Array2::from_elem((rows, cols), value))
    }

    /// Apply a function element-wise
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&Complex64) -> Complex64,
    {
        Point(self.0.map(&f))
    }

    /// Apply a function element-wise in place
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&Complex64) -> Complex64,
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
    /// * `Ok(Array2<Complex64>)` - The inverted matrix
    /// * `Err(InversionError)` - If the matrix is not square, singular, or other errors
    ///
    /// # Examples
    /// ```rust
    /// use ndarray::Array2;
    ///
    /// // Create a 2x2 matrix
    /// let matrix = Array2::from_shape_vec((2, 2), vec![
    ///     Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
    ///     Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let inv_matrix = Complex64::matrix_inverse(&matrix.view())?;
    /// ```
    fn inv(&self) -> Point<Complex64> {
        self.try_inv().unwrap()
    }

    fn try_inv(&self) -> Result<Point<Complex64>, InversionError> {
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
            augmented[[i, i + n]] = Complex64::one();
        }

        // Perform Gauss-Jordan elimination with partial pivoting
        for i in 0..n {
            // Find pivot row (row with largest absolute value in column i)
            let mut pivot_row = i;
            let mut max_abs = augmented[[i, i]].norm();

            for k in (i + 1)..n {
                let abs_val = augmented[[k, i]].norm();
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

        Ok(Point(inverse))
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
        b: &ArrayView2<Complex64>,
    ) -> Result<Array2<Complex64>, InversionError> {
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
            let mut max_abs = augmented[[i, i]].norm();

            for k in (i + 1)..n {
                let abs_val = augmented[[k, i]].norm();
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
    fn approx_eq(a: &ArrayView2<Complex64>, b: &ArrayView2<Complex64>, tol: f64) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        let (rows, cols) = a.dim();

        for i in 0..rows {
            for j in 0..cols {
                let diff = &a[[i, j]] - &b[[i, j]];
                if diff.norm() > tol {
                    return false;
                }
            }
        }

        true
    }
}

// Indexing
impl Index<(usize, usize)> for Point<Complex64> {
    type Output = Complex64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<[usize; 2]> for Point<Complex64> {
    type Output = Complex64;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<(usize, usize)> for Point<Complex64> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IndexMut<[usize; 2]> for Point<Complex64> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Addition implementations
impl Add for Point<Complex64> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 + &other.0)
    }
}

impl Add<&Point<Complex64>> for Point<Complex64> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 + &other.0)
    }
}

impl Add<Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn add(self, other: Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 + &other.0)
    }
}

impl Add<&Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn add(self, other: &Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for addition: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 + &other.0)
    }
}

// Scalar addition
impl Add<Complex64> for Point<Complex64> {
    type Output = Self;

    fn add(self, scalar: Complex64) -> Self {
        Point(self.0.map(|x| x + &scalar))
    }
}

impl Add<Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn add(self, scalar: Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x + &scalar))
    }
}

impl Add<&Complex64> for Point<Complex64> {
    type Output = Self;

    fn add(self, scalar: &Complex64) -> Self {
        Point(self.0.map(|x| x + scalar))
    }
}

impl Add<&Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn add(self, scalar: &Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x + scalar))
    }
}

impl Add<f64> for Point<Complex64> {
    type Output = Self;

    fn add(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self + scalar_complex
    }
}

impl Add<f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn add(self, scalar: f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self + scalar_complex
    }
}

impl Add<&f64> for Point<Complex64> {
    type Output = Self;

    fn add(self, scalar: &f64) -> Self {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self + scalar_complex
    }
}

impl Add<&f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn add(self, scalar: &f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self + scalar_complex
    }
}

// Subtraction implementations
impl Sub for Point<Complex64> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 - &other.0)
    }
}

impl Sub<&Point<Complex64>> for Point<Complex64> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 - &other.0)
    }
}

impl Sub<Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn sub(self, other: Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 - &other.0)
    }
}

impl Sub<&Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn sub(self, other: &Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for subtraction: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 - &other.0)
    }
}

// Scalar subtraction
impl Sub<Complex64> for Point<Complex64> {
    type Output = Self;

    fn sub(self, scalar: Complex64) -> Self {
        Point(self.0.map(|x| x - &scalar))
    }
}

impl Sub<Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn sub(self, scalar: Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x - &scalar))
    }
}

impl Sub<&Complex64> for Point<Complex64> {
    type Output = Self;

    fn sub(self, scalar: &Complex64) -> Self {
        Point(self.0.map(|x| x - scalar))
    }
}

impl Sub<&Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn sub(self, scalar: &Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x - scalar))
    }
}

impl Sub<f64> for Point<Complex64> {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self - scalar_complex
    }
}

impl Sub<f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn sub(self, scalar: f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self - scalar_complex
    }
}

impl Sub<&f64> for Point<Complex64> {
    type Output = Self;

    fn sub(self, scalar: &f64) -> Self {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self - scalar_complex
    }
}

impl Sub<&f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn sub(self, scalar: &f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self - scalar_complex
    }
}

// Element-wise multiplication
impl Mul for Point<Complex64> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 * &other.0)
    }
}

impl Mul<&Point<Complex64>> for Point<Complex64> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 * &other.0)
    }
}

impl Mul<Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn mul(self, other: Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 * &other.0)
    }
}

impl Mul<&Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn mul(self, other: &Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise multiplication: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 * &other.0)
    }
}

// Scalar multiplication
impl Mul<Complex64> for Point<Complex64> {
    type Output = Self;

    fn mul(self, scalar: Complex64) -> Self {
        Point(self.0.map(|x| x * &scalar))
    }
}

impl Mul<Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn mul(self, scalar: Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x * &scalar))
    }
}

impl Mul<&Complex64> for Point<Complex64> {
    type Output = Self;

    fn mul(self, scalar: &Complex64) -> Self {
        Point(self.0.map(|x| x * scalar))
    }
}

impl Mul<&Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn mul(self, scalar: &Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x * scalar))
    }
}

impl Mul<f64> for Point<Complex64> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self * scalar_complex
    }
}

impl Mul<f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn mul(self, scalar: f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self * scalar_complex
    }
}

impl Mul<&f64> for Point<Complex64> {
    type Output = Self;

    fn mul(self, scalar: &f64) -> Self {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self * scalar_complex
    }
}

impl Mul<&f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn mul(self, scalar: &f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self * scalar_complex
    }
}

// Division implementations
impl Div for Point<Complex64> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 / &other.0)
    }
}

impl Div<&Point<Complex64>> for Point<Complex64> {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 / &other.0)
    }
}

impl Div<Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn div(self, other: Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 / &other.0)
    }
}

impl Div<&Point<Complex64>> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn div(self, other: &Point<Complex64>) -> Point<Complex64> {
        if self.shape() != other.shape() {
            panic!(
                "Point dimensions must match for element-wise division: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }
        Point(&self.0 / &other.0)
    }
}

// Scalar division
impl Div<Complex64> for Point<Complex64> {
    type Output = Self;

    fn div(self, scalar: Complex64) -> Self {
        Point(self.0.map(|x| x / &scalar))
    }
}

impl Div<Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn div(self, scalar: Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x / &scalar))
    }
}

impl Div<&Complex64> for Point<Complex64> {
    type Output = Self;

    fn div(self, scalar: &Complex64) -> Self {
        Point(self.0.map(|x| x / scalar))
    }
}

impl Div<&Complex64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn div(self, scalar: &Complex64) -> Point<Complex64> {
        Point(self.0.map(|x| x / scalar))
    }
}

impl Div<f64> for Point<Complex64> {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self / scalar_complex
    }
}

impl Div<f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn div(self, scalar: f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(scalar, 0.0);
        self / scalar_complex
    }
}

impl Div<&f64> for Point<Complex64> {
    type Output = Self;

    fn div(self, scalar: &f64) -> Self {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self / scalar_complex
    }
}

impl Div<&f64> for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn div(self, scalar: &f64) -> Point<Complex64> {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        self / scalar_complex
    }
}

// Negation
impl Neg for Point<Complex64> {
    type Output = Self;

    fn neg(self) -> Self {
        Point(-self.0)
    }
}

impl Neg for &Point<Complex64> {
    type Output = Point<Complex64>;

    fn neg(self) -> Point<Complex64> {
        Point(-&self.0)
    }
}

// Assignment operators
impl AddAssign for Point<Complex64> {
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

impl AddAssign<&Point<Complex64>> for Point<Complex64> {
    fn add_assign(&mut self, other: &Point<Complex64>) {
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

impl SubAssign for Point<Complex64> {
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

impl SubAssign<&Point<Complex64>> for Point<Complex64> {
    fn sub_assign(&mut self, other: &Point<Complex64>) {
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

impl MulAssign for Point<Complex64> {
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

impl MulAssign<&Point<Complex64>> for Point<Complex64> {
    fn mul_assign(&mut self, other: &Point<Complex64>) {
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

impl DivAssign for Point<Complex64> {
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

impl DivAssign<&Point<Complex64>> for Point<Complex64> {
    fn div_assign(&mut self, other: &Point<Complex64>) {
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
impl AddAssign<Complex64> for Point<Complex64> {
    fn add_assign(&mut self, scalar: Complex64) {
        self.0.map_inplace(|x| *x = &*x + &scalar);
    }
}

impl AddAssign<&Complex64> for Point<Complex64> {
    fn add_assign(&mut self, scalar: &Complex64) {
        self.0.map_inplace(|x| *x = &*x + scalar);
    }
}

impl AddAssign<f64> for Point<Complex64> {
    fn add_assign(&mut self, scalar: f64) {
        let scalar_complex = Complex64::new(scalar, 0.0);
        *self += scalar_complex;
    }
}

impl AddAssign<&f64> for Point<Complex64> {
    fn add_assign(&mut self, scalar: &f64) {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        *self += scalar_complex;
    }
}

impl SubAssign<Complex64> for Point<Complex64> {
    fn sub_assign(&mut self, scalar: Complex64) {
        self.0.map_inplace(|x| *x = &*x - &scalar);
    }
}

impl SubAssign<&Complex64> for Point<Complex64> {
    fn sub_assign(&mut self, scalar: &Complex64) {
        self.0.map_inplace(|x| *x = &*x - scalar);
    }
}

impl SubAssign<f64> for Point<Complex64> {
    fn sub_assign(&mut self, scalar: f64) {
        let scalar_complex = Complex64::new(scalar, 0.0);
        *self -= scalar_complex;
    }
}

impl SubAssign<&f64> for Point<Complex64> {
    fn sub_assign(&mut self, scalar: &f64) {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        *self -= scalar_complex;
    }
}

impl MulAssign<Complex64> for Point<Complex64> {
    fn mul_assign(&mut self, scalar: Complex64) {
        self.0.map_inplace(|x| *x = &*x * &scalar);
    }
}

impl MulAssign<&Complex64> for Point<Complex64> {
    fn mul_assign(&mut self, scalar: &Complex64) {
        self.0.map_inplace(|x| *x = &*x * scalar);
    }
}

impl MulAssign<f64> for Point<Complex64> {
    fn mul_assign(&mut self, scalar: f64) {
        let scalar_complex = Complex64::new(scalar, 0.0);
        *self *= scalar_complex;
    }
}

impl MulAssign<&f64> for Point<Complex64> {
    fn mul_assign(&mut self, scalar: &f64) {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        *self *= scalar_complex;
    }
}

impl DivAssign<Complex64> for Point<Complex64> {
    fn div_assign(&mut self, scalar: Complex64) {
        self.0.map_inplace(|x| *x = &*x / &scalar);
    }
}

impl DivAssign<&Complex64> for Point<Complex64> {
    fn div_assign(&mut self, scalar: &Complex64) {
        self.0.map_inplace(|x| *x = &*x / scalar);
    }
}

impl DivAssign<f64> for Point<Complex64> {
    fn div_assign(&mut self, scalar: f64) {
        let scalar_complex = Complex64::new(scalar, 0.0);
        *self /= scalar_complex;
    }
}

impl DivAssign<&f64> for Point<Complex64> {
    fn div_assign(&mut self, scalar: &f64) {
        let scalar_complex = Complex64::new(*scalar, 0.0);
        *self /= scalar_complex;
    }
}

// Traits
impl Clone for Point<Complex64> {
    fn clone(&self) -> Self {
        Point(self.0.clone())
    }
}

impl Default for Point<Complex64> {
    fn default() -> Self {
        Point::zeros((0, 0))
    }
}

impl PartialEq for Point<Complex64> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl Zero for Point<Complex64> {
    fn zero() -> Self {
        Point::zeros((0, 0))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

// Display implementation
impl fmt::Display for Point<Complex64> {
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

impl fmt::Debug for Point<Complex64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point({}x{}) {}", self.nrows(), self.ncols(), self)
    }
}

// Conversion traits
impl From<Array2<Complex64>> for Point<Complex64> {
    fn from(array: Array2<Complex64>) -> Self {
        Point(array)
    }
}

impl From<ArrayView2<'_, Complex64>> for Point<Complex64> {
    fn from(array: ArrayView2<Complex64>) -> Self {
        Point(array.to_owned())
    }
}

impl From<Point<Complex64>> for Array2<Complex64> {
    fn from(matrix: Point<Complex64>) -> Self {
        matrix.0
    }
}

impl From<Vec<Vec<f64>>> for Point<Complex64> {
    fn from(data: Vec<Vec<f64>>) -> Self {
        Point::from_vec_float(data).expect("Invalid matrix data")
    }
}

impl From<Vec<Vec<(f64, f64)>>> for Point<Complex64> {
    fn from(data: Vec<Vec<(f64, f64)>>) -> Self {
        Point::from_vec_complex(data).expect("Invalid matrix data")
    }
}

impl From<Point<MyComplex>> for Point<Complex64> {
    fn from(point: Point<MyComplex>) -> Self {
        Point::from_shape_fn(point.dim(), |(j, k)| (&point[[j, k]]).into())
    }
}

impl From<&Point<MyComplex>> for Point<Complex64> {
    fn from(point: &Point<MyComplex>) -> Self {
        Point::from_shape_fn(point.dim(), |(j, k)| (&point[[j, k]]).into())
    }
}

#[cfg(test)]
mod point_c64_tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn test_creation() {
        let zeros = Point::<Complex64>::zeros((2, 3));
        assert_eq!(zeros.shape(), (2, 3));
        assert!(zeros[[0, 0]].is_zero());
        assert!(zeros[[1, 2]].is_zero());

        let ones = Point::<Complex64>::ones((3, 2));
        assert_eq!(ones.shape(), (3, 2));
        assert!(ones[[0, 0]].is_one());
        assert!(ones[[2, 1]].is_one());

        let eye = Point::<Complex64>::eye(3);
        assert_eq!(eye.shape(), (3, 3));
        assert!(eye[[0, 0]].is_one());
        assert!(eye[[1, 1]].is_one());
        assert!(eye[[2, 2]].is_one());
        assert!(eye[[0, 1]].is_zero());
        assert!(eye[[1, 0]].is_zero());
    }

    #[test]
    fn test_from_vec() {
        let data: Vec<Vec<Complex64>> = vec![
            vec![1.0.into(), 2.0.into(), 3.0.into()],
            vec![4.0.into(), 5.0.into(), 6.0.into()],
        ];
        let matrix = Point::<Complex64>::from_vec(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[0, 1]].re, 2.0);
        assert_eq!(matrix[[0, 2]].re, 3.0);
        assert_eq!(matrix[[1, 0]].re, 4.0);
        assert_eq!(matrix[[1, 1]].re, 5.0);
        assert_eq!(matrix[[1, 2]].re, 6.0);
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Point::<Complex64>::from_vec_f64(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[0, 1]].re, 2.0);
        assert_eq!(matrix[[0, 2]].re, 3.0);
        assert_eq!(matrix[[1, 0]].re, 4.0);
        assert_eq!(matrix[[1, 1]].re, 5.0);
        assert_eq!(matrix[[1, 2]].re, 6.0);
    }

    #[test]
    fn test_from_vec_float() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Point::<Complex64>::from_vec_float(data).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[0, 1]].re, 2.0);
        assert_eq!(matrix[[0, 2]].re, 3.0);
        assert_eq!(matrix[[1, 0]].re, 4.0);
        assert_eq!(matrix[[1, 1]].re, 5.0);
        assert_eq!(matrix[[1, 2]].re, 6.0);
    }

    #[test]
    fn test_from_vec_c64() {
        let data = vec![vec![(1.0, 2.0), (3.0, 4.0)], vec![(5.0, 6.0), (7.0, 8.0)]];
        let matrix = Point::<Complex64>::from_vec_c64(data).unwrap();

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[0, 0]].im, 2.0);
        assert_eq!(matrix[[1, 1]].re, 7.0);
        assert_eq!(matrix[[1, 1]].im, 8.0);
    }

    #[test]
    fn test_from_vec_complex() {
        let data = vec![vec![(1.0, 2.0), (3.0, 4.0)], vec![(5.0, 6.0), (7.0, 8.0)]];
        let matrix = Point::<Complex64>::from_vec_complex(data).unwrap();

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[0, 0]].im, 2.0);
        assert_eq!(matrix[[1, 1]].re, 7.0);
        assert_eq!(matrix[[1, 1]].im, 8.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Point::<Complex64>::from_vec_float(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[[0, 0]].re, 6.0);
        assert_eq!(sum[[0, 1]].re, 8.0);
        assert_eq!(sum[[1, 0]].re, 10.0);
        assert_eq!(sum[[1, 1]].re, 12.0);

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[[0, 0]].re, 4.0);
        assert_eq!(diff[[0, 1]].re, 4.0);
        assert_eq!(diff[[1, 0]].re, 4.0);
        assert_eq!(diff[[1, 1]].re, 4.0);

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[[0, 0]].re, 5.0);
        assert_eq!(prod[[0, 1]].re, 12.0);
        assert_eq!(prod[[1, 0]].re, 21.0);
        assert_eq!(prod[[1, 1]].re, 32.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Point::<Complex64>::from_vec_float(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        let result = a.dot(&b);

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result[[0, 0]].re, 19.0);
        assert_eq!(result[[0, 1]].re, 22.0);
        assert_eq!(result[[1, 0]].re, 43.0);
        assert_eq!(result[[1, 1]].re, 50.0);
    }

    #[test]
    fn test_transpose() {
        let matrix =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
                .unwrap();

        let transposed = matrix.transpose();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed[[0, 0]].re, 1.0);
        assert_eq!(transposed[[0, 1]].re, 4.0);
        assert_eq!(transposed[[1, 0]].re, 2.0);
        assert_eq!(transposed[[1, 1]].re, 5.0);
        assert_eq!(transposed[[2, 0]].re, 3.0);
        assert_eq!(transposed[[2, 1]].re, 6.0);
    }

    #[test]
    fn test_conjugate_transpose() {
        let matrix = Point::<Complex64>::from_vec_complex(vec![
            vec![(1.0, 2.0), (3.0, 4.0)],
            vec![(5.0, 6.0), (7.0, 8.0)],
        ])
        .unwrap();

        let conj_t = matrix.conj_transpose();
        assert_eq!(conj_t.shape(), (2, 2));

        // Original: [(1+2i, 3+4i), (5+6i, 7+8i)]
        // Conj transpose: [(1-2i, 5-6i), (3-4i, 7-8i)]
        assert_eq!(conj_t[[0, 0]].re, 1.0);
        assert_eq!(conj_t[[0, 0]].im, -2.0);
        assert_eq!(conj_t[[0, 1]].re, 5.0);
        assert_eq!(conj_t[[0, 1]].im, -6.0);
        assert_eq!(conj_t[[1, 0]].re, 3.0);
        assert_eq!(conj_t[[1, 0]].im, -4.0);
        assert_eq!(conj_t[[1, 1]].re, 7.0);
        assert_eq!(conj_t[[1, 1]].im, -8.0);
    }

    #[test]
    fn test_trace() {
        let matrix = Point::<Complex64>::from_vec_float(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        let trace = matrix.trace();
        assert_eq!(trace.re, 15.0); // 1 + 5 + 9 = 15
        assert_eq!(trace.im, 0.0);
    }

    #[test]
    fn test_determinant() {
        // Test 2x2 determinant
        let matrix2x2 =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let det2 = matrix2x2.det();
        assert_eq!(det2.re, -2.0); // 1*4 - 2*3 = -2

        // Test 3x3 determinant
        let matrix3x3 = Point::<Complex64>::from_vec_float(vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ])
        .unwrap();

        let det3 = matrix3x3.det();
        assert_eq!(det3.re, 1.0); // Should be 1
    }

    #[test]
    fn test_scalar_operations() {
        let matrix =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[[0, 0]].re, 6.0);
        assert_eq!(added[[1, 1]].re, 9.0);

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[[0, 0]].re, 2.0);
        assert_eq!(multiplied[[0, 1]].re, 4.0);
        assert_eq!(multiplied[[1, 0]].re, 6.0);
        assert_eq!(multiplied[[1, 1]].re, 8.0);
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let other =
            Point::<Complex64>::from_vec_float(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[[0, 0]].re, 6.0);
        assert_eq!(matrix[[1, 1]].re, 12.0);

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[[0, 0]].re, 12.0);
        assert_eq!(matrix[[1, 1]].re, 24.0);
    }

    #[test]
    fn test_row_col_operations() {
        let matrix = Point::<Complex64>::from_vec_float(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        // Test getting a row
        let row1 = matrix.row(1);
        assert_eq!(row1.shape(), (1, 3));
        assert_eq!(row1[[0, 0]].re, 4.0);
        assert_eq!(row1[[0, 1]].re, 5.0);
        assert_eq!(row1[[0, 2]].re, 6.0);

        // Test getting a column
        let col2 = matrix.col(2);
        assert_eq!(col2.shape(), (3, 1));
        assert_eq!(col2[[0, 0]].re, 3.0);
        assert_eq!(col2[[1, 0]].re, 6.0);
        assert_eq!(col2[[2, 0]].re, 9.0);

        // Test setting a row
        let mut matrix_mut = matrix.clone();
        let new_row = Point::<Complex64>::from_vec_float(vec![vec![10.0, 11.0, 12.0]]).unwrap();
        matrix_mut.set_row(0, &new_row);
        assert_eq!(matrix_mut[[0, 0]].re, 10.0);
        assert_eq!(matrix_mut[[0, 1]].re, 11.0);
        assert_eq!(matrix_mut[[0, 2]].re, 12.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix =
            Point::<Complex64>::from_vec_float(vec![vec![3.0, 4.0], vec![0.0, 0.0]]).unwrap();

        let norm = matrix.frobenius_norm();
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_map_functions() {
        let matrix =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[[0, 0]].re, 1.0);
        assert_eq!(squared[[0, 1]].re, 4.0);
        assert_eq!(squared[[1, 0]].re, 9.0);
        assert_eq!(squared[[1, 1]].re, 16.0);

        // Original should be unchanged
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[1, 1]].re, 4.0);

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * &Complex64::new(2.0, 0.0));
        assert_eq!(matrix_mut[[0, 0]].re, 2.0);
        assert_eq!(matrix_mut[[0, 1]].re, 4.0);
        assert_eq!(matrix_mut[[1, 0]].re, 6.0);
        assert_eq!(matrix_mut[[1, 1]].re, 8.0);
    }

    #[test]
    fn test_display() {
        let matrix =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let display_str = format!("{}", matrix);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
    }

    #[test]
    fn test_equality() {
        let matrix1 =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let matrix2 =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let matrix3 =
            Point::<Complex64>::from_vec_float(vec![vec![1.0, 2.0], vec![3.0, 5.0]]).unwrap();

        assert_eq!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
    }

    #[test]
    fn test_error_cases() {
        // Test mismatched dimensions for addition
        let matrix1 = Point::<Complex64>::zeros((2, 4));
        let matrix2 = Point::<Complex64>::zeros((3, 2));

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
        std::panic::catch_unwind(|| {
            let _ = matrix1.trace();
        })
        .expect_err("Should panic on trace of non-square matrix");
    }

    #[test]
    fn test_conversions() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        // Test From<Vec<Vec<f64>>>
        let matrix: Point<Complex64> = data.into();
        assert_eq!(matrix[[0, 0]].re, 1.0);
        assert_eq!(matrix[[1, 1]].re, 4.0);

        // Test conversion to Array2
        let array: Array2<Complex64> = matrix.into();
        assert_eq!(array[[0, 0]].re, 1.0);
        assert_eq!(array[[1, 1]].re, 4.0);
    }

    #[test]
    fn test_2x2_matrix_inversion() {
        // Test inverting a simple 2x2 matrix
        // A = [1 2]  =>  A^(-1) = [-2  1]
        //     [3 4]               [1.5 -0.5]

        let matrix = Point::<Complex64>::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Check specific values
        approx_eq!(f64, inverse[[0, 0]].re, -2.0, F64Margin::default());
        approx_eq!(f64, inverse[[0, 1]].re, 1.0, F64Margin::default());
        approx_eq!(f64, inverse[[1, 0]].re, 1.5, F64Margin::default());
        approx_eq!(f64, inverse[[1, 1]].re, -0.5, F64Margin::default());

        // Verify A * A^(-1) = I
        let product = &matrix.dot(&inverse);
        let identity = Point::eye(2);

        assert!(Point::approx_eq(&product.view(), &identity.view(), 1e-10));
    }

    #[test]
    fn test_complex_matrix_inversion() {
        // Test with complex numbers
        let matrix = Point::<Complex64>::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 1.0),
            ],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Verify A * A^(-1) = I
        let product = matrix.dot(&inverse);
        let identity = Point::<Complex64>::eye(2);

        assert!(Point::approx_eq(&product.view(), &identity.view(), 1e-10));
    }

    #[test]
    fn test_3x3_matrix_inversion() {
        // Test with a 3x3 matrix
        let matrix = Point::<Complex64>::from_shape_vec(
            (3, 3),
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Verify A * A^(-1) = I
        let product = matrix.dot(&inverse);
        let identity = Point::<Complex64>::eye(3);

        assert!(Point::approx_eq(&product.view(), &identity.view(), 1e-10));
    }

    #[test]
    fn test_singular_matrix() {
        // Test with a singular matrix (determinant = 0)
        let matrix = Point::<Complex64>::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0), // Second row is 2x first row
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
        let matrix = Point::<Complex64>::from_shape_vec(
            (2, 3),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
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
        let identity = Point::<Complex64>::eye(4);
        let inverse = identity.inv();

        assert!(Point::approx_eq(&identity.view(), &inverse.view(), 1e-10));
    }
}
