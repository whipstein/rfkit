use crate::{
    error::InversionError,
    num::{RFComplex, RFFloat, RFNum},
    pts::{Matrix, Points, Pts},
};
use ndarray::{Dimension, IntoDimension, SliceArg, SliceInfo, ViewRepr, linalg::Dot, prelude::*};
use ndarray_linalg::error::LinalgError;
use num::complex::Complex64;
use num_traits::{One, Zero};
use std::fmt;

// pub mod c64;
// pub mod f64;
// pub mod mycomplex;
// pub mod myfloat;

/// A matrix wrapper around ndarray::Array3 with T elements
// pub struct Points(Array3<T>);

impl<T> Points<T, Ix3>
where
    T: RFNum,
{
    fn check_vec<U>(vec: &Vec<Vec<Vec<U>>>) -> (usize, usize, usize) {
        let len = vec.len();
        let rows = vec[0].len();
        let cols = vec[0][0].len();

        // Check all rows have the same length
        assert!(
            vec.iter().all(|row| row.len() == rows),
            "All rows must have the same length"
        );

        // Check all cols have the same length
        assert!(
            vec.iter()
                .all(|row| row.iter().all(|col| col.len() == cols)),
            "All columns must have the same length"
        );

        (len, rows, cols)
    }

    /// Create a matrix from a 3D vector
    pub fn from_vec(data: Vec<Vec<Vec<T>>>) -> Self {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, pt) in data.iter().enumerate() {
            for (j, row) in pt.iter().enumerate() {
                for (k, value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = value.clone();
                }
            }
        }

        matrix
    }

    /// Create a matrix from a 3D vector
    pub fn from_real_vec(data: Vec<Vec<Vec<T::Real>>>) -> Self {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = <<T as RFNum>::Real as Into<T>>::into(value.clone());
                }
            }
        }

        matrix
    }

    pub fn to_vec(&self) -> Vec<Vec<Vec<T>>> {
        let mut out: Vec<Vec<Vec<T>>> = vec![];
        for pt in self.outer_iter() {
            let mut int: Vec<Vec<T>> = vec![];
            for row in pt.outer_iter() {
                int.push(row.to_vec());
            }
            out.push(int);
        }

        out
    }
}

impl<T> Points<T, Ix3>
where
    T: RFFloat,
{
    /// Create a matrix from a 3D vector of f64 values
    pub fn from_vec_f64(data: Vec<Vec<Vec<f64>>>) -> Self {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, &value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = T::from_f64(value);
                }
            }
        }

        matrix
    }

    /// Create a matrix from a 3D vector of float values
    pub fn from_vec_float<U>(data: Vec<Vec<Vec<U>>>) -> Self
    where
        T: From<U>,
        U: RFFloat,
    {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = T::from(value.clone());
                }
            }
        }

        matrix
    }
}

impl<T> Points<T, Ix3>
where
    T: RFComplex,
{
    /// Create a matrix from a 3D vector of Complex64
    pub fn from_vec_c64(data: Vec<Vec<Vec<Complex64>>>) -> Self {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, &value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = T::from(value);
                }
            }
        }

        matrix
    }

    /// Create a matrix from a 3D vector of complex
    pub fn from_vec_complex<U>(data: Vec<Vec<Vec<U>>>) -> Self
    where
        for<'a> T: From<&'a U>,
        U: RFComplex,
    {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, value) in row.iter().enumerate() {
                    matrix[(i, j, k)] = T::from(value);
                }
            }
        }

        matrix
    }

    /// Create a matrix from a 3D vector of real tuples (real, imag)
    pub fn from_vec_tuple<U>(data: Vec<Vec<Vec<(U, U)>>>) -> Self
    where
        T::Real: From<U>,
        U: RFFloat,
    {
        if data.is_empty() {
            return Self::new(Array3::from_elem((0, 0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, val) in data.iter().enumerate() {
            for (j, row) in val.iter().enumerate() {
                for (k, (real, imag)) in row.iter().enumerate() {
                    matrix[(i, j, k)] = T::new(
                        &<T as RFNum>::Real::from(real.clone()),
                        &<T as RFNum>::Real::from(imag.clone()),
                    );
                }
            }
        }

        matrix
    }
}

impl<T> Pts<T, Ix3> for Points<T, Ix3>
where
    T: RFNum,
{
    type Idx = (usize, usize, usize);
    type Tuple<'a>
        = &'a [(T, T)]
    where
        T: 'a;

    /// Create a new matrix with given dimensions filled with zeros
    fn zeros(shape: impl IntoDimension<Dim = Dim<[usize; 3]>>) -> Self {
        Points(Array3::from_elem(shape, T::zero()))
    }

    /// Create a new matrix with given dimensions filled with ones
    fn ones(shape: impl IntoDimension<Dim = Dim<[usize; 3]>>) -> Self {
        Points(Array3::from_elem(shape, T::one()))
    }

    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        shape: impl IntoDimension<Dim = Dim<[usize; 3]>>,
    ) -> Result<Self, &'static str> {
        let (depth, rows, cols) = shape.into_dimension().into_pattern();
        if data.len() != depth * rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        let mut matrix = Self::zeros((depth, rows, cols));
        for (idx, &value) in data.iter().enumerate() {
            let i = idx / (rows * cols);
            let j = idx / cols;
            let k = idx % cols;
            matrix[(i, j, k)] = T::from(value);
        }

        Ok(matrix)
    }

    fn from_elem(shape: impl IntoDimension<Dim = Ix3>, elem: T) -> Self {
        Points(Array3::from_elem(shape, elem))
    }

    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = Dim<[usize; 3]>>, f: F) -> Self
    where
        F: Fn(Self::Idx) -> T,
    {
        Points(Array3::<T>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(
        shape: impl IntoDimension<Dim = Dim<[usize; 3]>>,
        v: Vec<T>,
    ) -> Result<Self, &'static str> {
        match Array3::<T>::from_shape_vec(shape, v) {
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

    fn iter(&self) -> ndarray::iter::Iter<'_, T, Ix3> {
        self.0.iter()
    }

    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix3> {
        self.0.iter_mut()
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix2> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(&mut self, axis: Axis) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix2> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, T, ndarray::Ix3> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, T, ndarray::Ix3> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix2> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix2> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, T, I::OutDim>
    where
        I: SliceArg<Ix3>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, T, I::OutDim>
    where
        I: SliceArg<Ix3>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<T, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, T, ndarray::Ix2>,
    ) -> Result<(), ndarray::ShapeError>
    where
        T: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get and set outer axis point
    fn pt<I>(&self, index: usize) -> Points<T, I::OutDim>
    where
        I: SliceArg<Ix3, OutDim = Dim<[usize; 2]>>,
    {
        if index >= self.npts() {
            panic!(
                "Point index {} out of bounds for matrix with {} points",
                index,
                self.npts()
            );
        }

        Points(self.slice(s![index, .., ..]).to_owned())
    }

    fn pt_mut<I>(&mut self, index: usize) -> Points<T, I::OutDim>
    where
        I: SliceArg<Ix3, OutDim = Dim<[usize; 2]>>,
    {
        if index >= self.npts() {
            panic!(
                "Point index {} out of bounds for matrix with {} points",
                index,
                self.npts()
            );
        }

        Points(self.slice_mut(s![index, .., ..]).to_owned())
    }

    fn set_pt(&mut self, index: usize, pt: Points<T, Ix2>) {
        if index >= self.npts() {
            panic!(
                "Point index {} out of bounds for matrix with {} points",
                index,
                self.npts()
            );
        }
        if pt.nrows() != self.nrows() || pt.ncols() != self.ncols() {
            panic!(
                "Point dimensions incompatible: expected {}x{}, got {}x{}",
                self.nrows(),
                self.ncols(),
                pt.nrows(),
                pt.ncols()
            );
        }

        for j in 0..self.nrows() {
            for k in 0..self.ncols() {
                self[[index, j, k]] = pt[[j, k]].clone();
            }
        }
    }

    fn insert_axis(self, axis: Axis) -> Points<T, Ix4> {
        Points(self.inner().clone().insert_axis(axis))
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
        self.0.shape()[0]
    }

    /// Get the shape as (len, rows, cols)
    fn dim(&self) -> Self::Idx {
        self.0.dim()
    }

    /// Get the dimension
    fn raw_dim(&self) -> Ix3 {
        self.0.raw_dim()
    }

    /// Get the shape as (len, rows, cols)
    fn shape(&self) -> Self::Idx {
        let shape = self.0.dim();
        (shape.0, shape.1, shape.2)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView3<'_, T> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut3<'_, T> {
        self.0.view_mut()
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array3<T> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array3<T> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<T>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(shape: impl IntoDimension<Dim = Dim<[usize; 3]>>, value: T) -> Self {
        Points(Array3::from_elem(shape, value))
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
    fn approx_eq(a: &ArrayView3<T>, b: &ArrayView3<T>, tol: f64) -> bool
    where
        <T as RFNum>::Real: PartialOrd<f64>,
        for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    {
        if a.dim() != b.dim() {
            return false;
        }

        let (npts, rows, cols) = a.dim();

        for i in 0..npts {
            for j in 0..rows {
                for k in 0..cols {
                    let diff: T = &a[[i, j, k]] - &b[[i, j, k]];
                    if diff.norm() > tol {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl<T> Matrix<T, Ix3> for Points<T, Ix3>
where
    T: RFNum,
{
    /// Create an identity matrix of given size
    fn eye(size: impl IntoDimension<Dim = Dim<[usize; 2]>>) -> Self {
        let dim = size.into_dimension().into_pattern();
        Points(Array3::from_shape_fn((dim.0, dim.1, dim.1), |(_, j, k)| {
            if j == k { T::one() } else { T::zero() }
        }))
    }

    /// Get the number of rows
    fn nrows(&self) -> usize {
        self.0.shape()[1]
    }

    /// Get the number of cols
    fn ncols(&self) -> usize {
        self.0.shape()[2]
    }

    /// Check if the matrix is square
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
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
    fn trace(&self) -> Result<Array1<T>, LinalgError> {
        _ = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        };

        let mut trace = Array1::zeros(self.npts());
        for (i, pt) in self.outer_iter().enumerate() {
            let mut sum = T::zero();
            for j in 0..pt.nrows() {
                sum += &pt[[j, j]];
            }
            trace[i] = sum;
        }
        Ok(trace)
    }

    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Result<Array1<T>, LinalgError>
    where
        for<'a> &'a T: std::ops::Mul<T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    {
        _ = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        };

        let mut det = Array1::zeros(self.npts());
        for (i, pt) in self.outer_iter().enumerate() {
            let val = match pt.nrows() {
                0 => Ok(T::one()),
                1 => Ok(pt[[0, 0]].clone()),
                2 => Ok(&pt[[0, 0]] * &pt[[1, 1]] - &pt[[0, 1]] * &pt[[1, 0]]),
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

                    Ok(
                        a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
                            + a02 * (a10 * a21 - a11 * a20),
                    )
                }
                _ => Err(LinalgError::NotStandardShape {
                    obj: "Determinant calculation for matrices larger than 3x3 not implemented",
                    rows: self.nrows() as i32,
                    cols: self.ncols() as i32,
                }),
            };
            det[i] = val.unwrap();
        }
        Ok(det)
    }

    /// Element-wise conjugate
    fn conj(&self) -> Self {
        let mut result = self.0.clone();
        for element in result.iter_mut() {
            *element = element.conj();
        }
        Points(result)
    }

    /// Calculate the Frobenius norm
    fn frobenius_norm(&self) -> Array1<T::Real>
    where
        T::Real: RFNum,
    {
        let mut norm = Array1::<T::Real>::zeros(self.npts());
        for (i, pt) in self.outer_iter().enumerate() {
            let mut sum = T::Real::zero();
            for element in pt.iter() {
                sum += element.norm_sqr();
            }
            norm[i] = sum.sqrt()
        }
        norm
    }

    /// Get a row as a new matrix (1 x ncols)
    fn row(&self, index: usize) -> Points<T, Ix2> {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} rows",
                index,
                self.nrows()
            );
        }

        // let mut result = Self::zeros((self.npts(), 1, self.ncols()));
        // for i in 0..self.npts() {
        //     for k in 0..self.ncols() {
        //         result[[i, 0, k]] = self[[i, index, k]].clone();
        //     }
        // }
        // result
        let mut result = Points::<T, Ix2>::zeros((self.npts(), self.ncols()));
        for (i, pt) in self.outer_iter().enumerate() {
            for (j, val) in pt.row(index).into_iter().enumerate() {
                result[[i, j]] = val.clone();
            }
        }
        result
    }

    /// Get a column as a new matrix (nrows x 1)
    fn col(&self, index: usize) -> Points<T, Ix2> {
        if index >= self.ncols() {
            panic!(
                "Column index {} out of bounds for matrix with {} columns",
                index,
                self.ncols()
            );
        }

        // let mut result = Self::zeros((self.npts(), self.nrows(), 1));
        // for i in 0..self.npts() {
        //     for j in 0..self.nrows() {
        //         result[[i, j, 0]] = self[[i, j, index]].clone();
        //     }
        // }
        // result
        let mut result = Points::<T, Ix2>::zeros((self.npts(), self.nrows()));
        for (i, pt) in self.outer_iter().enumerate() {
            for (j, val) in pt.column(index).into_iter().enumerate() {
                result[[i, j]] = val.clone();
            }
        }
        result
    }

    /// Set a row from another matrix
    fn set_row(&mut self, index: usize, row: &Points<T, Ix2>) {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} rows",
                index,
                self.nrows()
            );
        }
        if row.ncols() != self.ncols() {
            panic!(
                "Row dimensions incompatible: expected {}, got {}",
                self.ncols(),
                row.ncols()
            );
        }

        for i in 0..self.npts() {
            for k in 0..self.ncols() {
                self[[i, index, k]] = row[[i, k]].clone();
            }
        }
    }

    /// Set a column from another matrix
    fn set_col(&mut self, index: usize, col: &Points<T, Ix2>) {
        if index >= self.ncols() {
            panic!(
                "Column index {} out of bounds for matrix with {} columns",
                index,
                self.ncols()
            );
        }
        if col.nrows() != self.nrows() {
            panic!(
                "Column dimensions incompatible: expected {}, got {}",
                self.nrows(),
                col.nrows()
            );
        }

        for i in 0..self.npts() {
            for j in 0..self.nrows() {
                self[[i, j, index]] = col[[i, j]].clone();
            }
        }
    }

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    fn inv(&self) -> Points<T, Ix3>
    where
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
        <T as RFNum>::Real: PartialOrd,
        <T as RFNum>::Real: PartialOrd<f64>,
    {
        self.try_inv().unwrap()
    }

    fn try_inv(&self) -> Result<Points<T, Ix3>, InversionError>
    where
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
        <T as RFNum>::Real: PartialOrd,
        <T as RFNum>::Real: PartialOrd<f64>,
    {
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
                augmented[[j, j + n]] = T::one();
            }

            // Perform Gauss-Jordan elimination with partial pivoting
            for j in 0..n {
                // Find pivot row (row with largest absolute value in column j)
                let mut pivot_row = j;
                let mut max_abs = augmented[[j, j]].norm();

                for l in (j + 1)..n {
                    let abs_val = augmented[[l, j]].norm();
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
    fn solve_linear_system(&self, b: &ArrayView3<T>) -> Result<Array3<T>, InversionError>
    where
        for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
        for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
        <T as RFNum>::Real: PartialOrd,
        <T as RFNum>::Real: PartialOrd<f64>,
    {
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
                let mut max_abs = augmented[[j, j]].norm();

                for l in (j + 1)..n {
                    let abs_val = augmented[[l, j]].norm();
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
}

// Dot product implementations
impl<T, U> Dot<Points<U, Ix2>> for Points<T, Ix3>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b U, Output = T>,
{
    type Output = Points<T, Ix3>;

    fn dot(&self, rhs: &Points<U, Ix2>) -> Self::Output {
        let mut result = Self::Output::zeros(self.dim());

        for i in 0..self.npts() {
            let a: Points<T, Ix2> = self
                .pt::<SliceInfo<[ndarray::SliceInfoElem; 3], Ix3, Ix2>>(i)
                .into();
            result.set_pt(i, a.dot(rhs));
        }

        result
    }
}

impl<T, U> Dot<Points<U, Ix3>> for Points<T, Ix3>
where
    T: RFNum,
    U: RFNum,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a> Points<T, ndarray::Dim<[usize; 2]>>: From<ndarray::ArrayBase<ViewRepr<&'a U>, ndarray::Dim<[usize; 2]>, U>>
        + From<Points<U, ndarray::Dim<[usize; 2]>>>,
{
    type Output = Points<T, Ix3>;

    fn dot(&self, rhs: &Points<U, Ix3>) -> Self::Output {
        let mut result = Self::Output::zeros(self.dim());

        for i in 0..self.npts() {
            let a: Points<T, Ix2> = self
                .pt::<SliceInfo<[ndarray::SliceInfoElem; 3], Ix3, Ix2>>(i)
                .into();
            let b: Points<T, Ix2> = rhs
                .pt::<SliceInfo<[ndarray::SliceInfoElem; 3], Ix3, Ix2>>(i)
                .into();
            result.set_pt(i, a.dot(&b));
        }

        result
    }
}

// Traits
impl<T> Default for Points<T, Ix3>
where
    T: RFNum,
{
    fn default() -> Self {
        Points::<T, Ix3>::zeros((0, 0, 0))
    }
}

impl<T> Zero for Points<T, Ix3>
where
    T: RFNum,
{
    fn zero() -> Self {
        Points::<T, Ix3>::zeros((0, 0, 0))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl<T> One for Points<T, Ix3>
where
    T: RFNum,
{
    fn one() -> Self {
        Points::<T, Ix3>::ones((0, 0, 0))
    }

    fn is_one(&self) -> bool {
        self.0.iter().all(|x| x.is_one())
    }
}

// Display implementation
impl<T> fmt::Display for Points<T, Ix3>
where
    T: RFNum,
{
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

impl<T> fmt::Debug for Points<T, Ix3>
where
    T: RFNum,
{
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
impl<T> From<Array3<T>> for Points<T, Ix3>
where
    T: RFNum,
{
    fn from(array: Array3<T>) -> Self {
        Points(array)
    }
}

impl<T> From<ArrayView3<'_, T>> for Points<T, Ix3>
where
    T: RFNum,
{
    fn from(array: ArrayView3<T>) -> Self {
        Points(array.to_owned())
    }
}

impl<T> From<Points<T, Ix3>> for Array3<T> {
    fn from(matrix: Points<T, Ix3>) -> Self {
        matrix.0
    }
}

impl<T> From<Vec<Vec<Vec<T>>>> for Points<T, Ix3>
where
    T: RFNum,
{
    fn from(data: Vec<Vec<Vec<T>>>) -> Self {
        Points::<T, Ix3>::from_vec(data)
    }
}

impl<T> From<Vec<Vec<Vec<(T::Real, T::Real)>>>> for Points<T, Ix3>
where
    T: RFComplex,
    <T as RFNum>::Real: RFFloat,
{
    fn from(data: Vec<Vec<Vec<(T::Real, T::Real)>>>) -> Self {
        Points::<T, Ix3>::from_vec_tuple(data)
    }
}

impl<T> From<&Points<T, Ix3>> for Points<T, Ix3>
where
    T: RFNum,
{
    fn from(point: &Points<T, Ix3>) -> Self {
        point.clone()
    }
}

#[cfg(test)]
mod points_ix3_f64_tests {
    use super::*;
    use crate::{num::MyFloat, util::comp_pts_ix3};
    use float_cmp::F64Margin;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn test_creation() {
        let zeros = Points::<f64, Ix3>::zeros((2, 3, 4));
        assert_eq!(zeros.shape(), (2, 3, 4));
        assert!(zeros[(0, 0, 0)].is_zero());
        assert!(zeros[(1, 2, 3)].is_zero());
        comp_pts_ix3(
            &array![
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]
                ]
            ]
            .into(),
            &zeros,
            MARGIN,
            "zeros()",
        );

        let ones = Points::<f64, Ix3>::ones((3, 2, 1));
        assert_eq!(ones.shape(), (3, 2, 1));
        assert!(ones[(0, 0, 0)].is_one());
        assert!(ones[(2, 1, 0)].is_one());
        comp_pts_ix3(
            &array![[[1.0], [1.0],], [[1.0], [1.0],], [[1.0], [1.0],]].into(),
            &ones,
            MARGIN,
            "ones()",
        );
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
        let matrix = Points::<f64, Ix3>::from_vec(data);

        assert_eq!(matrix.shape(), (2, 3, 4));
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(0, 1, 2)], 7.0);
        assert_eq!(matrix[(0, 2, 3)], 12.0);
        assert_eq!(matrix[(1, 0, 0)], 13.0);
        assert_eq!(matrix[(1, 1, 1)], 18.0);
        assert_eq!(matrix[(1, 2, 3)], 24.0);
        comp_pts_ix3(
            &array![
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0]
                ]
            ]
            .into(),
            &matrix,
            MARGIN,
            "from_vec()",
        );
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
        let matrix = Points::<f64, Ix3>::from_vec_f64(data);

        assert_eq!(matrix.shape(), (2, 3, 4));
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(0, 1, 2)], 7.0);
        assert_eq!(matrix[(0, 2, 3)], 12.0);
        assert_eq!(matrix[(1, 0, 0)], 13.0);
        assert_eq!(matrix[(1, 1, 1)], 18.0);
        assert_eq!(matrix[(1, 2, 3)], 24.0);
        comp_pts_ix3(
            &array![
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0]
                ]
            ]
            .into(),
            &matrix,
            MARGIN,
            "from_vec_f64()",
        );
    }

    #[test]
    fn test_from_vec_float() {
        let data: Vec<Vec<Vec<MyFloat>>> = vec![
            vec![
                vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()],
                vec![5.0.into(), 6.0.into(), 7.0.into(), 8.0.into()],
                vec![9.0.into(), 10.0.into(), 11.0.into(), 12.0.into()],
            ],
            vec![
                vec![13.0.into(), 14.0.into(), 15.0.into(), 16.0.into()],
                vec![17.0.into(), 18.0.into(), 19.0.into(), 20.0.into()],
                vec![21.0.into(), 22.0.into(), 23.0.into(), 24.0.into()],
            ],
        ];
        let matrix = Points::<f64, Ix3>::from_vec_float(data);

        assert_eq!(matrix.shape(), (2, 3, 4));
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(0, 1, 2)], 7.0);
        assert_eq!(matrix[(0, 2, 3)], 12.0);
        assert_eq!(matrix[(1, 0, 0)], 13.0);
        assert_eq!(matrix[(1, 1, 1)], 18.0);
        assert_eq!(matrix[(1, 2, 3)], 24.0);
        comp_pts_ix3(
            &array![
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0]
                ]
            ]
            .into(),
            &matrix,
            MARGIN,
            "from_vec_float()",
        );
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

        let b = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
        ]);

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[(0, 0, 0)], 6.0);
        assert_eq!(sum[(0, 1, 1)], 12.0);
        assert_eq!(sum[(1, 0, 0)], 14.0);
        assert_eq!(sum[(1, 1, 1)], 20.0);
        comp_pts_ix3(
            &array![[[6.0, 8.0], [10.0, 12.0],], [[14.0, 16.0], [18.0, 20.0],]].into(),
            &sum,
            MARGIN,
            "&a + &b",
        );

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[(0, 0, 0)], 4.0);
        assert_eq!(diff[(0, 1, 1)], 4.0);
        assert_eq!(diff[(1, 0, 0)], 4.0);
        assert_eq!(diff[(1, 1, 1)], 4.0);
        comp_pts_ix3(
            &array![[[4.0, 4.0], [4.0, 4.0],], [[4.0, 4.0], [4.0, 4.0],]].into(),
            &diff,
            MARGIN,
            "&a - &b",
        );

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[(0, 0, 0)], 5.0);
        assert_eq!(prod[(0, 1, 1)], 32.0);
        assert_eq!(prod[(1, 0, 0)], 45.0);
        assert_eq!(prod[(1, 1, 1)], 96.0);
        comp_pts_ix3(
            &array![[[5.0, 12.0], [21.0, 32.0],], [[45.0, 60.0], [77.0, 96.0],]].into(),
            &prod,
            MARGIN,
            "&a * &b",
        );
    }

    #[test]
    fn test_scalar_operations() {
        let matrix = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[(0, 0, 0)], 6.0);
        assert_eq!(added[(1, 1, 1)], 13.0);
        comp_pts_ix3(
            &array![[[6.0, 7.0], [8.0, 9.0],], [[10.0, 11.0], [12.0, 13.0]]].into(),
            &added,
            MARGIN,
            "&a + 5.0",
        );

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[(0, 0, 0)], 2.0);
        assert_eq!(multiplied[(0, 1, 1)], 8.0);
        assert_eq!(multiplied[(1, 0, 0)], 10.0);
        assert_eq!(multiplied[(1, 1, 1)], 16.0);
        comp_pts_ix3(
            &array![[[2.0, 4.0], [6.0, 8.0],], [[10.0, 12.0], [14.0, 16.0]]].into(),
            &multiplied,
            MARGIN,
            "&a * 2.0",
        );
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

        let other = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
        ]);

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[(0, 0, 0)], 6.0);
        assert_eq!(matrix[(1, 1, 1)], 20.0);
        comp_pts_ix3(
            &array![[[6.0, 8.0], [10.0, 12.0],], [[14.0, 16.0], [18.0, 20.0]]].into(),
            &matrix,
            MARGIN,
            "&a += &other",
        );

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[(0, 0, 0)], 12.0);
        assert_eq!(matrix[(1, 1, 1)], 40.0);
        comp_pts_ix3(
            &array![[[12.0, 16.0], [20.0, 24.0],], [[28.0, 32.0], [36.0, 40.0]]].into(),
            &matrix,
            MARGIN,
            "&a *= 2.0",
        );
    }

    #[test]
    fn test_map_functions() {
        let matrix = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[(0, 0, 0)], 1.0);
        assert_eq!(squared[(0, 1, 1)], 16.0);
        assert_eq!(squared[(1, 0, 0)], 25.0);
        assert_eq!(squared[(1, 1, 1)], 64.0);
        comp_pts_ix3(
            &array![[[1.0, 4.0], [9.0, 16.0],], [[25.0, 36.0], [49.0, 64.0]]].into(),
            &squared,
            MARGIN,
            "map()",
        );

        // Original should be unchanged
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(1, 1, 1)], 8.0);
        comp_pts_ix3(
            &array![[[1.0, 2.0], [3.0, 4.0],], [[5.0, 6.0], [7.0, 8.0]]].into(),
            &matrix,
            MARGIN,
            "map(orig)",
        );

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * 2.0);
        assert_eq!(matrix_mut[(0, 0, 0)], 2.0);
        assert_eq!(matrix_mut[(0, 1, 1)], 8.0);
        assert_eq!(matrix_mut[(1, 0, 0)], 10.0);
        assert_eq!(matrix_mut[(1, 1, 1)], 16.0);
        comp_pts_ix3(
            &array![[[2.0, 4.0], [6.0, 8.0],], [[10.0, 12.0], [14.0, 16.0]]].into(),
            &matrix_mut,
            MARGIN,
            "map_inplace()",
        );
    }

    #[test]
    fn test_display() {
        let matrix = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

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
        let matrix1 = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

        let matrix2 = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]);

        let matrix3 = Points::<f64, Ix3>::from_vec_f64(vec![
            vec![vec![1.0, 2.0], vec![3.0, 5.0]],
            vec![vec![5.0, 6.0], vec![7.0, 9.0]],
        ]);

        assert_eq!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
    }

    #[test]
    fn test_error_cases() {
        // Test mismatched dimensions for addition
        let matrix1 = Points::<f64, Ix3>::zeros((2, 4, 5));
        let matrix2 = Points::<f64, Ix3>::zeros((3, 2, 1));

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
        let matrix: Points<f64, Ix3> = data.into();
        assert_eq!(matrix[(0, 0, 0)], 1.0);
        assert_eq!(matrix[(1, 1, 1)], 8.0);
        comp_pts_ix3(
            &array![[[1.0, 2.0], [3.0, 4.0],], [[5.0, 6.0], [7.0, 8.0]]].into(),
            &matrix,
            MARGIN,
            "into()",
        );

        // Test conversion to Array3
        let array: Array3<f64> = matrix.into();
        assert_eq!(array[(0, 0, 0)], 1.0);
        assert_eq!(array[(1, 1, 1)], 8.0);
    }
}
