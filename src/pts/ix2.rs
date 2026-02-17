use crate::{
    error::InversionError,
    num::{ComplexScalar, Norm, RealScalar, Scalar},
    pts::{Matrix, MatrixComplex, MatrixReal, Points, Pts, PtsComplex, PtsReal},
};
use ndarray::{IntoDimension, SliceArg, SliceInfo, linalg::Dot, prelude::*};
use ndarray_linalg::error::LinalgError;
use num_complex::{Complex, Complex64, ComplexFloat};
use twofloat::TwoFloat;

pub type Points2<T> = Points<T, Ix2>;

/// A matrix wrapper around ndarray::Array2 with T elements
impl<T: Scalar> Points<T, Ix2> {
    fn check_vec<U>(vec: &Vec<Vec<U>>) -> (usize, usize) {
        let rows = vec.len();
        let cols = vec[0].len();

        // Check all cols have the same length
        assert!(
            vec.iter().all(|col| col.len() == cols),
            "All columns must have the same length"
        );

        (rows, cols)
    }

    /// Create a matrix from a 2D vector of f64 values
    pub fn from_vec(data: Vec<Vec<T>>) -> Self {
        let shape = (data.len(), data[0].len());
        Points(
            ndarray::Array2::from_shape_vec(shape, data.into_iter().flatten().collect()).unwrap(),
        )
        // if data.is_empty() {
        //     return Self::new(Array2::from_elem((0, 0), T::zero()));
        // }

        // let shape = Self::check_vec(&data);
        // let mut matrix = Points(ndarray::Array2::zeros(shape));
        // for (i, row) in data.iter().enumerate() {
        //     for (j, value) in row.iter().enumerate() {
        //         matrix[[i, j]] = value.clone();
        //     }
        // }

        // matrix
    }

    pub fn to_vec(&self) -> Vec<Vec<T>> {
        let mut out: Vec<Vec<T>> = vec![];
        for row in self.outer_iter() {
            out.push(row.to_vec());
        }

        out
    }
}

/// A matrix wrapper around ndarray::Array2 with T elements
impl<T: RealScalar> Points<T, Ix2> {
    /// Create a matrix from a 2D vector of f64 values
    pub fn from_vec_f64(data: Vec<Vec<f64>>) -> Self {
        if data.is_empty() {
            return Self::new(Array2::from_elem((0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Points(ndarray::Array2::zeros(shape));
        for (i, row) in data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                matrix[[i, j]] = <T as From<f64>>::from(value);
            }
        }

        matrix
    }

    /// Create a matrix from a 2D vector of f64 values
    pub fn from_vec_float<U>(data: Vec<Vec<U>>) -> Self
    where
        T: From<U>,
        U: RealScalar,
    {
        if data.is_empty() {
            return Self::new(Array2::from_elem((0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Points(ndarray::Array2::zeros(shape));
        for (i, row) in data.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                matrix[[i, j]] = <T as From<U>>::from(*value);
            }
        }

        matrix
    }
}

/// A matrix wrapper around ndarray::Array2 with T elements
impl<T> Points<T, Ix2>
where
    T: ComplexScalar + From<Complex64>,
    T::Real: RealScalar,
{
    /// Create a matrix from a 2D vector of Complex64
    pub fn from_vec_c64(data: Vec<Vec<Complex64>>) -> Self {
        if data.is_empty() {
            return Self::new(Array2::from_elem((0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, row) in data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                matrix[[i, j]] = T::new(value.re.into(), value.im.into());
            }
        }

        matrix
    }

    /// Create a matrix from a 2D vector of complex
    pub fn from_vec_complex<U>(data: Vec<Vec<U>>) -> Self
    where
        T: From<U>,
        U: Copy,
    {
        if data.is_empty() {
            return Self::new(Array2::from_elem((0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, row) in data.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                matrix[[i, j]] = <T as From<U>>::from(*value);
            }
        }

        matrix
    }

    /// Create a matrix from a 2D vector of f64 values
    pub fn from_real_vec(data: Vec<Vec<T::Real>>) -> Self
    where
        T: From<T::Real>,
        T::Real: Copy,
    {
        if data.is_empty() {
            return Self::new(Array2::from_elem((0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Self::zeros(shape);
        for (i, row) in data.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                matrix[[i, j]] = <T as From<T::Real>>::from(*value);
            }
        }

        matrix
    }

    /// Create a matrix from a 2D vector of float tuples (real, imag)
    pub fn from_vec_tuple<U>(data: Vec<Vec<(U, U)>>) -> Self
    where
        T: From<Complex<T::Real>>,
        T::Real: From<U>,
        U: RealScalar,
    {
        if data.is_empty() {
            return Self::new(Array2::from_elem((0, 0), T::zero()));
        }

        let shape = Self::check_vec(&data);
        let mut matrix = Points(Array2::zeros(shape));
        for (i, row) in data.iter().enumerate() {
            for (j, (real, imag)) in row.iter().enumerate() {
                matrix[(i, j)] =
                    Complex::<T::Real>::new(T::Real::from(*real), T::Real::from(*imag)).into();
            }
        }

        matrix
    }
}

impl<T: Scalar> Pts<T, Ix2> for Points<T, Ix2> {
    type Idx = (usize, usize);

    fn from_elem(shape: impl IntoDimension<Dim = Ix2>, elem: T) -> Self {
        Points(Array2::from_elem(shape, elem))
    }

    fn from_shape_fn<F>(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>, f: F) -> Self
    where
        F: Fn((usize, usize)) -> T,
    {
        Points(Array2::<T>::from_shape_fn(shape, f))
    }

    fn from_shape_vec(
        shape: impl IntoDimension<Dim = Dim<[usize; 2]>>,
        v: Vec<T>,
    ) -> Result<Self, &'static str> {
        match Array2::<T>::from_shape_vec(shape, v) {
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

    fn iter(&self) -> ndarray::iter::Iter<'_, T, Ix2> {
        self.0.iter()
    }

    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix2> {
        self.0.iter_mut()
    }

    fn axis_iter(&self, axis: Axis) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix1> {
        self.0.axis_iter(axis)
    }

    fn axis_iter_mut(&mut self, axis: Axis) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix1> {
        self.0.axis_iter_mut(axis)
    }

    fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, T, ndarray::Ix2> {
        self.0.indexed_iter()
    }

    fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, T, ndarray::Ix2> {
        self.0.indexed_iter_mut()
    }

    fn outer_iter(&self) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix1> {
        self.0.outer_iter()
    }

    fn outer_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix1> {
        self.0.outer_iter_mut()
    }

    fn slice<I>(&self, info: I) -> ndarray::ArrayView<'_, T, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix2>,
    {
        self.0.slice(info)
    }

    fn slice_mut<I>(&mut self, info: I) -> ndarray::ArrayViewMut<'_, T, I::OutDim>
    where
        I: ndarray::SliceArg<ndarray::Ix2>,
    {
        self.0.slice_mut(info)
    }

    fn assign<E: ndarray::Dimension>(&mut self, rhs: &Array<T, E>) {
        self.0.assign(rhs);
    }

    fn push(
        &mut self,
        axis: Axis,
        array: ndarray::ArrayView<'_, T, ndarray::Ix1>,
    ) -> Result<(), ndarray::ShapeError>
    where
        T: Clone,
    {
        self.0.push(axis, array)
    }

    /// Get and set outer axis point
    fn pt<I>(&self, index: usize) -> Points<T, I::OutDim>
    where
        I: SliceArg<Ix2, OutDim = Dim<[usize; 1]>>,
    {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} points",
                index,
                self.nrows()
            );
        }

        Points(self.slice(s![index, ..]).to_owned())
    }

    fn pt_mut<I>(&mut self, index: usize) -> Points<T, I::OutDim>
    where
        I: SliceArg<Ix2, OutDim = Dim<[usize; 1]>>,
    {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} points",
                index,
                self.nrows()
            );
        }

        Points(self.slice_mut(s![index, ..]).to_owned())
    }

    fn set_pt(&mut self, index: usize, pt: Points<T, Ix1>) {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} points",
                index,
                self.nrows()
            );
        }
        if pt.len() != self.ncols() {
            panic!(
                "Col dimensions incompatible: expected {}, got {}",
                self.ncols(),
                pt.len()
            );
        }

        for k in 0..self.ncols() {
            self[[index, k]] = pt[k].clone();
        }
    }

    fn insert_axis(self, axis: Axis) -> Points<T, Ix3> {
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
        1
    }

    /// Get the shape as (rows, cols)
    fn dim(&self) -> (usize, usize) {
        self.0.dim()
    }

    /// Get the dimension
    fn raw_dim(&self) -> Ix2 {
        self.0.raw_dim()
    }

    /// Get the shape as (rows, cols)
    fn shape(&self) -> (usize, usize) {
        self.dim()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a view of the matrix
    fn view(&self) -> ArrayView2<'_, T> {
        self.0.view()
    }

    /// Get a mutable view of the matrix
    fn view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.0.view_mut()
    }

    /// Access the inner ndarray (for advanced operations)
    fn inner(&self) -> &Array2<T> {
        &self.0
    }

    /// Convert to inner ndarray (consuming self)
    fn into_inner(self) -> Array2<T> {
        self.0
    }

    fn into_raw_vec_and_offset(self) -> (Vec<T>, Option<usize>) {
        self.0.into_raw_vec_and_offset()
    }

    /// Create a matrix filled with the given value
    fn fill(shape: impl IntoDimension<Dim = Dim<[usize; 2]>>, value: T) -> Self {
        Points(Array2::from_elem(shape, value))
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

    // /// Check if a matrix is approximately equal to another matrix within a tolerance
    // ///
    // /// # Arguments
    // /// * `a` - First matrix
    // /// * `b` - Second matrix
    // /// * `tol` - Tolerance for comparison
    // ///
    // /// # Returns
    // /// * `true` if matrices are approximately equal, `false` otherwise
    // fn approx_eq(a: &ArrayView2<T>, b: &ArrayView2<T>, tol: f64) -> bool
    // where
    //     <T as ComplexFloat>::Real: PartialOrd<f64>,
    //     for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    // {
    //     if a.dim() != b.dim() {
    //         return false;
    //     }

    //     let (rows, cols) = a.dim();

    //     for i in 0..rows {
    //         for j in 0..cols {
    //             let diff = &a[[i, j]] - &b[[i, j]];
    //             if diff.norm() > tol {
    //                 return false;
    //             }
    //         }
    //     }

    //     true
    // }
}

impl<T: ComplexScalar> PtsComplex<T, Ix2> for Points<T, Ix2>
where
    T::Real: RealScalar,
{
    // type Size = (usize, usize);
    type Tuple<'a>
        = (T, T)
    where
        T: 'a;
    // type Vec = Vec<Vec<(T::Real, T::Real)>>;

    // fn check_dims(vals: &Vec<Vec<(T::Real, T::Real)>>) -> Result<(usize, usize), &'static str> {
    //     let rows = vals.len();
    //     let cols = vals.first().map_or(0, |r| r.len());

    //     if vals.iter().any(|r| r.len() != cols) {
    //         return Err("Inconsistent row lengths");
    //     }

    //     Ok((rows, cols))
    // }

    // fn from_db(vals: &Vec<Vec<(T::Real, T::Real)>>) -> Result<Self, String> {
    //     let dim = Self::check_dims(vals)?;

    //     Ok(Points2::from_shape_fn(dim, |(i, j)| {
    //         let r = ComplexFloat::powf(T::Real::C10, vals[i][j].0 / T::Real::C20);
    //         let theta = vals[i][j].1.to_radians();
    //         T::new(
    //             r * <T::Real as Float>::cos(theta),
    //             r * <T::Real as Float>::sin(theta),
    //         )
    //     }))
    // }

    // fn from_magang(vals: &Vec<Vec<(T::Real, T::Real)>>) -> Result<Self, String> {
    //     let dim = Self::check_dims(vals)?;

    //     Ok(Points2::from_shape_fn(dim, |(i, j)| {
    //         let r = vals[i][j].0;
    //         let theta = vals[i][j].1.to_radians();
    //         T::new(
    //             r * <T::Real as Float>::cos(theta),
    //             r * <T::Real as Float>::sin(theta),
    //         )
    //     }))
    // }

    // fn from_reim(vals: &Vec<Vec<(T::Real, T::Real)>>) -> Result<Self, String> {
    //     let dim = Self::check_dims(vals)?;

    //     Ok(Points2::from_shape_fn(dim, |(i, j)| {
    //         T::new(vals[i][j].0, vals[i][j].1)
    //     }))
    // }

    // fn db(&self) -> Points<T::Real, Ix2> {
    //     Points::from_shape_fn(self.dim(), |idx| {
    //         T::Real::C20 * Float::log10(self[idx].abs())
    //     })
    // }

    // fn deg(&self) -> Points<T::Real, Ix2> {
    //     Points::from_shape_fn(self.dim(), |idx| self[idx].arg().to_degrees())
    // }

    // fn im(&self) -> Points<T::Real, Ix2> {
    //     Points::from_shape_fn(self.dim(), |idx| self[idx].im())
    // }

    // fn mag(&self) -> Points<T::Real, Ix2> {
    //     Points::from_shape_fn(self.dim(), |idx| self[idx].abs())
    // }

    // fn rad(&self) -> Points<<T>::Real, Ix2> {
    //     Points::from_shape_fn(self.dim(), |idx| self[idx].arg())
    // }

    // fn re(&self) -> Points<T::Real, Ix2> {
    //     Points::from_shape_fn(self.dim(), |idx| self[idx].re())
    // }
}

impl<T: RealScalar> PtsReal<T, Ix2> for Points<T, Ix2> {
    /// Create a matrix from a flat vector with specified dimensions
    fn from_flat_f64(
        data: Vec<f64>,
        shape: impl IntoDimension<Dim = Dim<[usize; 2]>>,
    ) -> Result<Self, &'static str> {
        let (rows, cols) = shape.into_dimension().into_pattern();
        if data.len() != rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        let mut matrix = Points(Array2::zeros((rows, cols)));
        for (idx, &value) in data.iter().enumerate() {
            let i = idx / cols;
            let j = idx % cols;
            matrix[[i, j]] = <T as From<f64>>::from(value);
        }

        Ok(matrix)
    }
}

impl<T: Scalar> Matrix<T, Ix2> for Points<T, Ix2> {
    /// Get the number of rows
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    /// Get the number of columns
    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    /// Get a row as a new matrix (1 x ncols)
    fn row(&self, index: usize) -> Points<T, Ix1> {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} rows",
                index,
                self.nrows()
            );
        }

        // let mut result = Points::<T, Ix1>::zeros(self.ncols());
        // for j in 0..self.ncols() {
        //     result[[0, j]] = self[[index, j]].clone();
        // }
        // result
        Points(self.0.row(index).to_owned())
    }

    /// Get a column as a new matrix (nrows x 1)
    fn col(&self, index: usize) -> Points<T, Ix1> {
        if index >= self.ncols() {
            panic!(
                "Column index {} out of bounds for matrix with {} columns",
                index,
                self.ncols()
            );
        }

        // let mut result = Self::zeros((self.nrows(), 1));
        // for i in 0..self.nrows() {
        //     result[[i, 0]] = self[[i, index]].clone();
        // }
        // result
        // let mut result = Points::<T, Ix1>::zeros(self.nrows());
        // for (i, val) in self.0.column(index).into_iter().enumerate() {
        //     result[i] = val.clone();
        // }
        // result
        Points(self.0.column(index).to_owned())
    }

    /// Set a row from another matrix
    fn set_row(&mut self, index: usize, row: &Points<T, Ix1>) {
        if index >= self.nrows() {
            panic!(
                "Row index {} out of bounds for matrix with {} rows",
                index,
                self.nrows()
            );
        }
        if row.len() != self.ncols() {
            panic!(
                "Row dimensions incompatible: expected {}, got {}",
                self.ncols(),
                row.len()
            );
        }

        for j in 0..self.ncols() {
            self[[index, j]] = row[j].clone();
        }
    }

    /// Set a column from another matrix
    fn set_col(&mut self, index: usize, col: &Points<T, Ix1>) {
        if index >= self.ncols() {
            panic!(
                "Column index {} out of bounds for matrix with {} columns",
                index,
                self.ncols()
            );
        }
        if col.len() != self.nrows() {
            panic!(
                "Column dimensions incompatible: expected {}, got {}",
                self.nrows(),
                col.len()
            );
        }

        for i in 0..self.nrows() {
            self[[i, index]] = col[i].clone();
        }
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

    /// Create an identity matrix of given size
    fn eye(size: impl IntoDimension<Dim = Dim<[usize; 1]>>) -> Self {
        Points(Array2::eye(size.into_dimension().into_pattern()))
    }

    /// Create an identity matrix of given size
    fn eye_value(size: impl IntoDimension<Dim = Dim<[usize; 1]>>, value: T) -> Self {
        let dim = size.into_dimension().into_pattern();
        Points(Array2::from_shape_fn((dim, dim), |(j, k)| {
            if j == k { value } else { T::zero() }
        }))
    }

    /// Calculate the trace (sum of diagonal elements)
    fn trace(&self) -> Result<Array0<T>, LinalgError>
    where
        T: std::iter::Sum,
    {
        let n = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        }?;
        let sum = (0..n as usize).map(|i| self[(i, i)]).sum();
        Ok(Array0::from_elem((), sum))
    }

    /// Calculate the determinant (only for small matrices)
    fn det(&self) -> Result<Array0<T>, LinalgError> {
        _ = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        };

        match self.nrows() {
            0 => Ok(Array0::from_elem((), T::one())),
            1 => Ok(Array0::from_elem((), self[[0, 0]].clone())),
            2 => Ok(Array0::from_elem(
                (),
                self[[0, 0]] * self[[1, 1]] - self[[0, 1]] * self[[1, 0]],
            )),
            3 => {
                let a00 = self[[0, 0]];
                let a01 = self[[0, 1]];
                let a02 = self[[0, 2]];
                let a10 = self[[1, 0]];
                let a11 = self[[1, 1]];
                let a12 = self[[1, 2]];
                let a20 = self[[2, 0]];
                let a21 = self[[2, 1]];
                let a22 = self[[2, 2]];

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

    /// Compute the inverse of a square matrix using LU decomposition with partial pivoting
    ///
    /// This function computes A^(-1) such that A * A^(-1) = I, where I is the identity matrix.
    ///
    /// # Arguments
    /// * `matrix` - A square matrix to invert
    ///
    /// # Returns
    /// * `Ok(Array2<T>)` - The inverted matrix
    /// * `Err(InversionError)` - If the matrix is not square, singular, or other errors
    ///
    /// # Examples
    /// ```rust
    /// use num::complex::Complex64;
    /// use ndarray::prelude::*;
    /// use rfkit::prelude::*;
    ///
    /// // Create a 2x2 matrix
    /// let matrix = Points::from_shape_vec((2, 2), vec![
    ///     Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
    ///     Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
    /// ]).unwrap();
    ///
    /// let inv_matrix = matrix.inv();
    /// ```
    fn inv(&self) -> Points<T, Ix2>
    where
        Self: Sized,
        T: Norm,
        <T as ComplexFloat>::Real: RealScalar,
    {
        self.try_inv().unwrap()
    }

    fn try_inv(&self) -> Result<Points<T, Ix2>, InversionError>
    where
        Self: Sized,
        T: Norm,
        <T as ComplexFloat>::Real: RealScalar,
    {
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
            augmented[[i, i + n]] = T::one();
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
            if max_abs < 1e-12.into() {
                return Err(InversionError::Singular(format!(
                    "Matrix is singular or nearly singular at pivot {}",
                    i
                )));
            }

            // Swap rows if necessary
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Scale pivot row to make diagonal element 1
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column i in all other rows
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        let temp = augmented[[i, j]] * factor;
                        augmented[[k, j]] -= temp;
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
}

impl<T: RealScalar> MatrixReal<T, Ix2> for Points<T, Ix2> {
    /// Solve the linear system Ax = b using LU decomposition
    ///
    /// # Arguments
    /// * `a` - Coefficient matrix A (n×n)
    /// * `b` - Right-hand side vector b (n×1)
    ///
    /// # Returns
    /// * `Ok(Array2<MyComplex>)` - Solution vector x
    /// * `Err(InversionError)` - If the system cannot be solved
    fn solve_linear_system(&self, b: &ArrayView<T, Ix2>) -> Result<Array<T, Ix2>, InversionError>
    where
        Self: Sized,
        T: Norm,
        <T as ComplexFloat>::Real: RealScalar,
    {
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
            if max_abs < 1e-12.into() {
                return Err(InversionError::Singular(format!(
                    "Matrix is singular at pivot {}",
                    i
                )));
            }

            // Swap rows if necessary
            if pivot_row != i {
                for j in 0..(n + 1) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Eliminate below pivot
            for k in (i + 1)..n {
                let factor = augmented[[k, i]] / augmented[[i, i]];
                for j in i..(n + 1) {
                    let temp = augmented[[i, j]] * factor;
                    augmented[[k, j]] -= temp;
                }
            }
        }

        // Back substitution
        let mut x = Array2::zeros((n, 1));
        for i in (0..n).rev() {
            let mut sum = augmented[[i, n]];
            for j in (i + 1)..n {
                sum -= augmented[[i, j]] * x[[j, 0]];
            }
            x[[i, 0]] = sum / augmented[[i, i]];
        }

        Ok(x)
    }
}

impl<T: ComplexScalar> MatrixComplex<T, Ix2> for Points<T, Ix2>
where
    T::Real: RealScalar,
{
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

    /// Element-wise conjugate
    fn conj(&self) -> Self {
        let mut result = self.clone();
        for element in result.0.iter_mut() {
            *element = element.conj();
        }
        result
    }

    // /// Calculate the Frobenius norm
    // fn frobenius_norm(&self) -> Array0<T::Real> {
    //     let mut sum = T::Real::zero();
    //     for element in self.0.iter() {
    //         sum += element.norm_sqr();
    //     }
    //     Array0::from_elem((), sum.sqrt())
    // }
}

// Dot product implementations
impl<T: Scalar> Dot<Points<T, Ix1>> for Points<T, Ix2> {
    type Output = Points<T, Ix1>;

    fn dot(&self, rhs: &Points<T, Ix1>) -> Self::Output {
        if self.ncols() != rhs.len() {
            panic!(
                "Point dimensions incompatible for multiplication: {}x{} * {}",
                self.nrows(),
                self.ncols(),
                rhs.len()
            );
        }

        let mut result = Points::zeros(self.nrows());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                result[i] += self[[i, j]] * rhs[j];
            }
        }

        result
    }
}

impl<T: Scalar> Dot<Points<T, Ix2>> for Points<T, Ix2> {
    type Output = Points<T, Ix2>;

    fn dot(&self, rhs: &Points<T, Ix2>) -> Self::Output {
        if self.ncols() != rhs.nrows() {
            panic!(
                "Point dimensions incompatible for multiplication: {}x{} * {}x{}",
                self.nrows(),
                self.ncols(),
                rhs.nrows(),
                rhs.ncols()
            );
        }

        let mut result = Self::zeros((self.nrows(), rhs.ncols()));

        for i in 0..self.nrows() {
            for j in 0..rhs.ncols() {
                let mut sum = T::zero();
                for k in 0..self.ncols() {
                    sum += self[[i, k]] * rhs[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }
}

impl<T: Scalar> Dot<Points<T, Ix3>> for Points<T, Ix2> {
    type Output = Points<T, Ix3>;

    fn dot(&self, rhs: &Points<T, Ix3>) -> Self::Output {
        let mut result = Self::Output::zeros(rhs.dim());

        for i in 0..self.npts() {
            let b: Points<T, Ix2> = rhs
                .pt::<SliceInfo<[ndarray::SliceInfoElem; 3], Ix3, Ix2>>(i)
                .into();
            result.set_pt(i, self.dot(&b));
        }

        result
    }
}

// Traits
impl<T: Scalar> Default for Points<T, Ix2> {
    fn default() -> Self {
        Points::zeros((0, 0))
    }
}

// Conversion traits
impl<T: Scalar> From<Array2<T>> for Points<T, Ix2> {
    fn from(array: Array2<T>) -> Self {
        Points(array)
    }
}

impl<T: Scalar> From<ArrayView2<'_, T>> for Points<T, Ix2> {
    fn from(array: ArrayView2<T>) -> Self {
        Points(array.to_owned())
    }
}

impl<T> From<Points<T, Ix2>> for Array2<T> {
    fn from(matrix: Points<T, Ix2>) -> Self {
        matrix.0
    }
}

impl<T: Scalar> From<Vec<Vec<T>>> for Points<T, Ix2> {
    fn from(data: Vec<Vec<T>>) -> Self {
        Points::<T, Ix2>::from_vec(data)
    }
}

impl From<Points<f64, Ix2>> for Points<TwoFloat, Ix2> {
    fn from(point: Points<f64, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| point[[i, j]].into())
    }
}

impl From<&Points<f64, Ix2>> for Points<TwoFloat, Ix2> {
    fn from(point: &Points<f64, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| point[[i, j]].into())
    }
}

impl From<Points<TwoFloat, Ix2>> for Points<f64, Ix2> {
    fn from(point: Points<TwoFloat, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| point[[i, j]].hi())
    }
}

impl From<&Points<TwoFloat, Ix2>> for Points<f64, Ix2> {
    fn from(point: &Points<TwoFloat, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| point[[i, j]].hi())
    }
}

impl From<Points<Complex<f64>, Ix2>> for Points<Complex<TwoFloat>, Ix2> {
    fn from(point: Points<Complex<f64>, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| {
            Complex::<TwoFloat>::new(point[[i, j]].re.into(), point[[i, j]].im.into())
        })
    }
}

impl From<&Points<Complex<f64>, Ix2>> for Points<Complex<TwoFloat>, Ix2> {
    fn from(point: &Points<Complex<f64>, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| {
            Complex::<TwoFloat>::new(point[[i, j]].re.into(), point[[i, j]].im.into())
        })
    }
}

impl From<Points<Complex<TwoFloat>, Ix2>> for Points<Complex<f64>, Ix2> {
    fn from(point: Points<Complex<TwoFloat>, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| {
            Complex::<f64>::new(point[[i, j]].re.hi(), point[[i, j]].im.hi())
        })
    }
}

impl From<&Points<Complex<TwoFloat>, Ix2>> for Points<Complex<f64>, Ix2> {
    fn from(point: &Points<Complex<TwoFloat>, Ix2>) -> Self {
        Points::from_shape_fn(point.dim(), |(i, j)| {
            Complex::<f64>::new(point[[i, j]].re.hi(), point[[i, j]].im.hi())
        })
    }
}

#[cfg(test)]
mod points_ix2_f64_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin, comp_pts_ix1, comp_pts_ix2};
    use num_complex::c64;
    use num_traits::{One, Zero};
    use twofloat::TwoFloat;

    const MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-4,
        relative: 1e-4,
        ulps: 10,
    };

    #[test]
    fn test_creation() {
        let zeros = Points::<f64, Ix2>::zeros((2, 3));
        assert_eq!(zeros.shape(), (2, 3));
        assert!(zeros[[0, 0]].is_zero());
        assert!(zeros[[1, 2]].is_zero());
        comp_pts_ix2(
            &points![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            &zeros,
            MARGIN,
            "zeros()",
        );

        let ones = Points::<f64, Ix2>::ones((3, 2));
        assert_eq!(ones.shape(), (3, 2));
        assert!(ones[[0, 0]].is_one());
        assert!(ones[[2, 1]].is_one());
        comp_pts_ix2(
            &points![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            &ones,
            MARGIN,
            "ones()",
        );

        let eye = Points::<f64, Ix2>::eye(3);
        assert_eq!(eye.shape(), (3, 3));
        assert!(eye[[0, 0]].is_one());
        assert!(eye[[1, 1]].is_one());
        assert!(eye[[2, 2]].is_one());
        assert!(eye[[0, 1]].is_zero());
        assert!(eye[[1, 0]].is_zero());
        comp_pts_ix2(
            &points![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &eye,
            MARGIN,
            "eye()",
        );
    }

    #[test]
    fn test_from_vec() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Points::<f64, Ix2>::from_vec(data);

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
        assert_eq!(matrix[[1, 0]], 4.0);
        assert_eq!(matrix[[1, 1]], 5.0);
        assert_eq!(matrix[[1, 2]], 6.0);
        comp_pts_ix2(
            &points![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &matrix,
            MARGIN,
            "from_vec()",
        );
    }

    #[test]
    fn test_from_vec_f64() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Points::<f64, Ix2>::from_vec_f64(data);

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
        assert_eq!(matrix[[1, 0]], 4.0);
        assert_eq!(matrix[[1, 1]], 5.0);
        assert_eq!(matrix[[1, 2]], 6.0);
        comp_pts_ix2(
            &points![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &matrix,
            MARGIN,
            "from_vec_f64()",
        );
    }

    #[test]
    fn test_from_vec_float() {
        let data: Vec<Vec<TwoFloat>> = vec![
            vec![1.0.into(), 2.0.into(), 3.0.into()],
            vec![4.0.into(), 5.0.into(), 6.0.into()],
        ];
        let matrix = Points::<f64, Ix2>::from_vec_float(data);

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
        assert_eq!(matrix[[1, 0]], 4.0);
        assert_eq!(matrix[[1, 1]], 5.0);
        assert_eq!(matrix[[1, 2]], 6.0);
        comp_pts_ix2(
            &points![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &matrix,
            MARGIN,
            "from_vec_float()",
        );
    }

    #[test]
    fn test_arithmetic() {
        let a = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let b = Points::<f64, Ix2>::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

        // Addition
        let sum = &a + &b;
        assert_eq!(sum[[0, 0]], 6.0);
        assert_eq!(sum[[0, 1]], 8.0);
        assert_eq!(sum[[1, 0]], 10.0);
        assert_eq!(sum[[1, 1]], 12.0);
        comp_pts_ix2(&points![[6.0, 8.0], [10.0, 12.0]], &sum, MARGIN, "&a + &b");

        // Subtraction
        let diff = &b - &a;
        assert_eq!(diff[[0, 0]], 4.0);
        assert_eq!(diff[[0, 1]], 4.0);
        assert_eq!(diff[[1, 0]], 4.0);
        assert_eq!(diff[[1, 1]], 4.0);
        comp_pts_ix2(&points![[4.0, 4.0], [4.0, 4.0]], &diff, MARGIN, "&a - &b");

        // Element-wise multiplication
        let prod = &a * &b;
        assert_eq!(prod[[0, 0]], 5.0);
        assert_eq!(prod[[0, 1]], 12.0);
        assert_eq!(prod[[1, 0]], 21.0);
        assert_eq!(prod[[1, 1]], 32.0);
        comp_pts_ix2(
            &points![[5.0, 12.0], [21.0, 32.0]],
            &prod,
            MARGIN,
            "&a * &b",
        );
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let b = Points::<f64, Ix2>::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

        let result = a.dot(&b);

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result[[0, 0]], 19.0);
        assert_eq!(result[[0, 1]], 22.0);
        assert_eq!(result[[1, 0]], 43.0);
        assert_eq!(result[[1, 1]], 50.0);
        comp_pts_ix2(
            &points![[19.0, 22.0], [43.0, 50.0]],
            &result,
            MARGIN,
            "a.dot(&b)",
        );
    }

    #[test]
    fn test_transpose() {
        let matrix = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let transposed = matrix.transpose();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed[[0, 0]], 1.0);
        assert_eq!(transposed[[0, 1]], 4.0);
        assert_eq!(transposed[[1, 0]], 2.0);
        assert_eq!(transposed[[1, 1]], 5.0);
        assert_eq!(transposed[[2, 0]], 3.0);
        assert_eq!(transposed[[2, 1]], 6.0);
        comp_pts_ix2(
            &points![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            &transposed,
            MARGIN,
            "transpose()",
        );
    }

    #[test]
    fn test_conjugate_transpose() {
        let matrix = Points::<Complex<f64>, Ix2>::from_vec(vec![
            vec![c64(1.0, 0.0), c64(3.0, 0.0)],
            vec![c64(5.0, 0.0), c64(7.0, 0.0)],
        ]);

        let conj_t = matrix.conj_transpose();
        assert_eq!(conj_t.shape(), (2, 2));

        // Original: [(1+2i, 3+4i), (5+6i, 7+8i)]
        // Conj transpose: [(1-2i, 5-6i), (3-4i, 7-8i)]
        assert_eq!(conj_t[[0, 0]], c64(1.0, 0.0));
        assert_eq!(conj_t[[0, 1]], c64(5.0, 0.0));
        assert_eq!(conj_t[[1, 0]], c64(3.0, 0.0));
        assert_eq!(conj_t[[1, 1]], c64(7.0, 0.0));
        comp_pts_ix2(
            &points![[1.0.into(), 5.0.into()], [3.0.into(), 7.0.into()]],
            &conj_t,
            MARGIN,
            "conj_transpose()",
        );
    }

    #[test]
    fn test_trace() {
        let matrix = Points::<f64, Ix2>::from_vec(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let trace = matrix.trace().unwrap().into_scalar();
        assert_eq!(trace, 15.0); // 1 + 5 + 9 = 15
        trace.assert_approx_eq(&15.0, MARGIN, "trace()", "");
    }

    #[test]
    fn test_determinant() {
        // Test 2x2 determinant
        let matrix2x2 = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let det2 = matrix2x2.det().unwrap().into_scalar();
        assert_eq!(det2, -2.0); // 1*4 - 2*3 = -2
        det2.assert_approx_eq(&-2.0, MARGIN, "det()", "");

        // Test 3x3 determinant
        let matrix3x3 = Points::<f64, Ix2>::from_vec(vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ]);

        let det3 = matrix3x3.det().unwrap().into_scalar();
        assert_eq!(det3, 1.0); // Should be 1
        det3.assert_approx_eq(&1.0, MARGIN, "det()", "");
    }

    #[test]
    fn test_scalar_operations() {
        let matrix = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // Scalar addition
        let added = &matrix + 5.0;
        assert_eq!(added[[0, 0]], 6.0);
        assert_eq!(added[[1, 1]], 9.0);
        comp_pts_ix2(
            &points![[6.0, 7.0], [8.0, 9.0]],
            &added,
            MARGIN,
            "&matrix + 5.0",
        );

        // Scalar multiplication
        let multiplied = &matrix * 2.0;
        assert_eq!(multiplied[[0, 0]], 2.0);
        assert_eq!(multiplied[[0, 1]], 4.0);
        assert_eq!(multiplied[[1, 0]], 6.0);
        assert_eq!(multiplied[[1, 1]], 8.0);
        comp_pts_ix2(
            &points![[2.0, 4.0], [6.0, 8.0]],
            &multiplied,
            MARGIN,
            "&a * 2.0",
        );
    }

    #[test]
    fn test_assignment_operators() {
        let mut matrix = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let other = Points::<f64, Ix2>::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

        // Test AddAssign
        matrix += &other;
        assert_eq!(matrix[[0, 0]], 6.0);
        assert_eq!(matrix[[1, 1]], 12.0);
        comp_pts_ix2(
            &points![[6.0, 8.0], [10.0, 12.0]],
            &matrix,
            MARGIN,
            "&a += &other",
        );

        // Test MulAssign with scalar
        matrix *= 2.0;
        assert_eq!(matrix[[0, 0]], 12.0);
        assert_eq!(matrix[[1, 1]], 24.0);
        comp_pts_ix2(
            &points![[12.0, 16.0], [20.0, 24.0]],
            &matrix,
            MARGIN,
            "&a *= 2.0",
        );
    }

    #[test]
    fn test_row_col_operations() {
        let matrix = Points::<f64, Ix2>::from_vec(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        // Test getting a row
        let row1 = matrix.row(1);
        assert_eq!(row1.shape(), 3);
        assert_eq!(row1[0], 4.0);
        assert_eq!(row1[1], 5.0);
        assert_eq!(row1[2], 6.0);
        comp_pts_ix1(&points![4.0, 5.0, 6.0], &row1, MARGIN, "row()");

        // Test getting a column
        let col2 = matrix.col(2);
        assert_eq!(col2.shape(), 3);
        assert_eq!(col2[0], 3.0);
        assert_eq!(col2[1], 6.0);
        assert_eq!(col2[2], 9.0);
        comp_pts_ix1(&points![3.0, 6.0, 9.0], &col2, MARGIN, "col()");

        // Test setting a row
        let mut matrix_mut = matrix.clone();
        let new_row = Points::<f64, Ix1>::from_vec(vec![10.0, 11.0, 12.0]);
        matrix_mut.set_row(0, &new_row);
        assert_eq!(matrix_mut[[0, 0]], 10.0);
        assert_eq!(matrix_mut[[0, 1]], 11.0);
        assert_eq!(matrix_mut[[0, 2]], 12.0);
        comp_pts_ix2(
            &points![[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &matrix_mut,
            MARGIN,
            "set_row()",
        );
    }

    #[test]
    fn test_map_functions() {
        let matrix = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // Test map (immutable)
        let squared = matrix.map(|x| x * x);
        assert_eq!(squared[[0, 0]], 1.0);
        assert_eq!(squared[[0, 1]], 4.0);
        assert_eq!(squared[[1, 0]], 9.0);
        assert_eq!(squared[[1, 1]], 16.0);
        comp_pts_ix2(&points![[1.0, 4.0], [9.0, 16.0]], &squared, MARGIN, "map()");

        // Original should be unchanged
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 4.0);
        comp_pts_ix2(
            &points![[1.0, 2.0], [3.0, 4.0]],
            &matrix,
            MARGIN,
            "map(orig)",
        );

        // Test map_inplace (mutable)
        let mut matrix_mut = matrix.clone();
        matrix_mut.map_inplace(|x| x * 2.0);
        assert_eq!(matrix_mut[[0, 0]], 2.0);
        assert_eq!(matrix_mut[[0, 1]], 4.0);
        assert_eq!(matrix_mut[[1, 0]], 6.0);
        assert_eq!(matrix_mut[[1, 1]], 8.0);
        comp_pts_ix2(
            &points![[2.0, 4.0], [6.0, 8.0]],
            &matrix_mut,
            MARGIN,
            "map_inplace()",
        );
    }

    #[test]
    fn test_display() {
        let matrix = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let display_str = format!("{}", matrix);
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
    }

    #[test]
    fn test_equality() {
        let matrix1 = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let matrix2 = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let matrix3 = Points::<f64, Ix2>::from_vec(vec![vec![1.0, 2.0], vec![3.0, 5.0]]);

        assert_eq!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
    }

    #[test]
    fn test_error_cases() {
        // Test mismatched dimensions for addition
        let matrix1 = Points::<f64, Ix2>::zeros((2, 4));
        let matrix2 = Points::<f64, Ix2>::zeros((3, 2));

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
        let matrix: Points<f64, Ix2> = data.into();
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 4.0);

        // Test conversion to Array2
        let array: Array2<f64> = matrix.into();
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 1]], 4.0);
    }

    #[test]
    fn test_2x2_matrix_inversion() {
        // Test inverting a simple 2x2 matrix
        // A = [1 2]  =>  A^(-1) = [-2  1]
        //     [3 4]               [1.5 -0.5]

        let matrix = Points::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let inverse = matrix.inv();

        // Check specific values
        // approx_eq!(f64, inverse[[0, 0]], -2.0, NumMargin::default());
        // approx_eq!(f64, inverse[[0, 1]], 1.0, NumMargin::default());
        // approx_eq!(f64, inverse[[1, 0]], 1.5, NumMargin::default());
        // approx_eq!(f64, inverse[[1, 1]], -0.5, NumMargin::default());
        comp_pts_ix2(
            &points![[-2.0, 1.0], [1.5, -0.5]],
            &inverse,
            MARGIN,
            "inv()",
        );

        // Verify A * A^(-1) = I
        let product = &matrix.dot(&inverse);
        let identity = Points::<f64, Ix2>::eye(2);

        assert!(product.approx_eq(
            &identity,
            NumMargin {
                epsilon: 1e-10,
                relative: 1e-10,
                ulps: 4
            }
        ));
        comp_pts_ix2(&points![[1.0, 0.0], [0.0, 1.0]], &product, MARGIN, "inv()");
    }

    #[test]
    fn test_complex_matrix_inversion() {
        // Test with complex numbers
        let matrix = Points::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 1.0]).unwrap();

        let inverse = matrix.inv();

        // Verify A * A^(-1) = I
        let product = matrix.dot(&inverse);
        let identity = Points::<f64, Ix2>::eye(2);

        assert!(product.approx_eq(
            &identity,
            NumMargin {
                epsilon: 1e-10,
                relative: 1e-10,
                ulps: 4
            }
        ));
        comp_pts_ix2(&points![[1.0, 0.0], [0.0, 1.0]], &product, MARGIN, "inv()");
    }

    #[test]
    fn test_3x3_matrix_inversion() {
        // Test with a 3x3 matrix
        let matrix = Points::from_shape_vec(
            (3, 3),
            vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        )
        .unwrap();

        let inverse = matrix.inv();

        // Verify A * A^(-1) = I
        let product = matrix.dot(&inverse);
        let identity = Points::<f64, Ix2>::eye(3);

        assert!(product.approx_eq(
            &identity,
            NumMargin {
                epsilon: 1e-10,
                relative: 1e-10,
                ulps: 4
            }
        ));
        comp_pts_ix2(
            &points![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &product,
            MARGIN,
            "inv()",
        );
    }

    #[test]
    fn test_singular_matrix() {
        // Test with a singular matrix (determinant = 0)
        let matrix = Points::from_shape_vec(
            (2, 2),
            vec![
                1.0, 2.0, 2.0, 4.0, // Second row is 2x first row
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
        let matrix = Points::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

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
        let identity = Points::<f64, Ix2>::eye(4);
        let inverse = identity.inv();

        assert!(inverse.approx_eq(
            &identity,
            NumMargin {
                epsilon: 1e-10,
                relative: 1e-10,
                ulps: 4
            }
        ));
        comp_pts_ix2(
            &points![
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            &inverse,
            MARGIN,
            "inv()",
        );
    }
}
