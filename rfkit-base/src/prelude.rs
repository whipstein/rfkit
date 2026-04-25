//! rfkit-base prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! use rfkit_base::prelude::*;
//!
//! ```

#[doc(no_inline)]
pub use crate::num::{
    ComplexNumber, ComplexNumberBuilder, ComplexNumberType, ComplexScalar, Norm, RealScalar,
    Scalar, ScalarConst, ToComplex, ToReal,
};

#[doc(no_inline)]
pub use crate::util::{
    ApproxCompare, ApproxEq, NumMargin, comp_array_c64, comp_array_f64, comp_line, comp_pts_ix1,
    comp_pts_ix2, comp_pts_ix3, comp_vec_c64, comp_vec_f64,
};

#[doc(no_inline)]
pub use crate::points;
#[doc(no_inline)]
pub use crate::pts::{
    Matrix, MatrixComplex, MatrixReal, Points, Points1, Points2, Points3, Pts, PtsComplex, PtsReal,
};

#[doc(no_inline)]
pub use crate::units::{
    ArrayUnitValue, FreqValue, Frequency, FrequencyBuilder, MapScalar, MapToComplex, MapToReal,
    ScalarUnitValue, Scale, Scaleable, Sweep, Unit, UnitValue, UnitValueBuilder,
};

#[doc(no_inline)]
pub use crate::math::*;

#[doc(no_inline)]
pub use crate::impedance::{Impedance, ImpedanceBuilder, ImpedanceMode, ImpedanceType};

pub use twofloat::TwoFloat;
