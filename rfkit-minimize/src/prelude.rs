//! rfkit-minimize prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! use rfkit_minimize::prelude::*;
//!
//! ```

#[doc(no_inline)]
pub use crate::minimize::{
    Constraint, F1dim, GF1dim, HF1dim, LinearConstraint, Minimizer, MultiDimFn, MultiDimGradFn,
    MultiDimHessFn, MultiDimNumGradFn, ObjDerFn, ObjFn, ObjGradFn, ObjHessFn, QuadraticConstraint,
    SingleDimDerFn, SingleDimFn, WolfeParams,
};

#[doc(no_inline)]
pub use crate::minimize::{
    Bracket, BracketOptions, BracketResult, Brent, BrentResult, CmaEs, CmaEsResult, ConjGrad,
    ConjGradMethod, ConjGradResult, DBrent, DBrentMethod, DBrentResult, InteriorPoint,
    InteriorPointMethod, InteriorPointParams, InteriorPointResult, NelderMead, NelderMeadMethod,
    NelderMeadOptions, NelderMeadResult, Powell, PowellResult, QuasiNewton, QuasiNewtonMethod,
    QuasiNewtonResult, Simplex, SimplexResult,
};
