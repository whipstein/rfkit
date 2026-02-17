//! rfkit-base-ndarray prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! use rfkit::prelude::*;
//!
//! ```

#[doc(no_inline)]
pub use crate::consts::{
    MathConst, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, pi_c, tau_c, e_c, frac_pi_2_c,
    frac_pi_3_c, frac_pi_4_c, frac_pi_6_c, frac_pi_8_c, frac_1_pi_c, frac_2_pi_c,
    frac_2_sqrt_pi_c, frac_1_sqrt_2_c, sqrt_2_c, ln_2_c, ln_10_c, log2_e_c, log10_e_c,
    log2_10_c, log10_2_c,
};

#[doc(no_inline)]
pub use crate::file::read_touchstone;

// #[doc(no_inline)]
// pub use crate::num::{mycomplex::*, myfloat::*};
// #[doc(no_inline)]
// pub use num::ToPrimitive;

#[doc(no_inline)]
pub use crate::frequency::{
    Freq, FreqArray, FreqScalar, Frequency, FrequencyBuilder, FrequencyScalar,
    FrequencyScalarBuilder, new_frequency, new_frequency_scaled,
};

#[doc(no_inline)]
pub use crate::circuit::Circuit;
#[doc(no_inline)]
pub use crate::element::{
    Elem, ElemType, Element, ElementBuilder,
    capacitor::{Capacitor, CapacitorBuilder},
    inductor::{Inductor, InductorBuilder},
    port::{Port, PortBuilder},
    q::{Q, QBuilder, QMode},
    resistor::{Resistor, ResistorBuilder},
};

#[doc(no_inline)]
pub use crate::impedance::{
    ComplexNumber, ComplexNumberBuilder, ComplexNumberType, Impedance, ImpedanceBuilder,
    ImpedanceMode, ImpedanceType,
};

#[doc(no_inline)]
pub use crate::minimize::{
    Bracket, BracketOptions, BracketResult, Brent, BrentResult, CmaEs, CmaEsResult, ConjGrad,
    ConjGradMethod, ConjGradResult, Constraint, DBrent, DBrentMethod, DBrentResult, F1dim, GF1dim,
    Golden, GoldenResult, HF1dim, InteriorPoint, InteriorPointMethod, InteriorPointParams,
    InteriorPointResult, LinearConstraint, Minimizer, MultiDimFn, MultiDimGradFn, MultiDimHessFn,
    MultiDimNumGradFn, NelderMead, NelderMeadMethod, NelderMeadOptions, NelderMeadResult, ObjDerFn,
    ObjFn, Powell, PowellResult, QuadraticConstraint, QuasiNewton, QuasiNewtonMethod,
    QuasiNewtonResult, Simplex, SimplexResult, SingleDimDerFn, SingleDimFn, WolfeParams,
};

#[doc(no_inline)]
pub use crate::network::{
    Network, NetworkBuilder, NetworkPoint, NetworkPortPoints, PortPoints, PortVal,
};

#[doc(no_inline)]
pub use crate::points;
#[doc(no_inline)]
pub use crate::pts::{Matrix, Points, Points1, Points2, Points3, Pts};

#[doc(no_inline)]
pub use crate::parameter::RFParameter;

#[doc(no_inline)]
pub use crate::scale::Scale;

#[doc(no_inline)]
pub use crate::unit::{Sweep, Unit, UnitVal, UnitValBuilder, UnitValue, Unitized};
