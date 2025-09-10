//! rfkit-base-ndarray prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! use rfkit_base_ndarray::prelude::*;
//!
//! ```

#[doc(no_inline)]
pub use crate::file::read_touchstone;

#[doc(no_inline)]
pub use crate::frequency::{Frequency, FrequencyBuilder};

#[doc(no_inline)]
pub use crate::elements::capacitor::{Capacitor, CapacitorBuilder};
#[doc(no_inline)]
pub use crate::elements::inductor::{Inductor, InductorBuilder};
#[doc(no_inline)]
pub use crate::elements::port::{Port, PortBuilder};
#[doc(no_inline)]
pub use crate::elements::resistor::{Resistor, ResistorBuilder};
#[doc(no_inline)]
pub use crate::elements::{Element, ElementBuilder};

#[doc(no_inline)]
pub use crate::impedance::{
    ComplexNumber, ComplexNumberBuilder, ComplexNumberType, Impedance, ImpedanceBuilder,
    ImpedanceMode, ImpedanceType,
};

#[doc(no_inline)]
pub use crate::minimize::{Minimizer, NelderMead, NelderMeadBounded, ObjectiveDerFn, ObjectiveFn};

#[doc(no_inline)]
pub use crate::mycomplex::*;
#[doc(no_inline)]
pub use crate::myfloat::MyFloat;
#[doc(no_inline)]
pub use num::ToPrimitive;

#[doc(no_inline)]
pub use crate::network::{
    Network, NetworkBuilder, NetworkPoint, NetworkPortPoints, PortPoints, PortPointsf64, PortVal,
};

#[doc(no_inline)]
pub use crate::point::{Point, PointComplex, PointFloat, Pointf64, Pt};
#[doc(no_inline)]
pub use crate::points::{Points, Pointsf64, Pts};

#[doc(no_inline)]
pub use crate::parameter::RFParameter;

#[doc(no_inline)]
pub use crate::scale::Scale;

#[doc(no_inline)]
pub use crate::unit::{Sweep, Unit, UnitVal, UnitValBuilder};
