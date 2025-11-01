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
pub use crate::element::capacitor::{Capacitor, CapacitorBuilder};
#[doc(no_inline)]
pub use crate::element::inductor::{Inductor, InductorBuilder};
#[doc(no_inline)]
pub use crate::element::port::{Port, PortBuilder};
#[doc(no_inline)]
pub use crate::element::resistor::{Resistor, ResistorBuilder};
#[doc(no_inline)]
pub use crate::element::{Element, ElementBuilder};

#[doc(no_inline)]
pub use crate::impedance::{
    ComplexNumber, ComplexNumberBuilder, ComplexNumberType, Impedance, ImpedanceBuilder,
    ImpedanceMode, ImpedanceType,
};

#[doc(no_inline)]
pub use crate::minimize::{Minimizer, ObjDerFn, ObjFn};

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
pub use crate::point::{Point, Pt};
#[doc(no_inline)]
pub use crate::points::{Points, Pts};

#[doc(no_inline)]
pub use crate::parameter::RFParameter;

#[doc(no_inline)]
pub use crate::scale::Scale;

#[doc(no_inline)]
pub use crate::unit::{Sweep, Unit, UnitVal, UnitValBuilder};
