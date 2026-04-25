//! Facade crate for the split `rfkit` workspace.
//!
//! The implementation lives in focused crates (`rfkit-base`, `rfkit-network`,
//! `rfkit-circuit`, and `rfkit-minimize`). This crate keeps the public API easy
//! to use while still exposing those internal boundaries when callers want them.

pub mod base {
    pub use rfkit_base::{error, impedance, math, num, prelude, pts, units, util};
    pub use rfkit_base::{points, prelude::*};
}

pub mod circuit {
    pub use rfkit_circuit::prelude::*;
    pub use rfkit_circuit::{circuit, element, prelude};
}

pub mod minimize {
    pub use rfkit_minimize::prelude::*;
    pub use rfkit_minimize::{error, minimize, prelude};
}

pub mod network {
    pub use rfkit_network::prelude::*;
    pub use rfkit_network::{network, parameter, prelude};
}

pub mod element {
    pub use rfkit_circuit::element::*;
}

pub mod error {
    pub use rfkit_base::error::*;
}

pub mod impedance {
    pub use rfkit_base::impedance::*;
}

pub mod math {
    pub use rfkit_base::math::*;
}

pub mod num {
    pub use rfkit_base::num::*;
}

pub mod parameter {
    pub use rfkit_network::parameter::*;
}

pub mod prelude {
    pub use rfkit_base::prelude::*;
    pub use rfkit_circuit::prelude::*;
    pub use rfkit_minimize::prelude::*;
    pub use rfkit_network::prelude::*;
}

pub mod pts {
    pub use rfkit_base::pts::*;
}

pub mod units {
    pub use rfkit_base::units::*;
}

pub mod util {
    pub use rfkit_base::util::*;
}

pub use rfkit_base::points;
pub use rfkit_base::pts::{Points, Points1, Points2, Points3};
