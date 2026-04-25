//! rfkit-network prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! use rfkit_network::prelude::*;
//!
//! ```

#[doc(no_inline)]
pub use crate::parameter::RFParameter;

#[doc(no_inline)]
pub use crate::network::{
    Network, NetworkBuilder, NetworkPoint, NetworkPortPoints, WaveType, read_touchstone,
};
