use super::*;
use crate::num::{RealScalar, Scalar, ToComplex, ToReal};
use ndarray::{Array, Dimension};
use num_complex::Complex;

/// Trait for types that can be used as values in UnitValue.
/// This allows both scalar values (f64, TwoFloat) and array values (Array1<f64>, Array1<TwoFloat>).
pub trait UnitValue<T: Scalar + Scaleable>: Clone + Default {
    type Value: Clone + Scaleable;

    fn new(val: &Self::Value, scale: Scale, unit: Unit) -> Self;
    fn new_scaled(val: &Self::Value, scale: Scale, unit: Unit) -> Self;
    fn builder() -> UnitValueBuilder<T, Self>;
    fn npts(&self) -> usize;
    fn val(&self) -> Self::Value;
    fn val_ref(&self) -> &Self::Value;
    fn val_scaled(&self) -> Self::Value;
    fn scale(&self) -> Scale;
    fn unit(&self) -> Unit;
    fn set_val(&mut self, val: &Self::Value) -> &Self;
    fn set_val_pt(&mut self, val: T, idx: usize) -> &Self;
    fn set_val_scaled(&mut self, val: &Self::Value) -> &Self;
    fn set_val_scaled_pt(&mut self, val: T, idx: usize) -> &Self;
    fn set_scale(&mut self, scale: Scale) -> &Self;
    fn set_scale_str(&mut self, scale: &str) -> &Self;
    fn set_unit(&mut self, unit: Unit) -> &Self;
    fn set_unit_str(&mut self, unit: &str) -> &Self;
    fn zero_value() -> Self::Value;
}

/// Trait extension for UnitValue that provides frequency-specific operations.
/// This trait adds methods for angular frequency, wavelength calculations,
/// and other frequency-specific functionality.
pub trait Frequency<T: RealScalar>: UnitValue<T> {
    fn new_freq(val: &Self::Value, scale: Scale) -> Self;
    fn new_freq_scaled(val: &Self::Value, scale: Scale) -> Self;
    fn fpts(&self) -> usize;
    fn freq(&self) -> Self::Value;
    fn freq_scalar(&self, idx: usize) -> ScalarUnitValue<T>;
    fn freq_pt(&self, idx: usize) -> T;
    fn freq_ref(&self) -> &Self::Value;
    fn freq_scaled(&self) -> Self::Value;
    fn w(&self) -> Self::Value;
    fn w_pt(&self, idx: usize) -> T;
    fn wavelength(&self, er: T) -> Self::Value;
    fn set_freq(&mut self, freq: &Self::Value);
    fn set_freq_pt(&mut self, freq: T, pt: usize);
    fn set_freq_scaled(&mut self, freq: &Self::Value);
    fn set_freq_scaled_pt(&mut self, freq: T, pt: usize);
}

/// Helper trait to eliminate verbosity when specifying a generic UnitValue that is a frequency
pub trait FreqValue<T: RealScalar>: UnitValue<T> + Frequency<T> + MapScalar<T> {}

impl<T: RealScalar, U: UnitValue<T> + Frequency<T> + MapScalar<T>> FreqValue<T> for U {}

/// Trait for mapping a scalar-freq function `f(T::Real) -> Complex<T::Real>` over a `UnitValue`.
///
/// - `ScalarUnitValue` → calls `f(freq)` once, returns `Complex<T>`
/// - `ArrayUnitValue`  → calls `f(freq[i])` for each point, returns `Array1<Complex<T>>`
///
/// This allows element impedance functions to be written once in scalar form and
/// automatically produce the correct output type depending on whether the input
/// frequency is scalar or array-valued.
pub trait MapToReal<T: RealScalar + ToReal>: ToReal {
    fn real_at(&self, val: &Self::ROutput, idx: usize) -> T;
    fn real_idx<D>(&self, val: &Array<T, D>, idx: D) -> T
    where
        D: Dimension,
    {
        val[idx]
    }
    fn map_to_real<F>(&self, f: F) -> Self::ROutput
    where
        F: Fn(T) -> T;
}

/// Trait for mapping a scalar-freq function `f(T::Real) -> Complex<T::Real>` over a `UnitValue`.
///
/// - `ScalarUnitValue` → calls `f(freq)` once, returns `Complex<T>`
/// - `ArrayUnitValue`  → calls `f(freq[i])` for each point, returns `Array1<Complex<T>>`
///
/// This allows element impedance functions to be written once in scalar form and
/// automatically produce the correct output type depending on whether the input
/// frequency is scalar or array-valued.
pub trait MapToComplex<T: RealScalar + ToComplex<COutput = Complex<T>>>: ToComplex {
    fn complex_at(&self, val: &Self::COutput, idx: usize) -> Complex<T>;
    fn complex_idx<D>(&self, val: &Array<Complex<T>, D>, idx: D) -> Complex<T>
    where
        D: Dimension,
    {
        val[idx]
    }
    fn map_to_complex<F>(&self, f: F) -> Self::COutput
    where
        F: Fn(T) -> Complex<T>;
}

pub trait MapScalar<T: RealScalar> {
    type ROutput;
    type COutput;

    fn map_scalar_to_real<F>(&self, f: F) -> Self::ROutput
    where
        F: Fn(&ScalarUnitValue<T>) -> T;

    fn map_scalar_to_complex<F>(&self, f: F) -> Self::COutput
    where
        F: Fn(&ScalarUnitValue<T>) -> Complex<T>;

    fn map_scalar_to_vec<R, F>(&self, f: F) -> Vec<R>
    where
        F: FnMut(&ScalarUnitValue<T>) -> R;
}

pub trait Scaleable {
    fn scale(&self, scale: Scale) -> Self;
    fn unscale(&self, scale: Scale) -> Self;
}

pub trait FrequencyBuilder<T: RealScalar, U: UnitValue<T>> {
    fn freq(self, freq: &U::Value) -> Self;
    fn freq_scaled(self, freq: &U::Value, scale: Scale) -> Self;
    fn start_stop_step(self, start: T, stop: T, step: T) -> Self;
    fn start_stop_step_scaled(self, start: T, stop: T, step: T, scale: Scale) -> Self;
    fn start_stop_npts(self, start: T, stop: T, npts: usize, sweep: Sweep) -> Self;
    fn start_stop_npts_scaled(
        self,
        start: T,
        stop: T,
        npts: usize,
        scale: Scale,
        sweep: Sweep,
    ) -> Self;
}
