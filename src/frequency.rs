use crate::{
    num::{RealScalar, ScalarConst},
    scale::Scale,
    unit::{ArrayUnitValue, ScalarUnitValue, Sweep, Unit, UnitValBuilder, UnitValue},
    util::{ApproxEq, NumMargin},
};
use ndarray::{concatenate, prelude::*};

// /// Type alias for a frequency array (maintains backward compatibility with existing code)
// pub type Frequency<T> = UnitValue<Array1<T>>;

// /// Type alias for a scalar frequency value
// pub type FrequencyScalar<T> = UnitValue<T>;

// /// Create a new Frequency from unscaled values in Hz
// /// This is a convenience function for backward compatibility
// pub fn new_frequency<T>(vals: Array1<T>, scale: Scale) -> Frequency<T> {
//     UnitValue::new(vals, scale, Unit::Hz)
// }

// /// Create a new Frequency from scaled values
// /// This is a convenience function for backward compatibility
// pub fn new_frequency_scaled(vals: Array1<f64>, scale: Scale) -> Frequency {
//     UnitValue::new_scaled(vals, scale, Unit::Hz)
// }

/// Trait extension for UnitValue that provides frequency-specific operations.
/// This trait adds methods for angular frequency, wavelength calculations,
/// and other frequency-specific functionality.
pub trait Frequency<T: UnitValue> {
    fn new_freq(val: &T::Value, scale: Scale) -> T;
    fn new_freq_scaled(val: &T::Value, scale: Scale) -> T;
    fn freqpts(&self) -> usize;
    fn freq(&self) -> T::Value;
    fn freq_ref(&self) -> &T::Value;
    fn freq_scaled(&self) -> T::Value;
    fn w(&self) -> T::Value;
    fn wavelength(&self, er: T::Scalar) -> T::Value;
    fn set_freq(&self, freq: &T::Value) -> Self;
    fn set_freq_inplace(&mut self, freq: &T::Value);
    fn set_freq_pt_inplace(&mut self, freq: T::Scalar, pt: usize);
    fn set_freq_scaled(&self, freq: &T::Value) -> Self;
    fn set_freq_scaled_inplace(&mut self, freq: &T::Value);
    fn set_freq_pt_scaled_inplace(&mut self, freq: T::Scalar, pt: usize);
}

impl<T: RealScalar> Frequency<ScalarUnitValue<T>> for ScalarUnitValue<T> {
    fn new_freq(
        val: &<ScalarUnitValue<T> as UnitValue>::Value,
        scale: Scale,
    ) -> ScalarUnitValue<T> {
        ScalarUnitValue::builder()
            .val(val)
            .scale(scale)
            .unit(Unit::Hz)
            .build()
            .unwrap()
    }

    fn new_freq_scaled(
        val: &<ScalarUnitValue<T> as UnitValue>::Value,
        scale: Scale,
    ) -> ScalarUnitValue<T> {
        ScalarUnitValue::builder()
            .val_scaled(val, scale)
            .unit(Unit::Hz)
            .build()
            .unwrap()
    }

    fn freqpts(&self) -> usize {
        self.npts()
    }

    fn freq(&self) -> <ScalarUnitValue<T> as UnitValue>::Value {
        self.val()
    }

    fn freq_ref(&self) -> &<ScalarUnitValue<T> as UnitValue>::Value {
        self.val_ref()
    }

    fn freq_scaled(&self) -> <ScalarUnitValue<T> as UnitValue>::Value {
        self.val_scaled()
    }

    fn w(&self) -> <ScalarUnitValue<T> as UnitValue>::Value {
        T::C2 * T::PI_C * self.val
    }

    fn wavelength(
        &self,
        er: <ScalarUnitValue<T> as UnitValue>::Scalar,
    ) -> <ScalarUnitValue<T> as UnitValue>::Value {
        T::C_C / (self.val * er.sqrt())
    }

    fn set_freq(&self, freq: &<ScalarUnitValue<T> as UnitValue>::Value) -> Self {
        self.set_val(*freq)
    }

    fn set_freq_inplace(&mut self, freq: &<ScalarUnitValue<T> as UnitValue>::Value) {
        self.set_val_inplace(*freq);
    }

    fn set_freq_pt_inplace(&mut self, freq: <ScalarUnitValue<T> as UnitValue>::Scalar, _pt: usize) {
        self.set_val_pt_inplace(freq);
    }

    fn set_freq_scaled(&self, freq: &<ScalarUnitValue<T> as UnitValue>::Value) -> Self {
        self.set_val_scaled(*freq)
    }

    fn set_freq_scaled_inplace(&mut self, freq: &<ScalarUnitValue<T> as UnitValue>::Value) {
        self.set_val_scaled_inplace(*freq);
    }

    fn set_freq_pt_scaled_inplace(
        &mut self,
        freq: <ScalarUnitValue<T> as UnitValue>::Scalar,
        _pt: usize,
    ) {
        self.set_val_pt_scaled_inplace(freq);
    }
}

impl<T: RealScalar> Frequency<ArrayUnitValue<T>> for ArrayUnitValue<T> {
    fn new_freq(val: &<ArrayUnitValue<T> as UnitValue>::Value, scale: Scale) -> ArrayUnitValue<T> {
        ArrayUnitValue::builder()
            .val(val)
            .scale(scale)
            .build()
            .unwrap()
    }

    fn new_freq_scaled(
        val: &<ArrayUnitValue<T> as UnitValue>::Value,
        scale: Scale,
    ) -> ArrayUnitValue<T> {
        ArrayUnitValue::builder()
            .val_scaled(val, scale)
            .build()
            .unwrap()
    }

    fn freqpts(&self) -> usize {
        self.npts()
    }
    fn freq(&self) -> <ArrayUnitValue<T> as UnitValue>::Value {
        self.val()
    }

    fn freq_ref(&self) -> &<ArrayUnitValue<T> as UnitValue>::Value {
        self.val_ref()
    }

    fn freq_scaled(&self) -> <ArrayUnitValue<T> as UnitValue>::Value {
        self.val_scaled()
    }

    fn w(&self) -> <ArrayUnitValue<T> as UnitValue>::Value {
        self.val.map(|&x| T::C2 * T::PI_C * x)
    }

    fn wavelength(
        &self,
        er: <ArrayUnitValue<T> as UnitValue>::Scalar,
    ) -> <ArrayUnitValue<T> as UnitValue>::Value {
        let er_sqrt = er.sqrt();
        self.val.map(|&x| T::C_C / (x * er_sqrt))
    }

    fn set_freq(&self, freq: &<ArrayUnitValue<T> as UnitValue>::Value) -> Self {
        self.set_val(freq)
    }

    fn set_freq_inplace(&mut self, freq: &<ArrayUnitValue<T> as UnitValue>::Value) {
        self.set_val_inplace(freq);
    }

    fn set_freq_pt_inplace(&mut self, freq: <ArrayUnitValue<T> as UnitValue>::Scalar, pt: usize) {
        self.set_val_pt_inplace(freq, pt);
    }

    fn set_freq_scaled(&self, freq: &<ArrayUnitValue<T> as UnitValue>::Value) -> Self {
        self.set_val_scaled(freq)
    }

    fn set_freq_scaled_inplace(&mut self, freq: &<ArrayUnitValue<T> as UnitValue>::Value) {
        self.set_val_scaled_inplace(freq);
    }

    fn set_freq_pt_scaled_inplace(
        &mut self,
        freq: <ArrayUnitValue<T> as UnitValue>::Scalar,
        pt: usize,
    ) {
        self.set_val_pt_scaled_inplace(freq, pt);
    }
}

pub trait FrequencyBuilder<T: UnitValue> {
    fn freq(self, freq: &T::Value) -> Self;
    fn freq_scaled(self, freq: &T::Value, scale: Scale) -> Self;
    fn start_stop_step(self, start: T::Scalar, stop: T::Scalar, step: T::Scalar) -> Self;
    fn start_stop_step_scaled(
        self,
        start: T::Scalar,
        stop: T::Scalar,
        step: T::Scalar,
        scale: Scale,
    ) -> Self;
    fn start_stop_npts(self, start: T::Scalar, stop: T::Scalar, npts: usize, sweep: Sweep) -> Self;
    fn start_stop_npts_scaled(
        self,
        start: T::Scalar,
        stop: T::Scalar,
        npts: usize,
        scale: Scale,
        sweep: Sweep,
    ) -> Self;
}

impl<T: RealScalar> FrequencyBuilder<ScalarUnitValue<T>> for UnitValBuilder<ScalarUnitValue<T>> {
    fn freq(mut self, freq: &T) -> Self {
        self.val = Some(*freq);
        self
    }

    fn freq_scaled(mut self, freq: &T, scale: Scale) -> Self {
        self.val = Some(ScalarUnitValue::<T>::unscale_val(freq, scale.multiplier()));
        self.scale = scale;
        self
    }

    /// Provide frequency from start, stop and step in Hz
    fn start_stop_step(self, start: T, _stop: T, _step: T) -> Self {
        self.val(&start)
    }

    /// Provide frequency from start, stop and step in scaled scale
    fn start_stop_step_scaled(self, start: T, _stop: T, _step: T, scale: Scale) -> Self {
        self.val_scaled(&start, scale)
    }

    /// Provide frequency from start, stop, number of points and sweep type in Hz
    fn start_stop_npts(self, start: T, _stop: T, _npts: usize, _sweep: Sweep) -> Self {
        self.val(&start)
    }

    /// Provide frequency from start, stop, number of points and sweep type in scaled scale
    fn start_stop_npts_scaled(
        self,
        start: T,
        _stop: T,
        _npts: usize,
        scale: Scale,
        _sweep: Sweep,
    ) -> Self {
        self.val_scaled(&start, scale)
    }
}

impl<T: RealScalar> FrequencyBuilder<ArrayUnitValue<T>> for UnitValBuilder<ArrayUnitValue<T>>
where
    <T as ApproxEq>::Compare: ScalarConst,
{
    fn freq(mut self, freq: &Array1<T>) -> Self {
        self.val = Some(freq.clone());
        self
    }

    fn freq_scaled(mut self, freq: &Array1<T>, scale: Scale) -> Self {
        self.val = Some(ArrayUnitValue::<T>::unscale_val(freq, scale.multiplier()));
        self.scale = scale;
        self
    }

    /// Provide frequency from start, stop and step in Hz
    fn start_stop_step(self, start: T, stop: T, step: T) -> Self {
        let npts: usize = ((stop - start) / step).floor().to_usize().unwrap() + 1;
        let mut out = Array1::<T>::zeros(npts);

        let mut last_val = T::C0;
        for (i, pt) in out.iter_mut().enumerate() {
            if i == 0 {
                *pt = start;
                last_val = start;
                continue;
            }
            *pt = last_val + step;
            last_val += step;
        }
        if out[npts - 1].approx_ne(&stop, NumMargin::default()) {
            out = concatenate![Axis(0), out, array![stop]];
        }
        self.val(&out)
    }

    /// Provide frequency from start, stop and step in scaled scale
    fn start_stop_step_scaled(self, start: T, stop: T, step: T, scale: Scale) -> Self {
        let npts: usize = ((stop - start) / step).floor().to_usize().unwrap() + 1;
        let mut out = Array1::<T>::zeros(npts);

        let mut last_val = T::C0;
        for (i, pt) in out.iter_mut().enumerate() {
            if i == 0 {
                *pt = start;
                last_val = start;
                continue;
            }
            *pt = last_val + step;
            last_val += step;
        }
        if out[npts - 1].approx_ne(&stop, NumMargin::default()) {
            out = concatenate![Axis(0), out, array![stop]];
        }
        self.val_scaled(&out, scale)
    }

    /// Provide frequency from start, stop, number of points and sweep type in Hz
    fn start_stop_npts(self, start: T, stop: T, npts: usize, sweep: Sweep) -> Self {
        let mut out = Array1::<T>::zeros(npts);

        let mut last_val = T::C0;
        let step = (stop - start) / T::from_usize(npts - 1);
        for (i, pt) in out.iter_mut().enumerate() {
            if i == 0 {
                *pt = start;
                last_val = start;
                continue;
            }
            match sweep {
                Sweep::Linear => {
                    *pt = last_val + step;
                    last_val += step;
                }
                Sweep::Log => {
                    *pt = T::C10.powf(
                        (stop.log10() - start.log10()) / T::from_usize((npts - 1) * i)
                            + start.log10(),
                    );
                    last_val = T::C10.powf(
                        (stop.log10() - start.log10()) / T::from_usize((npts - 1) * i)
                            + start.log10(),
                    );
                }
            }
        }
        out[npts - 1] = stop;

        self.val(&out)
    }

    /// Provide frequency from start, stop, number of points and sweep type in scaled scale
    fn start_stop_npts_scaled(
        self,
        start: T,
        stop: T,
        npts: usize,
        scale: Scale,
        sweep: Sweep,
    ) -> Self {
        let mut out = Array1::<T>::zeros(npts);

        let mut last_val = T::C0;
        let step = (stop - start) / T::from_usize(npts - 1);
        for (i, pt) in out.iter_mut().enumerate() {
            if i == 0 {
                *pt = start;
                last_val = start;
                continue;
            }
            match sweep {
                Sweep::Linear => {
                    *pt = last_val + step;
                    last_val += step;
                }
                Sweep::Log => {
                    *pt = T::C10.powf(
                        (stop.log10() - start.log10()) / T::from_usize((npts - 1) * i)
                            + start.log10(),
                    );
                    last_val = T::C10.powf(
                        (stop.log10() - start.log10()) / T::from_usize((npts - 1) * i)
                            + start.log10(),
                    );
                }
            }
        }
        out[npts - 1] = stop;

        self.val_scaled(&out, scale)
    }
}

// /// Trait for frequency-specific value operations
// pub trait FreqValue {
//     /// The output type for angular frequency
//     type Output;

//     /// Calculate angular frequency (ω = 2πf)
//     fn angular_freq(&self) -> Self::Output;

//     /// Calculate wavelength given relative permittivity
//     fn wavelength(&self, er: f64) -> Self::Output;
// }

// /// Extension trait for scalar frequency operations
// pub trait FreqScalar {
//     /// Get the unscaled frequency value in Hz
//     fn freq(&self) -> f64;

//     /// Get the scaled frequency value
//     fn freq_scaled(&self) -> f64;

//     /// Get angular frequency (ω = 2πf)
//     fn w(&self) -> f64;

//     /// Get wavelength given relative permittivity
//     fn wavelength(&self, er: f64) -> f64;
// }

// impl FreqScalar for UnitValue<f64> {
//     fn freq(&self) -> f64 {
//         self.val()
//     }

//     fn freq_scaled(&self) -> f64 {
//         self.val_scaled()
//     }

//     fn w(&self) -> f64 {
//         self.val() * 2.0 * PI
//     }

//     fn wavelength(&self, er: f64) -> f64 {
//         3e8 / (self.val() * er.sqrt())
//     }
// }

// /// Extension trait for array frequency operations
// pub trait FreqArray {
//     /// Get the unscaled frequency values in Hz
//     fn freqs(&self) -> Array1<f64>;

//     /// Get the scaled frequency values
//     fn freqs_scaled(&self) -> Array1<f64>;

//     /// Get single frequency at index (unscaled)
//     fn freq(&self, pt: usize) -> f64;

//     /// Get single frequency at index (scaled)
//     fn freq_scaled(&self, pt: usize) -> f64;

//     /// Get angular frequencies (ω = 2πf)
//     fn w(&self) -> Array1<f64>;

//     /// Get angular frequency at index
//     fn w_pt(&self, pt: usize) -> f64;

//     /// Get wavelengths given relative permittivity
//     fn wavelengths(&self, er: f64) -> Array1<f64>;

//     /// Get wavelength at index given relative permittivity
//     fn wavelength(&self, er: f64, pt: usize) -> f64;

//     /// Get number of frequency points
//     fn npts(&self) -> usize;

//     /// Get UnitValue for a single frequency point
//     fn unitval(&self, pt: usize) -> UnitValue<f64>;

//     /// Set single frequency at index (unscaled)
//     fn set_freq(&mut self, val: f64, pt: usize) -> &mut Self;

//     /// Set single frequency at index (scaled)
//     fn set_freq_scaled(&mut self, val: f64, pt: usize) -> &mut Self;

//     /// Set all frequencies (unscaled)
//     fn set_freqs(&mut self, vals: Array1<f64>) -> &mut Self;

//     /// Set all frequencies (scaled)
//     fn set_freqs_scaled(&mut self, vals: Array1<f64>) -> &mut Self;

//     /// Set scale using string
//     fn set_scale_str(&mut self, scale: &str) -> &mut Self;
// }

// impl FreqArray for UnitValue<Array1<f64>> {
//     fn freqs(&self) -> Array1<f64> {
//         self.val_ref().clone()
//     }

//     fn freqs_scaled(&self) -> Array1<f64> {
//         self.val_scaled()
//     }

//     fn freq(&self, pt: usize) -> f64 {
//         self.val_ref()[pt]
//     }

//     fn freq_scaled(&self, pt: usize) -> f64 {
//         self.scale().scale(self.val_ref()[pt])
//     }

//     fn w(&self) -> Array1<f64> {
//         let vals = self.val_ref();
//         Array1::<f64>::from_shape_fn(vals.dim(), |i| vals[i] * 2.0 * PI)
//     }

//     fn w_pt(&self, pt: usize) -> f64 {
//         self.val_ref()[pt] * 2.0 * PI
//     }

//     fn wavelengths(&self, er: f64) -> Array1<f64> {
//         let vals = self.val_ref();
//         Array1::<f64>::from_shape_fn(vals.dim(), |i| 3e8 / (vals[i] * er.sqrt()))
//     }

//     fn wavelength(&self, er: f64, pt: usize) -> f64 {
//         3e8 / (self.val_ref()[pt] * er.sqrt())
//     }

//     fn npts(&self) -> usize {
//         self.val_ref().len()
//     }

//     fn unitval(&self, pt: usize) -> UnitValue<f64> {
//         UnitValBuilder::new()
//             .val(self.val_ref()[pt])
//             .scale(self.scale())
//             .unit(Unit::Hz)
//             .build()
//     }

//     fn set_freq(&mut self, val: f64, pt: usize) -> &mut Self {
//         // We need to modify the internal array
//         // Since we can't directly mutate val_ref, we get the array, modify it, and set it back
//         let mut vals = self.val_ref().clone();
//         vals[pt] = val;
//         self.set_val(vals);
//         self
//     }

//     fn set_freq_scaled(&mut self, val: f64, pt: usize) -> &mut Self {
//         let mut vals = self.val_ref().clone();
//         vals[pt] = self.scale().unscale(val);
//         self.set_val(vals);
//         self
//     }

//     fn set_freqs(&mut self, vals: Array1<f64>) -> &mut Self {
//         self.set_val(vals);
//         self
//     }

//     fn set_freqs_scaled(&mut self, vals: Array1<f64>) -> &mut Self {
//         self.set_val_scaled(vals);
//         self
//     }

//     fn set_scale_str(&mut self, scale: &str) -> &mut Self {
//         self.set_scale(Scale::from_str(scale).unwrap());
//         self
//     }
// }

// /// Builder design pattern for FrequencyScalar (single frequency value)
// ///
// /// ## Example
// /// ```
// /// use rfkit::prelude::*;
// ///
// /// let freq = FrequencyScalarBuilder::new().val_scaled(1.0, Scale::Giga).build();
// /// ```
// #[derive(Default)]
// pub struct FrequencyScalarBuilder {
//     val: f64,
//     scale: Scale,
// }

// impl FrequencyScalarBuilder {
//     pub fn new() -> Self {
//         FrequencyScalarBuilder {
//             val: 1e9,
//             scale: Scale::Giga,
//         }
//     }

//     /// Provide frequency value in Hz
//     pub fn val(mut self, val: f64) -> Self {
//         self.val = val;
//         self
//     }

//     /// Provide frequency value in scaled scale
//     pub fn val_scaled(mut self, val: f64, scale: Scale) -> Self {
//         self.val = scale.unscale(val);
//         self.scale = scale;
//         self
//     }

//     /// Provide scale for frequency
//     pub fn scale(mut self, scale: Scale) -> Self {
//         self.scale = scale;
//         self
//     }

//     pub fn build(self) -> FrequencyScalar {
//         UnitValue::new(self.val, self.scale, Unit::Hz)
//     }
// }

// /// Builder design pattern for Frequency (array of frequencies)
// ///
// /// ## Example
// /// ```
// /// use ndarray::prelude::*;
// /// use rfkit::prelude::*;
// ///
// /// let freq1 = FrequencyBuilder::new().freqs_scaled(array![1.0, 2.0, 3.0], Scale::Giga).build();
// ///
// /// let freq2 = FrequencyBuilder::new().start_stop_step_scaled(1.0, 3.0, 1.0, Scale::Giga).build();
// /// ```
// #[derive(Default)]
// pub struct FrequencyBuilder {
//     step: Option<(f64, f64, f64)>,
//     pts: Option<(f64, f64, usize, Sweep)>,
//     vals: Array1<f64>,
//     scale: Scale,
// }

// impl FrequencyBuilder {
//     pub fn new() -> Self {
//         FrequencyBuilder {
//             step: None,
//             pts: None,
//             vals: array![1e9],
//             scale: Scale::Giga,
//         }
//     }

//     /// Provide Array1<f64> of frequency values in Hz
//     pub fn freqs(mut self, vals: Array1<f64>) -> Self {
//         self.vals = vals;
//         self
//     }

//     /// Provide Array1<f64> of frequency values in scaled scale
//     pub fn freqs_scaled(mut self, vals: Array1<f64>, scale: Scale) -> Self {
//         self.vals = Array1::<f64>::from_shape_fn(vals.dim(), |i| scale.unscale(vals[i]));
//         self.scale = scale;
//         self
//     }

//     /// Provide frequency from start, stop and step in Hz
//     pub fn start_stop_step(mut self, start: f64, stop: f64, step: f64) -> Self {
//         self.step = Some((start, stop, step));
//         self
//     }

//     /// Provide frequency from start, stop and step in scaled scale
//     pub fn start_stop_step_scaled(
//         mut self,
//         start: f64,
//         stop: f64,
//         step: f64,
//         scale: Scale,
//     ) -> Self {
//         self.scale = scale;
//         self.step = Some((
//             self.scale.unscale(start),
//             self.scale.unscale(stop),
//             self.scale.unscale(step),
//         ));
//         self
//     }

//     /// Provide frequency from start, stop, number of points and sweep type in Hz
//     pub fn start_stop_npts(mut self, start: f64, stop: f64, npts: usize, sweep: Sweep) -> Self {
//         self.pts = Some((start, stop, npts, sweep));
//         self
//     }

//     /// Provide frequency from start, stop, number of points and sweep type in scaled scale
//     pub fn start_stop_npts_scaled(
//         mut self,
//         start: f64,
//         stop: f64,
//         npts: usize,
//         scale: Scale,
//         sweep: Sweep,
//     ) -> Self {
//         self.scale = scale;
//         self.pts = Some((
//             self.scale.unscale(start),
//             self.scale.unscale(stop),
//             npts,
//             sweep,
//         ));
//         self
//     }

//     /// Provide scale for frequencies
//     pub fn scale(mut self, scale: Scale) -> Self {
//         self.scale = scale;
//         self
//     }

//     pub fn build(self) -> Frequency {
//         let mut last_val = 0.0;
//         match (self.step, self.pts) {
//             (Some(x), _) => {
//                 let npts: usize = ((x.1 - x.0) / x.2).floor() as usize + 1;
//                 let mut out = Array1::<f64>::zeros(npts);

//                 for (i, pt) in out.iter_mut().enumerate() {
//                     if i == 0 {
//                         pt.assign(x.0);
//                         last_val = x.0;
//                         continue;
//                     }
//                     pt.assign(last_val + x.2);
//                     last_val += x.2;
//                 }
//                 if out[npts - 1].approx_ne(x.1, NumMargin::default()) {
//                     out = concatenate![Axis(0), out, array![x.1]];
//                 }

//                 UnitValue::new(out, self.scale, Unit::Hz)
//             }
//             (_, Some(x)) => {
//                 let mut out = Array1::<f64>::zeros(x.2);

//                 let step = (x.1 - x.0) / (x.2 - 1) as f64;
//                 for (i, pt) in out.iter_mut().enumerate() {
//                     if i == 0 {
//                         pt.assign(x.0);
//                         last_val = x.0;
//                         continue;
//                     }
//                     match x.3 {
//                         Sweep::Linear => {
//                             pt.assign(last_val + step);
//                             last_val += step;
//                         }
//                         Sweep::Log => {
//                             pt.assign(10_f64.powf(
//                                 ((x.1).log10() - x.0.log10()) / (x.2 - 1) as f64 * i as f64
//                                     + x.0.log10(),
//                             ));
//                             last_val = 10_f64.powf(
//                                 ((x.1).log10() - x.0.log10()) / (x.2 - 1) as f64 * i as f64
//                                     + x.0.log10(),
//                             );
//                         }
//                     }
//                 }
//                 out[x.2 - 1] = x.1;

//                 UnitValue::new(out, self.scale, Unit::Hz)
//             }
//             (_, _) => UnitValue::new(self.vals, self.scale, Unit::Hz),
//         }
//     }
// }

#[cfg(test)]
mod frequency_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin, comp_array_f64};

    #[test]
    fn test_frequency_scalar() {
        let val = 280e9;
        let val_scaled = 280.0;
        let scale = Scale::Giga;
        let er: f64 = 3.4;

        let freq = ScalarUnitValue::builder()
            .val(&val)
            .scale(scale)
            .build()
            .unwrap();

        val.assert_approx_eq(&freq.freq(), NumMargin::default(), "frequency::freq()", "");
        val_scaled.assert_approx_eq(
            &freq.freq_scaled(),
            NumMargin::default(),
            "frequency::freq_scaled()",
            "",
        );

        let w = val * 2.0 * std::f64::consts::PI;
        w.assert_approx_eq(&freq.w(), NumMargin::default(), "frequency::w()", "");

        let wavelength = 3e8 / (val * er.sqrt());
        wavelength.assert_approx_eq(
            &freq.wavelength(er),
            NumMargin::default(),
            "frequency::wavelength()",
            "",
        );
    }

    #[test]
    fn test_frequency_array() {
        let vals = array![280e9, 300e9, 350e9, 500e9];
        let vals_scaled = array![280.0, 300.0, 350.0, 500.0];
        let w = array![
            1759291886010.2842,
            1884955592153.876,
            2199114857512.855,
            3141592653589.793,
        ];
        let er = 3.4;
        let wavelength = array![
            0.0005810637262999719,
            0.0005423261445466404,
            0.00046485098103997755,
            0.0003253956867279843,
        ];
        let scale = Scale::Giga;

        let freq = ArrayUnitValue::builder()
            .freq(&vals)
            .scale(scale)
            .build()
            .unwrap();

        vals[0].assert_approx_eq(
            &freq.freq()[0],
            NumMargin::default(),
            "frequency::freq()",
            "0",
        );
        vals_scaled[0].assert_approx_eq(
            &freq.freq_scaled()[0],
            NumMargin::default(),
            "frequency::freq_scaled()",
            "0",
        );
        comp_array_f64(
            vals.view(),
            freq.freq().view(),
            NumMargin::default(),
            "frequency::freq()",
        );
        comp_array_f64(
            vals_scaled.view(),
            freq.freq_scaled().view(),
            NumMargin::default(),
            "frequency::freq_scaled()",
        );
        comp_array_f64(
            w.view(),
            freq.w().view(),
            NumMargin::default(),
            "frequency::w()",
        );
        w[0].assert_approx_eq(&freq.w()[0], NumMargin::default(), "frequency::w_pt()", "0");
        comp_array_f64(
            wavelength.view(),
            freq.wavelength(er).view(),
            NumMargin::default(),
            "frequency::wavelengths()",
        );
        wavelength[0].assert_approx_eq(
            &freq.wavelength(er)[0],
            NumMargin::default(),
            "frequency::wavelength()",
            "0",
        );

        let freq = ArrayUnitValue::builder()
            .freq_scaled(&vals_scaled, scale)
            .build()
            .unwrap();
        comp_array_f64(
            vals.view(),
            freq.freq().view(),
            NumMargin::default(),
            "frequency_scaled::freq()",
        );
        comp_array_f64(
            vals_scaled.view(),
            freq.freq_scaled().view(),
            NumMargin::default(),
            "frequency_scaled::freq_scaled()",
        );
        comp_array_f64(
            freq.w().view(),
            w.view(),
            NumMargin::default(),
            "frequency::w()",
        );
        comp_array_f64(
            wavelength.view(),
            freq.wavelength(er).view(),
            NumMargin::default(),
            "frequency_scaled::wavelengths()",
        );

        let mut freq = ArrayUnitValue::builder()
            .freq(&vals)
            .scale(scale)
            .build()
            .unwrap();
        let freq_vals = array![290e9, 300e9, 350e9, 500e9];
        freq.set_freq_pt_inplace(290e9, 0);
        comp_array_f64(
            freq_vals.view(),
            freq.freq().view(),
            NumMargin::default(),
            "frequency::set_freq()",
        );

        let freq_vals2 = array![250e9, 300e9, 350e9, 500e9];
        freq.set_freq_pt_scaled_inplace(250.0, 0);
        comp_array_f64(
            freq_vals2.view(),
            freq.freq().view(),
            NumMargin::default(),
            "frequency::set_freq_pt_scaled_inplace()",
        );

        let freq_vals_new = array![200e9, 320e9, 310e9, 550e9];
        freq.set_freq_inplace(&freq_vals_new);
        comp_array_f64(
            freq_vals_new.view(),
            freq.freq().view(),
            NumMargin::default(),
            "frequency::set_freqs()",
        );

        let freq_vals_new_scaled = array![210.0, 340.0, 400.0, 510.0];
        freq.set_freq_scaled_inplace(&freq_vals_new_scaled);
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freq_scaled().view(),
            NumMargin::default(),
            "frequency::set_freqs_scaled()",
        );

        let freq_vals_new_scaled = array![210e3, 340e3, 400e3, 510e3];
        freq.set_scale(Scale::Mega);
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freq_scaled().view(),
            NumMargin::default(),
            "frequency::set_scale()",
        );

        let freq_vals_new_scaled = array![210e6, 340e6, 400e6, 510e6];
        freq.set_scale_str("kHz");
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freq_scaled().view(),
            NumMargin::default(),
            "frequency::set_scale_str()",
        );
    }

    // #[test]
    // fn test_frequencybuilder() {
    //     let vals = array![280e9, 300e9, 350e9, 500e9];
    //     let vals_scaled = array![280.0, 300.0, 350.0, 500.0];
    //     let vals_linear = array![280e9, 335e9, 390e9, 445e9, 500e9];
    //     let vals_log = array![
    //         280e9,
    //         323.6763921413955e9,
    //         374.1657386773942e9,
    //         432.53077270721106e9,
    //         500e9,
    //     ];
    //     let vals_linear2 = array![280e9, 330e9, 380e9, 430e9, 480e9, 500e9];
    //     let scale = Scale::Giga;

    //     let freq = FrequencyBuilder::new()
    //         .freqs(vals.clone())
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .freqs_scaled(vals_scaled.clone(), scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_step(280e9, 500e9, 55e9)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_linear);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_step(280e9, 500e9, 50e9)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_linear2);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_step_scaled(280.0, 500.0, 55.0, scale)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_linear);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_npts(280e9, 500e9, 5, Sweep::Linear)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_linear);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_npts(280e9, 500e9, 5, Sweep::Log)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_log);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_npts_scaled(280.0, 500.0, 5, scale, Sweep::Linear)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_linear);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);

    //     let freq = FrequencyBuilder::new()
    //         .start_stop_npts_scaled(280.0, 500.0, 5, scale, Sweep::Log)
    //         .scale(scale)
    //         .build();
    //     assert_eq!(freq.val_ref(), &vals_log);
    //     assert_eq!(freq.scale(), scale);
    //     assert_eq!(freq.unit(), Unit::Hz);
    // }
}
