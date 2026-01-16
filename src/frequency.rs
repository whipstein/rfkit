use crate::{
    scale::Scale,
    unit::{Sweep, Unit, UnitVal, UnitValBuilder, UnitValue},
};
use float_cmp::{F64Margin, approx_eq};
use ndarray::{concatenate, prelude::*};
use rug::Assign;
use std::{f64::consts::PI, str::FromStr};

/// Type alias for a frequency array (maintains backward compatibility with existing code)
pub type Frequency = UnitValue<Array1<f64>>;

/// Type alias for a scalar frequency value
pub type FrequencyScalar = UnitValue<f64>;

/// Create a new Frequency from unscaled values in Hz
/// This is a convenience function for backward compatibility
pub fn new_frequency(vals: Array1<f64>, scale: Scale) -> Frequency {
    UnitValue::new(vals, scale, Unit::Hz)
}

/// Create a new Frequency from scaled values
/// This is a convenience function for backward compatibility
pub fn new_frequency_scaled(vals: Array1<f64>, scale: Scale) -> Frequency {
    UnitValue::new_scaled(vals, scale, Unit::Hz)
}

/// Trait extension for UnitValue that provides frequency-specific operations.
/// This trait adds methods for angular frequency, wavelength calculations,
/// and other frequency-specific functionality.
pub trait Freq<T: UnitVal> {
    /// Create a new frequency from unscaled value(s) in Hz
    fn new_freq(val: T, scale: Scale) -> UnitValue<T>;

    /// Create a new frequency from scaled value(s)
    fn new_freq_scaled(val: T, scale: Scale) -> UnitValue<T>;
}

/// Trait for frequency-specific value operations
pub trait FreqValue {
    /// The output type for angular frequency
    type Output;

    /// Calculate angular frequency (ω = 2πf)
    fn angular_freq(&self) -> Self::Output;

    /// Calculate wavelength given relative permittivity
    fn wavelength(&self, er: f64) -> Self::Output;
}

impl Freq<f64> for UnitValue<f64> {
    fn new_freq(val: f64, scale: Scale) -> UnitValue<f64> {
        UnitValue::new(val, scale, Unit::Hz)
    }

    fn new_freq_scaled(val: f64, scale: Scale) -> UnitValue<f64> {
        UnitValue::new_scaled(val, scale, Unit::Hz)
    }
}

impl Freq<Array1<f64>> for UnitValue<Array1<f64>> {
    fn new_freq(val: Array1<f64>, scale: Scale) -> UnitValue<Array1<f64>> {
        UnitValue::new(val, scale, Unit::Hz)
    }

    fn new_freq_scaled(val: Array1<f64>, scale: Scale) -> UnitValue<Array1<f64>> {
        UnitValue::new_scaled(val, scale, Unit::Hz)
    }
}

/// Extension trait for scalar frequency operations
pub trait FreqScalar {
    /// Get the unscaled frequency value in Hz
    fn freq(&self) -> f64;

    /// Get the scaled frequency value
    fn freq_scaled(&self) -> f64;

    /// Get angular frequency (ω = 2πf)
    fn w(&self) -> f64;

    /// Get wavelength given relative permittivity
    fn wavelength(&self, er: f64) -> f64;
}

impl FreqScalar for UnitValue<f64> {
    fn freq(&self) -> f64 {
        self.val()
    }

    fn freq_scaled(&self) -> f64 {
        self.val_scaled()
    }

    fn w(&self) -> f64 {
        self.val() * 2.0 * PI
    }

    fn wavelength(&self, er: f64) -> f64 {
        3e8 / (self.val() * er.sqrt())
    }
}

/// Extension trait for array frequency operations
pub trait FreqArray {
    /// Get the unscaled frequency values in Hz
    fn freqs(&self) -> Array1<f64>;

    /// Get the scaled frequency values
    fn freqs_scaled(&self) -> Array1<f64>;

    /// Get single frequency at index (unscaled)
    fn freq(&self, pt: usize) -> f64;

    /// Get single frequency at index (scaled)
    fn freq_scaled(&self, pt: usize) -> f64;

    /// Get angular frequencies (ω = 2πf)
    fn w(&self) -> Array1<f64>;

    /// Get angular frequency at index
    fn w_pt(&self, pt: usize) -> f64;

    /// Get wavelengths given relative permittivity
    fn wavelengths(&self, er: f64) -> Array1<f64>;

    /// Get wavelength at index given relative permittivity
    fn wavelength(&self, er: f64, pt: usize) -> f64;

    /// Get number of frequency points
    fn npts(&self) -> usize;

    /// Get UnitValue for a single frequency point
    fn unitval(&self, pt: usize) -> UnitValue<f64>;

    /// Set single frequency at index (unscaled)
    fn set_freq(&mut self, val: f64, pt: usize) -> &mut Self;

    /// Set single frequency at index (scaled)
    fn set_freq_scaled(&mut self, val: f64, pt: usize) -> &mut Self;

    /// Set all frequencies (unscaled)
    fn set_freqs(&mut self, vals: Array1<f64>) -> &mut Self;

    /// Set all frequencies (scaled)
    fn set_freqs_scaled(&mut self, vals: Array1<f64>) -> &mut Self;

    /// Set scale using string
    fn set_scale_str(&mut self, scale: &str) -> &mut Self;
}

impl FreqArray for UnitValue<Array1<f64>> {
    fn freqs(&self) -> Array1<f64> {
        self.val_ref().clone()
    }

    fn freqs_scaled(&self) -> Array1<f64> {
        self.val_scaled()
    }

    fn freq(&self, pt: usize) -> f64 {
        self.val_ref()[pt]
    }

    fn freq_scaled(&self, pt: usize) -> f64 {
        self.scale().scale(self.val_ref()[pt])
    }

    fn w(&self) -> Array1<f64> {
        let vals = self.val_ref();
        Array1::<f64>::from_shape_fn(vals.dim(), |i| vals[i] * 2.0 * PI)
    }

    fn w_pt(&self, pt: usize) -> f64 {
        self.val_ref()[pt] * 2.0 * PI
    }

    fn wavelengths(&self, er: f64) -> Array1<f64> {
        let vals = self.val_ref();
        Array1::<f64>::from_shape_fn(vals.dim(), |i| 3e8 / (vals[i] * er.sqrt()))
    }

    fn wavelength(&self, er: f64, pt: usize) -> f64 {
        3e8 / (self.val_ref()[pt] * er.sqrt())
    }

    fn npts(&self) -> usize {
        self.val_ref().len()
    }

    fn unitval(&self, pt: usize) -> UnitValue<f64> {
        UnitValBuilder::new()
            .val(self.val_ref()[pt])
            .scale(self.scale())
            .unit(Unit::Hz)
            .build()
    }

    fn set_freq(&mut self, val: f64, pt: usize) -> &mut Self {
        // We need to modify the internal array
        // Since we can't directly mutate val_ref, we get the array, modify it, and set it back
        let mut vals = self.val_ref().clone();
        vals[pt] = val;
        self.set_val(vals);
        self
    }

    fn set_freq_scaled(&mut self, val: f64, pt: usize) -> &mut Self {
        let mut vals = self.val_ref().clone();
        vals[pt] = self.scale().unscale(val);
        self.set_val(vals);
        self
    }

    fn set_freqs(&mut self, vals: Array1<f64>) -> &mut Self {
        self.set_val(vals);
        self
    }

    fn set_freqs_scaled(&mut self, vals: Array1<f64>) -> &mut Self {
        self.set_val_scaled(vals);
        self
    }

    fn set_scale_str(&mut self, scale: &str) -> &mut Self {
        self.set_scale(Scale::from_str(scale).unwrap());
        self
    }
}

/// Builder design pattern for FrequencyScalar (single frequency value)
///
/// ## Example
/// ```
/// use rfkit::prelude::*;
///
/// let freq = FrequencyScalarBuilder::new().val_scaled(1.0, Scale::Giga).build();
/// ```
#[derive(Default)]
pub struct FrequencyScalarBuilder {
    val: f64,
    scale: Scale,
}

impl FrequencyScalarBuilder {
    pub fn new() -> Self {
        FrequencyScalarBuilder {
            val: 1e9,
            scale: Scale::Giga,
        }
    }

    /// Provide frequency value in Hz
    pub fn val(mut self, val: f64) -> Self {
        self.val = val;
        self
    }

    /// Provide frequency value in scaled scale
    pub fn val_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.val = scale.unscale(val);
        self.scale = scale;
        self
    }

    /// Provide scale for frequency
    pub fn scale(mut self, scale: Scale) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> FrequencyScalar {
        UnitValue::new(self.val, self.scale, Unit::Hz)
    }
}

/// Builder design pattern for Frequency (array of frequencies)
///
/// ## Example
/// ```
/// use ndarray::prelude::*;
/// use rfkit::prelude::*;
///
/// let freq1 = FrequencyBuilder::new().freqs_scaled(array![1.0, 2.0, 3.0], Scale::Giga).build();
///
/// let freq2 = FrequencyBuilder::new().start_stop_step_scaled(1.0, 3.0, 1.0, Scale::Giga).build();
/// ```
#[derive(Default)]
pub struct FrequencyBuilder {
    step: Option<(f64, f64, f64)>,
    pts: Option<(f64, f64, usize, Sweep)>,
    vals: Array1<f64>,
    scale: Scale,
}

impl FrequencyBuilder {
    pub fn new() -> Self {
        FrequencyBuilder {
            step: None,
            pts: None,
            vals: array![1e9],
            scale: Scale::Giga,
        }
    }

    /// Provide Array1<f64> of frequency values in Hz
    pub fn freqs(mut self, vals: Array1<f64>) -> Self {
        self.vals = vals;
        self
    }

    /// Provide Array1<f64> of frequency values in scaled scale
    pub fn freqs_scaled(mut self, vals: Array1<f64>, scale: Scale) -> Self {
        self.vals = Array1::<f64>::from_shape_fn(vals.dim(), |i| scale.unscale(vals[i]));
        self.scale = scale;
        self
    }

    /// Provide frequency from start, stop and step in Hz
    pub fn start_stop_step(mut self, start: f64, stop: f64, step: f64) -> Self {
        self.step = Some((start, stop, step));
        self
    }

    /// Provide frequency from start, stop and step in scaled scale
    pub fn start_stop_step_scaled(
        mut self,
        start: f64,
        stop: f64,
        step: f64,
        scale: Scale,
    ) -> Self {
        self.scale = scale;
        self.step = Some((
            self.scale.unscale(start),
            self.scale.unscale(stop),
            self.scale.unscale(step),
        ));
        self
    }

    /// Provide frequency from start, stop, number of points and sweep type in Hz
    pub fn start_stop_npts(mut self, start: f64, stop: f64, npts: usize, sweep: Sweep) -> Self {
        self.pts = Some((start, stop, npts, sweep));
        self
    }

    /// Provide frequency from start, stop, number of points and sweep type in scaled scale
    pub fn start_stop_npts_scaled(
        mut self,
        start: f64,
        stop: f64,
        npts: usize,
        scale: Scale,
        sweep: Sweep,
    ) -> Self {
        self.scale = scale;
        self.pts = Some((
            self.scale.unscale(start),
            self.scale.unscale(stop),
            npts,
            sweep,
        ));
        self
    }

    /// Provide scale for frequencies
    pub fn scale(mut self, scale: Scale) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> Frequency {
        let mut last_val = 0.0;
        match (self.step, self.pts) {
            (Some(x), _) => {
                let npts: usize = ((x.1 - x.0) / x.2).floor() as usize + 1;
                let mut out = Array1::<f64>::zeros(npts);

                for (i, pt) in out.iter_mut().enumerate() {
                    if i == 0 {
                        pt.assign(x.0);
                        last_val = x.0;
                        continue;
                    }
                    pt.assign(last_val + x.2);
                    last_val += x.2;
                }
                if !approx_eq!(f64, out[npts - 1], x.1, F64Margin::default()) {
                    out = concatenate![Axis(0), out, array![x.1]];
                }

                UnitValue::new(out, self.scale, Unit::Hz)
            }
            (_, Some(x)) => {
                let mut out = Array1::<f64>::zeros(x.2);

                let step = (x.1 - x.0) / (x.2 - 1) as f64;
                for (i, pt) in out.iter_mut().enumerate() {
                    if i == 0 {
                        pt.assign(x.0);
                        last_val = x.0;
                        continue;
                    }
                    match x.3 {
                        Sweep::Linear => {
                            pt.assign(last_val + step);
                            last_val += step;
                        }
                        Sweep::Log => {
                            pt.assign(10_f64.powf(
                                ((x.1).log10() - x.0.log10()) / (x.2 - 1) as f64 * i as f64
                                    + x.0.log10(),
                            ));
                            last_val = 10_f64.powf(
                                ((x.1).log10() - x.0.log10()) / (x.2 - 1) as f64 * i as f64
                                    + x.0.log10(),
                            );
                        }
                    }
                }
                out[x.2 - 1] = x.1;

                UnitValue::new(out, self.scale, Unit::Hz)
            }
            (_, _) => UnitValue::new(self.vals, self.scale, Unit::Hz),
        }
    }
}

#[cfg(test)]
mod frequency_tests {
    use super::*;
    use crate::util::{comp_array_f64, comp_f64};
    use float_cmp::F64Margin;

    #[test]
    fn test_frequency_scalar() {
        let val = 280e9;
        let val_scaled = 280.0;
        let scale = Scale::Giga;
        let er: f64 = 3.4;

        let freq = FrequencyScalarBuilder::new().val(val).scale(scale).build();

        comp_f64(
            &val,
            &freq.freq(),
            F64Margin::default(),
            "frequency::freq()",
            "",
        );
        comp_f64(
            &val_scaled,
            &freq.freq_scaled(),
            F64Margin::default(),
            "frequency::freq_scaled()",
            "",
        );

        let w = val * 2.0 * PI;
        comp_f64(&w, &freq.w(), F64Margin::default(), "frequency::w()", "");

        let wavelength = 3e8 / (val * er.sqrt());
        comp_f64(
            &wavelength,
            &freq.wavelength(er),
            F64Margin::default(),
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

        let freq = FrequencyBuilder::new()
            .freqs(vals.clone())
            .scale(scale)
            .build();

        comp_f64(
            &vals[0],
            &freq.freq(0),
            F64Margin::default(),
            "frequency::freq()",
            "0",
        );
        comp_f64(
            &vals_scaled[0],
            &freq.freq_scaled(0),
            F64Margin::default(),
            "frequency::freq_scaled()",
            "0",
        );
        comp_array_f64(
            vals.view(),
            freq.freqs().view(),
            F64Margin::default(),
            "frequency::freqs()",
        );
        comp_array_f64(
            vals_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency::freqs_scaled()",
        );
        comp_array_f64(
            w.view(),
            freq.w().view(),
            F64Margin::default(),
            "frequency::w()",
        );
        comp_f64(
            &w[0],
            &freq.w_pt(0),
            F64Margin::default(),
            "frequency::w_pt()",
            "0",
        );
        comp_array_f64(
            wavelength.view(),
            freq.wavelengths(er).view(),
            F64Margin::default(),
            "frequency::wavelengths()",
        );
        comp_f64(
            &wavelength[0],
            &freq.wavelength(er, 0),
            F64Margin::default(),
            "frequency::wavelength()",
            "0",
        );

        let freq = FrequencyBuilder::new()
            .freqs_scaled(vals_scaled.clone(), scale)
            .build();
        comp_array_f64(
            vals.view(),
            freq.freqs().view(),
            F64Margin::default(),
            "frequency_scaled::freqs()",
        );
        comp_array_f64(
            vals_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency_scaled::freqs_scaled()",
        );
        comp_array_f64(
            freq.w().view(),
            w.view(),
            F64Margin::default(),
            "frequency::w()",
        );
        comp_array_f64(
            wavelength.view(),
            freq.wavelengths(er).view(),
            F64Margin::default(),
            "frequency_scaled::wavelengths()",
        );

        let mut freq = FrequencyBuilder::new()
            .freqs(vals.clone())
            .scale(scale)
            .build();
        let freq_vals = array![290e9, 300e9, 350e9, 500e9];
        freq.set_freq(290e9, 0);
        comp_array_f64(
            freq_vals.view(),
            freq.freqs().view(),
            F64Margin::default(),
            "frequency::set_freq()",
        );

        let freq_vals2 = array![250e9, 300e9, 350e9, 500e9];
        freq.set_freq_scaled(250.0, 0);
        comp_array_f64(
            freq_vals2.view(),
            freq.freqs().view(),
            F64Margin::default(),
            "frequency::set_freq_scaled()",
        );

        let freq_vals_new = array![200e9, 320e9, 310e9, 550e9];
        freq.set_freqs(freq_vals_new.clone());
        comp_array_f64(
            freq_vals_new.view(),
            freq.freqs().view(),
            F64Margin::default(),
            "frequency::set_freqs()",
        );

        let freq_vals_new_scaled = array![210.0, 340.0, 400.0, 510.0];
        freq.set_freqs_scaled(freq_vals_new_scaled.clone());
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency::set_freqs_scaled()",
        );

        let freq_vals_new_scaled = array![210e3, 340e3, 400e3, 510e3];
        freq.set_scale(Scale::Mega);
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency::set_scale()",
        );

        let freq_vals_new_scaled = array![210e6, 340e6, 400e6, 510e6];
        freq.set_scale_str("kHz");
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency::set_scale_str()",
        );

        let val = UnitValue::new(300e9, Scale::Giga, Unit::Hz);
        let freq_unitval =
            <UnitValue<Array1<f64>> as Freq<Array1<f64>>>::new_freq(array![val.val()], val.scale());
        comp_array_f64(
            array![300e9].view(),
            freq_unitval.freqs().view(),
            F64Margin::default(),
            "frequency::from_unitval()",
        );
        assert_eq!(Scale::Giga, freq_unitval.scale());
        assert_eq!(Unit::Hz, freq_unitval.unit());
    }

    #[test]
    fn test_frequencybuilder() {
        let vals = array![280e9, 300e9, 350e9, 500e9];
        let vals_scaled = array![280.0, 300.0, 350.0, 500.0];
        let vals_linear = array![280e9, 335e9, 390e9, 445e9, 500e9];
        let vals_log = array![
            280e9,
            323.6763921413955e9,
            374.1657386773942e9,
            432.53077270721106e9,
            500e9,
        ];
        let vals_linear2 = array![280e9, 330e9, 380e9, 430e9, 480e9, 500e9];
        let scale = Scale::Giga;

        let freq = FrequencyBuilder::new()
            .freqs(vals.clone())
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .freqs_scaled(vals_scaled.clone(), scale)
            .build();
        assert_eq!(freq.val_ref(), &vals);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_step(280e9, 500e9, 55e9)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_linear);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_step(280e9, 500e9, 50e9)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_linear2);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_step_scaled(280.0, 500.0, 55.0, scale)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_linear);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_npts(280e9, 500e9, 5, Sweep::Linear)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_linear);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_npts(280e9, 500e9, 5, Sweep::Log)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_log);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_npts_scaled(280.0, 500.0, 5, scale, Sweep::Linear)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_linear);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);

        let freq = FrequencyBuilder::new()
            .start_stop_npts_scaled(280.0, 500.0, 5, scale, Sweep::Log)
            .scale(scale)
            .build();
        assert_eq!(freq.val_ref(), &vals_log);
        assert_eq!(freq.scale(), scale);
        assert_eq!(freq.unit(), Unit::Hz);
    }
}
