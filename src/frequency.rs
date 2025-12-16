use crate::scale::Scale;
use crate::unit::{Sweep, Unit, UnitVal, UnitValBuilder};
use float_cmp::{F64Margin, approx_eq};
use ndarray::concatenate;
use ndarray::prelude::*;
use rug::Assign;
use std::f64::consts::PI;
use std::str::FromStr;

/// Representation of a set of frequencies
#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
    vals: Array1<f64>,
    scale: Scale,
    unit: Unit,
}

impl Frequency {
    /// Create a new Frequency from base scale frequencies
    pub fn new(vals: Array1<f64>, scale: Scale) -> Frequency {
        Frequency {
            vals,
            scale,
            unit: Unit::Hz,
        }
    }

    /// Create a new Frequency from scaled frequencies based on scale
    pub fn new_scaled(vals: Array1<f64>, scale: Scale) -> Frequency {
        let freqs = Array1::<f64>::from_shape_fn(vals.dim(), |i| scale.unscale(vals[i]));

        Frequency {
            vals: freqs,
            scale,
            unit: Unit::Hz,
        }
    }

    /// Create a new Frequency from a UnitVal
    pub fn from_unitval(val: &UnitVal) -> Frequency {
        Frequency {
            vals: array![val.val()],
            scale: val.scale(),
            unit: val.unit(),
        }
    }

    /// Retrieve single frequency at pt unscaled
    pub fn freq(&self, pt: usize) -> f64 {
        self.vals[pt]
    }

    /// Retrieve single frequency at pt in scaled scale
    pub fn freq_scaled(&self, pt: usize) -> f64 {
        self.scale.scale(self.vals[pt])
    }

    /// Retrieve frequencies unscaled
    pub fn freqs(&self) -> Array1<f64> {
        self.vals.clone()
    }

    /// Retrieve frequencies in scaled scale
    pub fn freqs_scaled(&self) -> Array1<f64> {
        Array1::<f64>::from_shape_fn(self.vals.dim(), |i| self.scale.scale(self.vals[i]))
    }

    /// Retrieve single frequency at pt as UnitVal
    pub fn unitval(&self, pt: usize) -> UnitVal {
        UnitValBuilder::new()
            .val(self.vals[pt])
            .scale(self.scale)
            .build()
    }

    /// Retrieve Unit of Frequency
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// Retrieve number of frequency points
    pub fn npts(&self) -> usize {
        self.vals.len()
    }

    /// Set single frequency at pt unscaled
    pub fn set_freq(&mut self, val: f64, pt: usize) -> &Self {
        self.vals[pt] = val;
        self
    }

    /// Set single frequency at pt in scaled scale
    pub fn set_freq_scaled(&mut self, val: f64, pt: usize) -> &Self {
        self.vals[pt] = self.scale.unscale(val);
        self
    }

    /// Set frequencies unscaled
    pub fn set_freqs(&mut self, vals: Array1<f64>) -> &Self {
        self.vals = vals;
        self
    }

    /// Set frequencies in scaled scale
    pub fn set_freqs_scaled(&mut self, vals: Array1<f64>) -> &Self {
        self.vals = vals;
        for val in &mut self.vals {
            *val = self.scale.unscale(*val);
        }

        self
    }

    /// Set Unit of Frequency
    pub fn set_unit(&mut self, scale: Scale) -> &Self {
        self.scale = scale;
        self
    }

    /// Set Unit of Frequency
    pub fn set_unit_str(&mut self, scale: &str) -> &Self {
        self.scale = Scale::from_str(scale).unwrap();
        self
    }

    /// Retrieve angular frequencies
    pub fn w(&self) -> Array1<f64> {
        Array1::<f64>::from_shape_fn(self.vals.dim(), |i| self.vals[i] * 2.0 * PI)
    }

    /// Retrieve angular frequency at pt
    pub fn w_pt(&self, pt: usize) -> f64 {
        self.vals[pt] * 2.0 * PI
    }

    /// Retrieve wavelengths of Frequency
    pub fn wavelengths(&self, er: f64) -> Array1<f64> {
        Array1::<f64>::from_shape_fn(self.vals.dim(), |i| 3e8 / (self.vals[i] * er.sqrt()))
    }

    /// Retrieve wavelength of Frequency at pt
    pub fn wavelength(&self, er: f64, pt: usize) -> f64 {
        3e8 / (self.vals[pt] * er.sqrt())
    }
}

impl Default for Frequency {
    fn default() -> Self {
        Frequency {
            vals: array![1e9],
            scale: Scale::Giga,
            unit: Unit::Hz,
        }
    }
}

/// Builder design pattern for Frequency
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

                Frequency {
                    vals: out,
                    scale: self.scale,
                    unit: Unit::Hz,
                }
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

                Frequency {
                    vals: out,
                    scale: self.scale,
                    unit: Unit::Hz,
                }
            }
            (_, _) => Frequency {
                vals: self.vals,
                scale: self.scale,
                unit: Unit::Hz,
            },
        }
    }
}

#[cfg(test)]
mod frequency_tests {
    use super::*;
    use crate::util::{comp_array_f64, comp_f64};
    use float_cmp::F64Margin;

    #[test]
    fn test_frequency() {
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

        let freq = Frequency::new(vals.clone(), scale);
        comp_f64(
            &vals[0],
            &freq.freq(0),
            F64Margin::default(),
            "frequency::freq()",
            "0",
        );
        comp_f64(
            &vals_scaled[0],
            &&freq.freq_scaled(0),
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

        let freq = Frequency::new_scaled(vals_scaled.clone(), scale);
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

        let mut freq = Frequency::new(vals.clone(), scale);
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
        freq.set_unit(Scale::Mega);
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency::set_unit()",
        );

        let freq_vals_new_scaled = array![210e6, 340e6, 400e6, 510e6];
        freq.set_unit_str("kHz");
        comp_array_f64(
            freq_vals_new_scaled.view(),
            freq.freqs_scaled().view(),
            F64Margin::default(),
            "frequency::set_unit_str()",
        );

        let mut val = UnitVal::new(300e9, Scale::Giga, Unit::Hz);
        val.set_scale(Scale::Giga);
        val.set_unit(Unit::Hz);
        let freq_unitval = Frequency::from_unitval(&val);
        comp_array_f64(
            array![300e9].view(),
            freq_unitval.freqs().view(),
            F64Margin::default(),
            "frequency::from_unitval()",
        );
        assert_eq!(Scale::Giga, freq_unitval.scale);
        assert_eq!(Unit::Hz, freq_unitval.unit);
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
        assert_eq!(
            freq,
            Frequency {
                vals: vals.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .freqs_scaled(vals_scaled.clone(), scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_step(280e9, 500e9, 55e9)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_linear.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_step(280e9, 500e9, 50e9)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_linear2.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_step_scaled(280.0, 500.0, 55.0, scale)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_linear.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_npts(280e9, 500e9, 5, Sweep::Linear)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_linear.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_npts(280e9, 500e9, 5, Sweep::Log)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_log.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_npts_scaled(280.0, 500.0, 5, scale, Sweep::Linear)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_linear.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );

        let freq = FrequencyBuilder::new()
            .start_stop_npts_scaled(280.0, 500.0, 5, scale, Sweep::Log)
            .scale(scale)
            .build();
        assert_eq!(
            freq,
            Frequency {
                vals: vals_log.clone(),
                scale: scale,
                unit: Unit::Hz,
            }
        );
    }
}
