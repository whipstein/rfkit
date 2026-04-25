use super::*;
use crate::{
    num::{RealScalar, Scalar, ScalarConst},
    util::{ApproxEq, NumMargin},
};
use ndarray::{Array1, Axis, array, concatenate};
use std::str::FromStr;

/// Builder design pattern for UnitValue.
///
/// ## Example
/// ```
/// use ndarray::array;
/// use rfkit_base::prelude::*;
///
/// let sunitval: ScalarUnitValue<f64> = ScalarUnitValue::builder()
///     .val_scaled(&1.2, Scale::Pico)
///     .build()
///     .unwrap();
///
/// let values = array![1.2, 1.5];
/// let aunitval: ArrayUnitValue<f64> = ArrayUnitValue::builder()
///     .val_scaled(&values, Scale::Pico)
///     .build()
///     .unwrap();
/// ```
pub struct UnitValueBuilder<T: Scalar, U: UnitValue<T>> {
    pub(crate) val: Option<U::Value>,
    pub(crate) scale: Scale,
    pub(crate) unit: Unit,
}

impl<T: Scalar + Scaleable, U: UnitValue<T>> UnitValueBuilder<T, U> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn val(mut self, val: &U::Value) -> Self {
        self.val = Some(val.clone());
        self
    }

    pub fn val_scaled(mut self, val: &U::Value, scale: Scale) -> Self {
        self.val = Some(val.unscale(scale));
        self.scale = scale;
        self
    }

    pub fn scale(mut self, scale: Scale) -> Self {
        self.scale = scale;
        self
    }

    pub fn scale_str(mut self, scale: &str) -> Self {
        self.scale = Scale::from_str(scale).unwrap();
        self
    }

    pub fn unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }

    pub fn unit_str(mut self, unit: &str) -> Self {
        self.unit = Unit::from_str(unit).unwrap();
        self
    }

    pub fn build(self) -> Result<U, String> {
        let val = self.val.ok_or("value is required")?;
        Ok(U::new(&val, self.scale, self.unit))
    }
}

impl<T: Scalar, U: UnitValue<T>> Default for UnitValueBuilder<T, U> {
    fn default() -> Self {
        UnitValueBuilder {
            val: None,
            scale: Scale::Base,
            unit: Unit::None,
        }
    }
}

impl<T: RealScalar + Scaleable> FrequencyBuilder<T, ScalarUnitValue<T>>
    for UnitValueBuilder<T, ScalarUnitValue<T>>
{
    fn freq(mut self, freq: &T) -> Self {
        self.val = Some(*freq);
        self
    }

    fn freq_scaled(mut self, freq: &T, scale: Scale) -> Self {
        self.val = Some(freq.unscale(scale));
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

impl<T: RealScalar> FrequencyBuilder<T, ArrayUnitValue<T>>
    for UnitValueBuilder<T, ArrayUnitValue<T>>
where
    <T as ApproxEq>::Compare: ScalarConst,
{
    fn freq(mut self, freq: &Array1<T>) -> Self {
        self.val = Some(freq.clone());
        self.unit = Unit::Hz;
        self
    }

    fn freq_scaled(mut self, freq: &Array1<T>, scale: Scale) -> Self {
        self.val = Some(freq.unscale(scale));
        self.scale = scale;
        self.unit = Unit::Hz;
        self
    }

    /// Provide frequency from start, stop and step in Hz
    fn start_stop_step(self, start: T, stop: T, step: T) -> Self {
        let npts: usize = ((stop - start) / step).floor().to_usize().unwrap() + 1;
        let mut out = Array1::<T>::zeros(npts);

        let mut last_val = T::ZERO;
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

        let mut last_val = T::ZERO;
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

        let mut last_val = T::ZERO;
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
                    *pt = T::from_f64(10.0).powf(
                        (stop.log10() - start.log10()) / ((npts - 1) * i) as f64 + start.log10(),
                    );
                    last_val = T::from_f64(10.0).powf(
                        (stop.log10() - start.log10()) / ((npts - 1) * i) as f64 + start.log10(),
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

        let mut last_val = T::ZERO;
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
                    *pt = T::from_f64(10.0).powf(
                        (stop.log10() - start.log10()) / ((npts - 1) * i) as f64 + start.log10(),
                    );
                    last_val = T::from_f64(10.0).powf(
                        (stop.log10() - start.log10()) / ((npts - 1) * i) as f64 + start.log10(),
                    );
                }
            }
        }
        out[npts - 1] = stop;

        self.val_scaled(&out, scale)
    }
}

#[cfg(test)]
mod units_builder_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_unitvalbuilder_scalar() {
        let val: f64 = 10.34e-12;
        let val_scaled: f64 = 10.34;
        let scale = Scale::Pico;
        let unit = Unit::Second;

        let unitval = ScalarUnitValue::builder()
            .val(&val)
            .scale(scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(
            unitval,
            ScalarUnitValue {
                val: val,
                scale: scale,
                unit: unit,
            }
        );

        let unitval2 = ScalarUnitValue::builder()
            .val_scaled(&val_scaled, scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(
            unitval2,
            ScalarUnitValue {
                val: val,
                scale: scale,
                unit: unit,
            }
        );

        let val: f64 = 1.34e9;
        let val_scaled: f64 = 1.34;
        let scale = Scale::Giga;
        let unit = Unit::Hz;

        let unitval = ScalarUnitValue::builder()
            .val(&val)
            .scale(scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(
            unitval,
            ScalarUnitValue {
                val: val,
                scale: scale,
                unit: unit,
            }
        );

        let unitval2 = ScalarUnitValue::builder()
            .val_scaled(&val_scaled, scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(
            unitval2,
            ScalarUnitValue {
                val: val,
                scale: scale,
                unit: unit,
            }
        );
    }

    #[test]
    fn test_unitvalbuilder_array() {
        let val = array![10.34e-12, 10.65e-12];
        let val_scaled = array![10.34, 10.65];
        let scale = Scale::Pico;
        let unit = Unit::Second;
        let exemplar = ArrayUnitValue {
            val: val.clone(),
            scale: scale,
            unit: unit,
        };

        let unitval = ArrayUnitValue::builder()
            .val(&val)
            .scale(scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(unitval, exemplar);

        let unitval2 = ArrayUnitValue::builder()
            .val_scaled(&val_scaled, scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(unitval2, exemplar);

        let val = array![1.34e9, 5.51e9];
        let val_scaled = array![1.34, 5.51];
        let scale = Scale::Giga;
        let unit = Unit::Hz;
        let exemplar = ArrayUnitValue {
            val: val.clone(),
            scale: scale,
            unit: unit,
        };

        let unitval = ArrayUnitValue::builder()
            .val(&val)
            .scale(scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(unitval, exemplar);

        let unitval2 = ArrayUnitValue::builder()
            .val_scaled(&val_scaled, scale)
            .unit(unit)
            .build()
            .unwrap();
        assert_eq!(unitval2, exemplar);
    }
}
