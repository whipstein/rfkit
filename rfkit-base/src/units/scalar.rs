use super::*;
use crate::num::{RealScalar, Scalar, ToComplex, ToReal};
use num_complex::Complex;
use serde::Serialize;
use std::{fmt, str::FromStr};

/// Encapsulation of a value with scale. Value is stored unscaled.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct ScalarUnitValue<T: Scalar> {
    pub(crate) val: T,
    pub(crate) scale: Scale,
    pub(crate) unit: Unit,
}

impl<T: Scalar + Scaleable> UnitValue<T> for ScalarUnitValue<T> {
    type Value = T;

    fn new(val: &T, scale: Scale, unit: Unit) -> Self {
        ScalarUnitValue {
            val: *val,
            scale,
            unit,
        }
    }

    fn new_scaled(val: &T, scale: Scale, unit: Unit) -> Self {
        ScalarUnitValue {
            val: val.unscale(scale),
            scale,
            unit,
        }
    }

    fn builder() -> UnitValueBuilder<T, Self> {
        UnitValueBuilder::new()
    }

    fn npts(&self) -> usize {
        1
    }

    fn val(&self) -> T {
        self.val
    }

    fn val_ref(&self) -> &T {
        &self.val
    }

    fn val_scaled(&self) -> T {
        self.val.scale(self.scale)
    }

    fn scale(&self) -> Scale {
        self.scale
    }

    fn unit(&self) -> Unit {
        self.unit
    }

    fn set_val(&mut self, val: &T) -> &Self {
        self.val = *val;
        self
    }

    fn set_val_pt(&mut self, val: T, _idx: usize) -> &Self {
        self.val = val;
        self
    }

    fn set_val_scaled(&mut self, val: &T) -> &Self {
        self.val = val.unscale(self.scale);
        self
    }

    fn set_val_scaled_pt(&mut self, val: T, _idx: usize) -> &Self {
        self.val = val.unscale(self.scale);
        self
    }

    fn set_scale(&mut self, scale: Scale) -> &Self {
        self.scale = scale;
        self
    }

    fn set_scale_str(&mut self, scale: &str) -> &Self {
        self.scale = Scale::from_str(scale).unwrap();
        self
    }

    fn set_unit(&mut self, unit: Unit) -> &Self {
        self.unit = unit;
        self
    }

    fn set_unit_str(&mut self, unit: &str) -> &Self {
        self.unit = Unit::from_str(unit).unwrap();
        self
    }

    fn zero_value() -> Self::Value {
        T::ZERO
    }
}

impl<T: RealScalar> Frequency<T> for ScalarUnitValue<T> {
    fn new_freq(
        val: &<ScalarUnitValue<T> as UnitValue<T>>::Value,
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
        val: &<ScalarUnitValue<T> as UnitValue<T>>::Value,
        scale: Scale,
    ) -> ScalarUnitValue<T> {
        ScalarUnitValue::builder()
            .val_scaled(val, scale)
            .unit(Unit::Hz)
            .build()
            .unwrap()
    }

    fn fpts(&self) -> usize {
        self.npts()
    }

    fn freq(&self) -> <ScalarUnitValue<T> as UnitValue<T>>::Value {
        self.val()
    }

    fn freq_scalar(&self, _idx: usize) -> ScalarUnitValue<T> {
        self.clone()
    }

    fn freq_pt(&self, _idx: usize) -> T {
        self.freq()
    }

    fn freq_ref(&self) -> &<ScalarUnitValue<T> as UnitValue<T>>::Value {
        self.val_ref()
    }

    fn freq_scaled(&self) -> <ScalarUnitValue<T> as UnitValue<T>>::Value {
        self.val_scaled()
    }

    fn w(&self) -> <ScalarUnitValue<T> as UnitValue<T>>::Value {
        self.val * 2.0 * std::f64::consts::PI
    }

    fn w_pt(&self, _idx: usize) -> T {
        self.w()
    }

    fn wavelength(&self, er: T) -> <ScalarUnitValue<T> as UnitValue<T>>::Value {
        T::from_f64(3e8) / (self.val * er.sqrt())
    }

    fn set_freq(&mut self, freq: &<ScalarUnitValue<T> as UnitValue<T>>::Value) {
        self.set_val(freq);
    }

    fn set_freq_pt(&mut self, freq: T, pt: usize) {
        self.set_val_pt(freq, pt);
    }

    fn set_freq_scaled(&mut self, freq: &<ScalarUnitValue<T> as UnitValue<T>>::Value) {
        self.set_val_scaled(freq);
    }

    fn set_freq_scaled_pt(&mut self, freq: T, pt: usize) {
        self.set_val_scaled_pt(freq, pt);
    }
}

impl<T: RealScalar> ToReal for ScalarUnitValue<T> {
    type ROutput = T;

    fn to_real(self) -> Self::ROutput {
        self.val
    }
}

impl<T: RealScalar + ToComplex<COutput = Complex<T>>> ToComplex for ScalarUnitValue<T> {
    type COutput = Complex<T>;

    fn to_complex(self) -> Self::COutput {
        Complex::new(self.val, T::ZERO)
    }
}

impl<T: RealScalar + ToReal> MapToReal<T> for ScalarUnitValue<T> {
    fn real_at(&self, val: &T, _idx: usize) -> T {
        *val
    }

    fn map_to_real<F>(&self, f: F) -> T
    where
        F: Fn(T) -> T,
    {
        f(self.val)
    }
}

impl<T: RealScalar + ToComplex<COutput = Complex<T>>> MapToComplex<T> for ScalarUnitValue<T> {
    fn complex_at(&self, val: &Complex<T>, _idx: usize) -> Complex<T> {
        *val
    }

    fn map_to_complex<F>(&self, f: F) -> Complex<T>
    where
        F: Fn(T) -> Complex<T>,
    {
        f(self.val)
    }
}

// ScalarUnitValue — passes itself
impl<T: RealScalar + Scaleable> MapScalar<T> for ScalarUnitValue<T> {
    type ROutput = T;
    type COutput = Complex<T>;

    fn map_scalar_to_real<F>(&self, f: F) -> T
    where
        F: Fn(&ScalarUnitValue<T>) -> T,
    {
        f(self)
    }

    fn map_scalar_to_complex<F>(&self, f: F) -> Complex<T>
    where
        F: Fn(&ScalarUnitValue<T>) -> Complex<T>,
    {
        f(self)
    }

    fn map_scalar_to_vec<R, F>(&self, mut f: F) -> Vec<R>
    where
        F: FnMut(&ScalarUnitValue<T>) -> R,
    {
        vec![f(self)]
    }
}

impl<T: Scalar + Scaleable + Copy> IntoIterator for ScalarUnitValue<T> {
    type Item = ScalarUnitValue<T>;
    type IntoIter = std::iter::Once<ScalarUnitValue<T>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self)
    }
}

impl<T: Scalar> fmt::Display for ScalarUnitValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}{}",
            self.val_scaled(),
            self.scale.to_str(),
            self.unit.to_str()
        )
    }
}

#[cfg(test)]
mod unit_scalarunitvalue_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin};

    #[test]
    fn test_scalarunitvalue() {
        let val: f64 = 10.34e-12;
        let val_scaled: f64 = 10.34;
        let scale = Scale::Pico;
        let unit = Unit::Farad;
        let mut unitval = ScalarUnitValue::new(&val, scale, unit);
        let val2: f64 = 4.74e-15;
        let scale2 = Scale::Femto;

        unitval
            .val()
            .assert_approx_eq(&val, NumMargin::default(), "val()", "");
        unitval.val_scaled().assert_approx_eq(
            &val_scaled,
            NumMargin::default(),
            "val_scaled()",
            "",
        );
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_val(&val2); // {val2, scale}
        unitval
            .val()
            .assert_approx_eq(&val2, NumMargin::default(), "set_val()", "");
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_val_scaled(&val_scaled); // {val, scale}
        unitval
            .val()
            .assert_approx_eq(&val, NumMargin::default(), "set_val_scaled()", "");
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_scale(scale2); // {val, scale2}
        unitval
            .val()
            .assert_approx_eq(&val, NumMargin::default(), "set_unit()", "");
        assert_eq!(&unitval.scale(), &scale2);
    }
}
