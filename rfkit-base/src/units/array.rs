use super::*;
use crate::num::{RealScalar, Scalar, ToComplex, ToReal};
use ndarray::Array1;
use num_complex::Complex;
use serde::Serialize;
use std::{fmt, str::FromStr};

/// Encapsulation of a value with scale. Value is stored unscaled.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ArrayUnitValue<T: Scalar> {
    pub(crate) val: Array1<T>,
    pub(crate) scale: Scale,
    pub(crate) unit: Unit,
}

impl<T: Scalar + Scaleable> UnitValue<T> for ArrayUnitValue<T> {
    type Value = Array1<T>;

    fn new(val: &Array1<T>, scale: Scale, unit: Unit) -> Self {
        ArrayUnitValue {
            val: val.clone(),
            scale,
            unit,
        }
    }

    fn new_scaled(val: &Array1<T>, scale: Scale, unit: Unit) -> Self {
        ArrayUnitValue {
            val: val.unscale(scale),
            scale,
            unit,
        }
    }

    fn builder() -> UnitValueBuilder<T, Self> {
        UnitValueBuilder::new()
    }

    fn npts(&self) -> usize {
        self.val.len()
    }

    fn val(&self) -> Array1<T> {
        self.val.clone()
    }

    fn val_ref(&self) -> &Array1<T> {
        &self.val
    }

    fn val_scaled(&self) -> Array1<T> {
        self.val.scale(self.scale)
    }

    fn scale(&self) -> Scale {
        self.scale
    }

    fn unit(&self) -> Unit {
        self.unit
    }

    fn set_val(&mut self, val: &Array1<T>) -> &Self {
        self.val = val.clone();
        self
    }

    fn set_val_pt(&mut self, val: T, idx: usize) -> &Self {
        self.val[idx] = val;
        self
    }

    fn set_val_scaled(&mut self, val: &Array1<T>) -> &Self {
        self.val = val.unscale(self.scale);
        self
    }

    fn set_val_scaled_pt(&mut self, val: T, idx: usize) -> &Self {
        self.val[idx] = val.unscale(self.scale);
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
        Array1::zeros(0)
    }
}

impl<T: RealScalar> Frequency<T> for ArrayUnitValue<T> {
    fn new_freq(
        val: &<ArrayUnitValue<T> as UnitValue<T>>::Value,
        scale: Scale,
    ) -> ArrayUnitValue<T> {
        ArrayUnitValue::builder()
            .val(val)
            .scale(scale)
            .unit(Unit::Hz)
            .build()
            .unwrap()
    }

    fn new_freq_scaled(
        val: &<ArrayUnitValue<T> as UnitValue<T>>::Value,
        scale: Scale,
    ) -> ArrayUnitValue<T> {
        ArrayUnitValue::builder()
            .val_scaled(val, scale)
            .unit(Unit::Hz)
            .build()
            .unwrap()
    }

    fn fpts(&self) -> usize {
        self.npts()
    }

    fn freq(&self) -> <ArrayUnitValue<T> as UnitValue<T>>::Value {
        self.val()
    }

    fn freq_scalar(&self, idx: usize) -> ScalarUnitValue<T> {
        ScalarUnitValue {
            val: self.freq_pt(idx),
            scale: self.scale,
            unit: self.unit,
        }
    }

    fn freq_pt(&self, idx: usize) -> T {
        self.freq()[idx]
    }

    fn freq_ref(&self) -> &<ArrayUnitValue<T> as UnitValue<T>>::Value {
        self.val_ref()
    }

    fn freq_scaled(&self) -> <ArrayUnitValue<T> as UnitValue<T>>::Value {
        self.val_scaled()
    }

    fn w(&self) -> <ArrayUnitValue<T> as UnitValue<T>>::Value {
        self.val.map(|&x| x * 2.0 * std::f64::consts::PI)
    }

    fn w_pt(&self, idx: usize) -> T {
        self.w()[idx]
    }

    fn wavelength(&self, er: T) -> <ArrayUnitValue<T> as UnitValue<T>>::Value {
        let er_sqrt = er.sqrt();
        self.val.map(|&x| T::from_f64(3e8) / (x * er_sqrt))
    }

    fn set_freq(&mut self, freq: &<ArrayUnitValue<T> as UnitValue<T>>::Value) {
        self.set_val(freq);
    }

    fn set_freq_pt(&mut self, freq: T, pt: usize) {
        self.set_val_pt(freq, pt);
    }

    fn set_freq_scaled(&mut self, freq: &<ArrayUnitValue<T> as UnitValue<T>>::Value) {
        self.set_val_scaled(freq);
    }

    fn set_freq_scaled_pt(&mut self, freq: T, pt: usize) {
        self.set_val_scaled_pt(freq, pt);
    }
}

impl<T: RealScalar> ToReal for ArrayUnitValue<T> {
    type ROutput = Array1<T>;

    fn to_real(self) -> Self::ROutput {
        self.val.map(|&x| x.to_real())
    }
}

impl<T: RealScalar + ToComplex<COutput = Complex<T>>> ToComplex for ArrayUnitValue<T> {
    type COutput = Array1<Complex<T>>;

    fn to_complex(self) -> Self::COutput {
        self.val.map(|&x| x.to_complex())
    }
}

impl<T: RealScalar + ToReal> MapToReal<T> for ArrayUnitValue<T> {
    fn real_at(&self, val: &Array1<T>, idx: usize) -> T {
        val[idx]
    }

    fn map_to_real<F>(&self, f: F) -> Array1<T>
    where
        F: Fn(T) -> T,
    {
        self.val.map(|&x| f(x))
    }
}

impl<T: RealScalar + ToComplex<COutput = Complex<T>>> MapToComplex<T> for ArrayUnitValue<T> {
    fn complex_at(&self, val: &Array1<Complex<T>>, idx: usize) -> Complex<T> {
        val[idx]
    }

    fn map_to_complex<F>(&self, f: F) -> Array1<Complex<T>>
    where
        F: Fn(T) -> Complex<T>,
    {
        self.val.map(|&x| f(x))
    }
}

// ArrayUnitValue — wraps each element
impl<T: RealScalar + Scaleable> MapScalar<T> for ArrayUnitValue<T> {
    type ROutput = Array1<T>;
    type COutput = Array1<Complex<T>>;

    fn map_scalar_to_real<F>(&self, f: F) -> Array1<T>
    where
        F: Fn(&ScalarUnitValue<T>) -> T,
    {
        let scale = self.scale;
        let unit = self.unit;
        self.val.map(|&v| {
            f(&ScalarUnitValue {
                val: v,
                scale,
                unit,
            })
        })
    }

    fn map_scalar_to_complex<F>(&self, f: F) -> Array1<Complex<T>>
    where
        F: Fn(&ScalarUnitValue<T>) -> Complex<T>,
    {
        let scale = self.scale;
        let unit = self.unit;
        self.val.map(|&v| {
            f(&ScalarUnitValue {
                val: v,
                scale,
                unit,
            })
        })
    }

    fn map_scalar_to_vec<R, F>(&self, mut f: F) -> Vec<R>
    where
        F: FnMut(&ScalarUnitValue<T>) -> R,
    {
        self.val
            .iter()
            .map(|&v| {
                f(&ScalarUnitValue {
                    val: v,
                    scale: self.scale,
                    unit: self.unit,
                })
            })
            .collect()
    }
}

impl<T: Scalar + Scaleable + Copy> IntoIterator for ArrayUnitValue<T> {
    type Item = ScalarUnitValue<T>;
    type IntoIter = std::vec::IntoIter<ScalarUnitValue<T>>;

    fn into_iter(self) -> Self::IntoIter {
        let scale = self.scale;
        let unit = self.unit;
        self.val
            .iter()
            .map(move |&v| ScalarUnitValue {
                val: v,
                scale,
                unit,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a, T: Scalar + Scaleable + Copy> IntoIterator for &'a ArrayUnitValue<T> {
    type Item = ScalarUnitValue<T>;
    type IntoIter = std::vec::IntoIter<ScalarUnitValue<T>>;

    fn into_iter(self) -> Self::IntoIter {
        let scale = self.scale;
        let unit = self.unit;
        self.val
            .iter()
            .map(move |&v| ScalarUnitValue {
                val: v,
                scale,
                unit,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<T: Scalar> fmt::Display for ArrayUnitValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.npts() == 0 {
            return write!(f, "[]");
        }

        writeln!(f, "[")?;
        for i in 0..self.npts() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{} {}{}",
                self.val_scaled()[i],
                self.scale.to_str(),
                self.unit.to_str()
            )?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod unit_arrayunitvalue_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin};
    use ndarray::array;

    #[test]
    fn test_arrayunitvalue() {
        let val = array![10.34e-12, 10.65e-12];
        let val_scaled = array![10.34, 10.65];
        let scale = Scale::Pico;
        let unit = Unit::Farad;
        let mut unitval = ArrayUnitValue::new(&val, scale, unit);
        let val2 = array![4.74e-15, 6.45e-15];
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
