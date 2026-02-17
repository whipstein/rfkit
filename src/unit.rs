use crate::{num::RealScalar, scale::Scale};
use ndarray::Array1;
use serde::Serialize;
use std::{fmt, str::FromStr};

/// Descriptor of sweep type
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Sweep {
    #[default]
    Linear,
    Log,
}

/// Descriptor of unit
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Unit {
    #[default]
    None, // No Unit
    Hz,     // Frequency in Hz
    Degree, // Number in degree
    Radian, // Number in radians
    Lambda, // Length in wavelengths
    Second, // Length in seconds
    Meter,  // Length in meters
    Inch,   // Length in inches
    Farad,  // Capacitance in farads
    Henry,  // Inductance in henries
    Ohm,    // Resistance in ohms
    Sieman, // Conductance in siemans
    Neper,  // Legacy unit of nepers
}

impl Unit {
    pub fn to_long_string(&self) -> String {
        match self {
            Unit::None => "".to_string(),
            Unit::Hz => "hertz".to_string(),
            Unit::Degree => "degree".to_string(),
            Unit::Radian => "radian".to_string(),
            Unit::Lambda => "lambda".to_string(),
            Unit::Second => "second".to_string(),
            Unit::Meter => "meter".to_string(),
            Unit::Inch => "inch".to_string(),
            Unit::Farad => "farad".to_string(),
            Unit::Henry => "henry".to_string(),
            Unit::Ohm => "ohm".to_string(),
            Unit::Sieman => "sieman".to_string(),
            Unit::Neper => "neper".to_string(),
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            Unit::None => "",
            Unit::Hz => "Hz",
            Unit::Degree => "°",
            Unit::Radian => "rad",
            Unit::Lambda => "λ",
            Unit::Second => "s",
            Unit::Meter => "m",
            Unit::Inch => "in",
            Unit::Farad => "F",
            Unit::Henry => "H",
            Unit::Ohm => "Ω",
            Unit::Sieman => "S",
            Unit::Neper => "Np",
        }
    }
}

impl FromStr for Unit {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "HZ" | "Hz" | "hz" => Ok(Unit::Hz),
            "Degree" | "degree" | "deg" | "°" => Ok(Unit::Degree),
            "Radian" | "radian" | "rad" => Ok(Unit::Radian),
            "Lambda" | "lambda" | "λ" => Ok(Unit::Lambda),
            "Second" | "second" | "sec" | "s" => Ok(Unit::Second),
            "Meter" | "meter" | "m" => Ok(Unit::Meter),
            "Inch" | "inch" | "in" => Ok(Unit::Inch),
            "Farad" | "farad" | "F" => Ok(Unit::Farad),
            "Henry" | "henry" | "H" => Ok(Unit::Henry),
            "Ohm" | "ohm" | "Ω" => Ok(Unit::Ohm),
            "Sieman" | "sieman" | "S" => Ok(Unit::Sieman),
            "Neper" | "neper" | "Np" => Ok(Unit::Neper),
            _ => Ok(Unit::None),
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

// pub trait Unitized<T>
// where
//     T: RealScalar,
// {
//     // fn val_scaled(&self) -> T;
//     // fn unitval(&self) -> Self;
//     // fn scale(&self) -> Scale;
//     // fn unit(&self) -> Unit;
//     // fn set_unitval(&mut self, val: Self);
//     // fn set_val_scaled(&mut self, val: T);
//     // fn set_scale(&mut self, scale: Scale);
//     // fn set_unit(&mut self, unit: Unit);
// }

/// Trait for types that can be used as values in UnitValue.
/// This allows both scalar values (f64, TwoFloat) and array values (Array1<f64>, Array1<TwoFloat>).
pub trait UnitValue: Clone + Default {
    type Scalar: RealScalar;
    type Value: Clone;

    fn new(val: &Self::Value, scale: Scale, unit: Unit) -> Self;
    fn new_scaled(val: &Self::Value, scale: Scale, unit: Unit) -> Self;
    fn npts(&self) -> usize;
    fn val_ref(&self) -> &Self::Value;
    fn val_scaled(&self) -> Self::Value;

    fn zero_value() -> Self::Value;
    /// Apply scaling (divide by multiplier)
    fn scale_val(val: &Self::Value, multiplier: Self::Scalar) -> Self::Value;
    /// Apply unscaling (multiply by multiplier)
    fn unscale_val(val: &Self::Value, multiplier: Self::Scalar) -> Self::Value;
}

impl<T: RealScalar> UnitValue for ScalarUnitValue<T> {
    type Scalar = T;
    type Value = T;

    fn new(val: &T, scale: Scale, unit: Unit) -> Self {
        Self {
            val: *val,
            scale,
            unit,
        }
    }

    fn new_scaled(val: &T, scale: Scale, unit: Unit) -> Self {
        Self {
            val: scale.unscale(*val),
            scale,
            unit,
        }
    }

    fn npts(&self) -> usize {
        1
    }

    fn val_ref(&self) -> &T {
        &self.val
    }

    fn val_scaled(&self) -> T {
        self.scale.scale(self.val)
    }

    fn zero_value() -> Self::Value {
        Self::Value::zero()
    }

    fn scale_val(val: &Self::Value, multiplier: T) -> Self::Value {
        *val / multiplier
    }

    fn unscale_val(val: &Self::Value, multiplier: T) -> Self::Value {
        *val * multiplier
    }
}

impl<T: RealScalar> UnitValue for ArrayUnitValue<T> {
    type Scalar = T;
    type Value = Array1<T>;

    fn new(val: &Array1<T>, scale: Scale, unit: Unit) -> Self {
        Self {
            val: val.clone(),
            scale,
            unit,
        }
    }

    fn new_scaled(val: &Array1<T>, scale: Scale, unit: Unit) -> Self {
        Self {
            val: scale.unscale_array(val),
            scale,
            unit,
        }
    }

    fn npts(&self) -> usize {
        self.val.len()
    }

    fn val_ref(&self) -> &Array1<T> {
        &self.val
    }

    fn val_scaled(&self) -> Array1<T> {
        self.scale.scale_array(&self.val)
    }

    fn zero_value() -> Self::Value {
        Array1::zeros(0)
    }
    fn scale_val(val: &Self::Value, multiplier: T) -> Self::Value {
        val.map(|&x| x / multiplier)
    }

    fn unscale_val(val: &Self::Value, multiplier: T) -> Self::Value {
        val.map(|&x| x * multiplier)
    }
}

/// Encapsulation of a value with scale. Value is stored unscaled.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct ScalarUnitValue<T: RealScalar> {
    pub(crate) val: T,
    pub(crate) scale: Scale,
    pub(crate) unit: Unit,
}

impl<T: RealScalar> ScalarUnitValue<T> {
    pub fn new(val: T, scale: Scale, unit: Unit) -> Self {
        ScalarUnitValue { val, scale, unit }
    }

    pub fn new_scaled(val: T, scale: Scale, unit: Unit) -> Self {
        ScalarUnitValue {
            val: scale.unscale(val),
            scale,
            unit,
        }
    }

    pub fn builder() -> UnitValBuilder<Self> {
        UnitValBuilder::new()
    }

    /// Retrieve value unscaled
    pub fn val(&self) -> T {
        self.val
    }

    // /// Retrieve value unscaled (as reference)
    // pub fn val_ref(&self) -> &T {
    //     &self.val
    // }

    // /// Retrieve value in scaled scale
    // pub fn val_scaled(&self) -> T {
    //     self.val.scale(self.scale.multiplier())
    // }

    /// Retrieve scale
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// Retrieve unit
    pub fn unit(&self) -> Unit {
        self.unit
    }

    /// Set value unscaled
    pub fn set_val(&self, val: T) -> Self {
        let mut out = self.clone();
        out.val = val;
        out
    }

    /// Set value unscaled
    pub fn set_val_inplace(&mut self, val: T) {
        self.val = val;
    }

    pub fn set_val_pt_inplace(&mut self, val: T) {
        self.set_val_inplace(val);
    }

    /// Set value in scaled scale
    pub fn set_val_scaled(&self, val: T) -> Self {
        let mut out = self.clone();
        out.val = self.scale.unscale(val);
        out
    }

    /// Set value in scaled scale
    pub fn set_val_scaled_inplace(&mut self, val: T) {
        self.val = self.scale.unscale(val);
    }

    pub fn set_val_pt_scaled_inplace(&mut self, val: T) {
        self.set_val_scaled_inplace(val);
    }

    /// Set scale
    pub fn set_scale(&mut self, scale: Scale) {
        self.scale = scale;
    }

    /// Set scale
    pub fn set_scale_str(&mut self, scale: &str) {
        self.scale = Scale::from_str(scale).unwrap();
    }

    /// Set unit
    pub fn set_unit(&mut self, unit: Unit) {
        self.unit = unit;
    }

    /// Set scale
    pub fn set_unit_str(&mut self, unit: &str) {
        self.unit = Unit::from_str(unit).unwrap();
    }
}

// impl<T: RealScalar> Default for ScalarUnitValue<T> {
//     fn default() -> Self {
//         ScalarUnitValue {
//             val: T::zero(),
//             scale: Scale::Base,
//             unit: Unit::None,
//         }
//     }
// }

/// Encapsulation of a value with scale. Value is stored unscaled.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ArrayUnitValue<T: RealScalar> {
    pub(crate) val: Array1<T>,
    pub(crate) scale: Scale,
    pub(crate) unit: Unit,
}

impl<T: RealScalar> ArrayUnitValue<T> {
    pub fn new(val: &Array1<T>, scale: Scale, unit: Unit) -> Self {
        ArrayUnitValue {
            val: val.clone(),
            scale,
            unit,
        }
    }

    pub fn new_scaled(val: &Array1<T>, scale: Scale, unit: Unit) -> Self {
        ArrayUnitValue {
            val: scale.unscale_array(val),
            scale,
            unit,
        }
    }

    pub fn builder() -> UnitValBuilder<Self> {
        UnitValBuilder::new()
    }

    /// Retrieve value unscaled
    pub fn val(&self) -> Array1<T> {
        self.val.clone()
    }

    // /// Retrieve value unscaled (as reference)
    // pub fn val_ref(&self) -> &Array1<T> {
    //     &self.val
    // }

    // /// Retrieve value in scaled scale
    // pub fn val_scaled(&self) -> Array1<T> {
    //     self.val.scale(self.scale.multiplier())
    // }

    /// Retrieve scale
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// Retrieve unit
    pub fn unit(&self) -> Unit {
        self.unit
    }

    /// Set value unscaled
    pub fn set_val(&self, val: &Array1<T>) -> Self {
        let mut out = self.clone();
        out.val = val.clone();
        out
    }

    /// Set value unscaled
    pub fn set_val_inplace(&mut self, val: &Array1<T>) {
        self.val = val.clone();
    }

    pub fn set_val_pt_inplace(&mut self, val: T, pt: usize) {
        self.val[pt] = val;
    }

    /// Set value in scaled scale
    pub fn set_val_scaled(&self, val: &Array1<T>) -> Self {
        let mut out = self.clone();
        out.val = self.scale.unscale_array(val);
        out
    }

    /// Set value in scaled scale
    pub fn set_val_scaled_inplace(&mut self, val: &Array1<T>) {
        self.val = self.scale.unscale_array(val);
    }

    pub fn set_val_pt_scaled_inplace(&mut self, val: T, pt: usize) {
        self.val[pt] = self.scale.unscale(val);
    }

    /// Set scale
    pub fn set_scale(&mut self, scale: Scale) {
        self.scale = scale;
    }

    /// Set scale
    pub fn set_scale_str(&mut self, scale: &str) {
        self.scale = Scale::from_str(scale).unwrap();
    }

    /// Set unit
    pub fn set_unit(&mut self, unit: Unit) {
        self.unit = unit;
    }

    /// Set scale
    pub fn set_unit_str(&mut self, unit: &str) {
        self.unit = Unit::from_str(unit).unwrap();
    }
}

// impl<T: RealScalar> Default for ArrayUnitValue<T> {
//     fn default() -> Self {
//         ArrayUnitValue {
//             val: Array1::zeros(0),
//             scale: Scale::Base,
//             unit: Unit::None,
//         }
//     }
// }

/// Builder design pattern for UnitValue.
///
/// ## Example
/// ```
/// use rfkit::prelude::*;
///
/// let sunitval = ScalarUnitValue::builder().val_scaled(1.2, Scale::Pico).build().unwrap();
/// let aunitval = ArrayUnitValue::builder().val_scaled(array![1.2, 1.5], Scale::Pico).build().unwrap();
/// ```
pub struct UnitValBuilder<T: UnitValue> {
    pub(crate) val: Option<T::Value>,
    pub(crate) scale: Scale,
    pub(crate) unit: Unit,
}

impl<T: UnitValue> UnitValBuilder<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn val(mut self, val: &T::Value) -> Self {
        self.val = Some(val.clone());
        self
    }

    pub fn val_scaled(mut self, val: &T::Value, scale: Scale) -> Self {
        self.val = Some(T::unscale_val(val, scale.multiplier()));
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

    pub fn build(self) -> Result<T, String> {
        let val = self.val.ok_or("value is required")?;
        Ok(T::new(&val, self.scale, self.unit))
    }
}

impl<T: UnitValue> Default for UnitValBuilder<T> {
    fn default() -> Self {
        UnitValBuilder {
            val: None,
            scale: Scale::Base,
            unit: Unit::None,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin};
    use ndarray::array;
    use std::str::FromStr;

    #[test]
    fn test_parse_unit() {
        assert_eq!(Unit::None, Unit::from_str("").unwrap());
        assert_eq!(Unit::None, Unit::from_str("x").unwrap());
        assert_eq!(Unit::Hz, Unit::from_str("HZ").unwrap());
        assert_eq!(Unit::Hz, Unit::from_str("Hz").unwrap());
        assert_eq!(Unit::Hz, Unit::from_str("hz").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("Degree").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("degree").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("deg").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("°").unwrap());
        assert_eq!(Unit::Radian, Unit::from_str("Radian").unwrap());
        assert_eq!(Unit::Radian, Unit::from_str("radian").unwrap());
        assert_eq!(Unit::Radian, Unit::from_str("rad").unwrap());
        assert_eq!(Unit::Lambda, Unit::from_str("Lambda").unwrap());
        assert_eq!(Unit::Lambda, Unit::from_str("lambda").unwrap());
        assert_eq!(Unit::Lambda, Unit::from_str("λ").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("Second").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("second").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("sec").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("s").unwrap());
        assert_eq!(Unit::Meter, Unit::from_str("Meter").unwrap());
        assert_eq!(Unit::Meter, Unit::from_str("meter").unwrap());
        assert_eq!(Unit::Meter, Unit::from_str("m").unwrap());
        assert_eq!(Unit::Inch, Unit::from_str("Inch").unwrap());
        assert_eq!(Unit::Inch, Unit::from_str("inch").unwrap());
        assert_eq!(Unit::Inch, Unit::from_str("in").unwrap());
        assert_eq!(Unit::Farad, Unit::from_str("Farad").unwrap());
        assert_eq!(Unit::Farad, Unit::from_str("farad").unwrap());
        assert_eq!(Unit::Farad, Unit::from_str("F").unwrap());
        assert_eq!(Unit::Henry, Unit::from_str("Henry").unwrap());
        assert_eq!(Unit::Henry, Unit::from_str("henry").unwrap());
        assert_eq!(Unit::Henry, Unit::from_str("H").unwrap());
        assert_eq!(Unit::Ohm, Unit::from_str("Ohm").unwrap());
        assert_eq!(Unit::Ohm, Unit::from_str("ohm").unwrap());
        assert_eq!(Unit::Ohm, Unit::from_str("Ω").unwrap());
        assert_eq!(Unit::Sieman, Unit::from_str("Sieman").unwrap());
        assert_eq!(Unit::Sieman, Unit::from_str("sieman").unwrap());
        assert_eq!(Unit::Sieman, Unit::from_str("S").unwrap());
        assert_eq!(Unit::Neper, Unit::from_str("Neper").unwrap());
        assert_eq!(Unit::Neper, Unit::from_str("neper").unwrap());
        assert_eq!(Unit::Neper, Unit::from_str("Np").unwrap());
    }

    #[test]
    fn test_unit_from_str() {
        let hz = ["HZ", "Hz", "hz"];
        let deg = ["Degree", "degree", "deg", "°"];
        let rad = ["Radian", "radian", "rad"];
        let lambda = ["Lambda", "lambda", "λ"];
        let sec = ["Second", "second", "sec", "s"];
        let meter = ["Meter", "meter", "m"];
        let inch = ["Inch", "inch", "in"];
        let farad = ["Farad", "farad", "F"];
        let henry = ["Henry", "henry", "H"];
        let ohm = ["Ohm", "ohm", "Ω"];
        let sieman = ["Sieman", "sieman", "S"];
        let neper = ["Neper", "neper", "Np"];
        let nada = ["", "google", ".sfwe"];

        for mult in hz.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Hz);
        }

        for mult in deg.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Degree);
        }

        for mult in rad.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Radian);
        }

        for mult in lambda.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Lambda);
        }

        for mult in sec.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Second);
        }

        for mult in meter.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Meter);
        }

        for mult in inch.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Inch);
        }

        for mult in farad.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Farad);
        }

        for mult in henry.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Henry);
        }

        for mult in ohm.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Ohm);
        }

        for mult in sieman.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Sieman);
        }

        for mult in neper.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Neper);
        }

        for mult in nada.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::None);
        }
    }

    #[test]
    fn test_scalarunitvalue() {
        let val: f64 = 10.34e-12;
        let val_scaled: f64 = 10.34;
        let scale = Scale::Pico;
        let unit = Unit::Farad;
        let mut unitval = ScalarUnitValue::new(val, scale, unit);
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

        unitval.set_val_inplace(val2); // {val2, scale}
        unitval
            .val()
            .assert_approx_eq(&val2, NumMargin::default(), "set_val()", "");
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_val_scaled_inplace(val_scaled); // {val, scale}
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

        unitval.set_val_inplace(&val2); // {val2, scale}
        unitval
            .val()
            .assert_approx_eq(&val2, NumMargin::default(), "set_val()", "");
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_val_scaled_inplace(&val_scaled); // {val, scale}
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
