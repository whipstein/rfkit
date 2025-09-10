use crate::scale::Scale;
use serde::Serialize;
use std::fmt;
use std::str::FromStr;

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

pub trait Unitized {
    fn val_scaled(&self) -> f64;
    fn unitval(&self) -> UnitVal;
    fn scale(&self) -> Scale;
    fn unit(&self) -> Unit;
    fn set_unitval(&mut self, val: UnitVal);
    fn set_val_scaled(&mut self, val: f64);
    fn set_scale(&mut self, scale: Scale);
    fn set_unit(&mut self, unit: Unit);
}

/// Encapsulation of a value with scale. Value is stored unscaled.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct UnitVal {
    val: f64,
    scale: Scale,
    unit: Unit,
}

impl UnitVal {
    pub fn new(val: f64, scale: Scale, unit: Unit) -> Self {
        UnitVal {
            val,
            scale: scale,
            unit: unit,
        }
    }

    pub fn new_scaled(val: f64, scale: Scale, unit: Unit) -> Self {
        UnitVal {
            val: scale.unscale(val),
            scale: scale,
            unit: unit,
        }
    }

    /// Retrieve value unscaled
    pub fn val(&self) -> f64 {
        self.val
    }

    /// Retrieve value in scaled scale
    pub fn val_scaled(&self) -> f64 {
        self.scale.scale(self.val)
    }

    /// Retrieve scale
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// Retrieve unit
    pub fn unit(&self) -> Unit {
        self.unit
    }

    /// Set value unscaled
    pub fn set_val(&mut self, val: f64) -> &Self {
        self.val = val;
        self
    }

    /// Set value in scaled scale
    pub fn set_val_scaled(&mut self, val: f64) -> &Self {
        self.val = self.scale.unscale(val);
        self
    }

    /// Set scale
    pub fn set_scale(&mut self, scale: Scale) -> &Self {
        self.scale = scale;
        self
    }

    /// Set unit
    pub fn set_unit(&mut self, unit: Unit) -> &Self {
        self.unit = unit;
        self
    }
}

impl Default for UnitVal {
    fn default() -> Self {
        UnitVal {
            val: 0.0,
            scale: Scale::Base,
            unit: Unit::None,
        }
    }
}

/// Builder design pattern for UnitVal.
///
/// ## Example
/// ```
/// use rfkit_base_ndarray::unit::{Unit, UnitValBuilder};
/// use rfkit_base_ndarray::scale::Scale;
///
/// let unitval = UnitValBuilder::new().val_scaled(1.2, Unit::Pico).build();
/// ```
#[derive(Default)]
pub struct UnitValBuilder {
    val: f64,
    scale: Scale,
    unit: Unit,
}

impl UnitValBuilder {
    pub fn new() -> Self {
        UnitValBuilder {
            val: 0.0,
            scale: Scale::Base,
            unit: Unit::None,
        }
    }

    pub fn val(mut self, val: f64) -> Self {
        self.val = val;
        self
    }

    pub fn val_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.val = scale.unscale(val);
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

    pub fn build(self) -> UnitVal {
        UnitVal {
            val: self.val,
            scale: self.scale,
            unit: self.unit,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::comp_f64;
    use float_cmp::F64Margin;

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
    fn test_unitval() {
        let val: f64 = 10.34e-12;
        let val_scaled: f64 = 10.34;
        let scale = Scale::Pico;
        let unit = Unit::Farad;
        let mut unitval = UnitVal::new(val, scale, unit);
        let val2: f64 = 4.74e-15;
        let scale2 = Scale::Femto;

        comp_f64(&unitval.val(), &val, F64Margin::default(), "val()", "");
        comp_f64(
            &unitval.val_scaled(),
            &val_scaled,
            F64Margin::default(),
            "val_scaled()",
            "",
        );
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_val(val2); // {val2, scale}
        comp_f64(&unitval.val(), &val2, F64Margin::default(), "set_val()", "");
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_val_scaled(val_scaled); // {val, scale}
        comp_f64(
            &unitval.val(),
            &val,
            F64Margin::default(),
            "set_val_scaled()",
            "",
        );
        assert_eq!(&unitval.scale(), &scale);

        unitval.set_scale(scale2); // {val, scale2}
        comp_f64(&unitval.val(), &val, F64Margin::default(), "set_unit()", "");
        assert_eq!(&unitval.scale(), &scale2);
    }

    #[test]
    fn test_unitvalbuilder() {
        let val: f64 = 10.34e-12;
        let val_scaled: f64 = 10.34;
        let scale = Scale::Pico;
        let unit = Unit::Second;

        let unitval = UnitValBuilder::new()
            .val(val)
            .scale(scale)
            .unit(unit)
            .build();
        assert_eq!(
            unitval,
            UnitVal {
                val: val,
                scale: scale,
                unit: unit,
            }
        );

        let unitval2 = UnitValBuilder::new()
            .val_scaled(val_scaled, scale)
            .unit(unit)
            .build();
        assert_eq!(
            unitval2,
            UnitVal {
                val: val,
                scale: scale,
                unit: unit,
            }
        );

        let val: f64 = 1.34e9;
        let val_scaled: f64 = 1.34;
        let scale = Scale::Giga;
        let unit = Unit::Hz;

        let unitval = UnitValBuilder::new()
            .val(val)
            .scale(scale)
            .unit(unit)
            .build();
        assert_eq!(
            unitval,
            UnitVal {
                val: val,
                scale: scale,
                unit: unit,
            }
        );

        let unitval2 = UnitValBuilder::new()
            .val_scaled(val_scaled, scale)
            .unit(unit)
            .build();
        assert_eq!(
            unitval2,
            UnitVal {
                val: val,
                scale: scale,
                unit: unit,
            }
        );
    }
}
