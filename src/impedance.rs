use crate::math::*;
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, UnitValBuilder};
use num::complex::{Complex64, c64};
use serde::Serialize;
use std::f64::consts::PI;
use std::fmt;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct Impedance {
    y: Complex64,
    z: Complex64,
    g: Complex64,
    r: UnitVal,
    c: UnitVal,
    z0: f64,
    freq: UnitVal,
}

impl Impedance {
    pub fn new_from_gamma(gamma: Complex64, z0: f64, freq: &UnitVal) -> Self {
        let mut out = Impedance::default();
        out.set_gamma(gamma);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_y(y: Complex64, z0: f64, freq: &UnitVal) -> Self {
        let mut out = Impedance::default();
        out.set_y(y);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_z(z: Complex64, z0: f64, freq: &UnitVal) -> Self {
        let mut out = Impedance::default();
        out.set_z(z);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_rc(r: &UnitVal, c: &UnitVal, z0: f64, freq: &UnitVal) -> Self {
        let mut out = Impedance::default();
        out.set_r(r);
        out.set_c(c);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn gamma(&self) -> Complex64 {
        self.g
    }

    pub fn y(&self) -> Complex64 {
        self.y
    }

    pub fn z(&self) -> Complex64 {
        self.z
    }

    pub fn r(&self) -> UnitVal {
        self.r
    }

    pub fn c(&self) -> UnitVal {
        self.c
    }

    pub fn z0(&self) -> f64 {
        self.z0
    }

    pub fn freq(&self) -> UnitVal {
        self.freq
    }

    pub fn set_gamma(&mut self, g: Complex64) -> &Self {
        self.g = g;
        self.z = gamma_to_z(self.g, self.z0);
        self.y = z_to_y(self.z);
        let (r_val, c_val) = z_to_rc(self.z, &self.freq);
        self.r.set_val(r_val);
        self.c.set_val(c_val);
        self
    }

    pub fn set_y(&mut self, y: Complex64) -> &Self {
        self.y = y;
        self.z = y_to_z(self.y);
        self.g = z_to_gamma(self.z, self.z0);
        let (r_val, c_val) = z_to_rc(self.z, &self.freq);
        self.r.set_val(r_val);
        self.c.set_val(c_val);
        self
    }

    pub fn set_z(&mut self, z: Complex64) -> &Self {
        self.z = z;
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        let (r_val, c_val) = z_to_rc(self.z, &self.freq);
        self.r.set_val(r_val);
        self.c.set_val(c_val);
        self
    }

    pub fn set_z0(&mut self, z0: f64) -> &Self {
        self.z0 = z0;
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_r(&mut self, r: &UnitVal) -> &Self {
        self.r = *r;
        self.z = rc_to_z(self.r.val(), self.c.val(), &self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_c(&mut self, c: &UnitVal) -> &Self {
        self.c = *c;
        self.z = rc_to_z(self.r.val(), self.c.val(), &self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_freq(&mut self, freq: &UnitVal) -> &Self {
        let z = rc_to_z(self.r.val(), self.c.val(), &self.freq);
        self.freq = *freq;
        let (r_val, c_val) = z_to_rc(z, &self.freq);
        self.r.set_val(r_val);
        self.c.set_val(c_val);
        self
    }
}

impl Default for Impedance {
    fn default() -> Self {
        Self {
            y: c64(0.01, 0.0),
            z: c64(100.0, 0.0),
            g: c64(0.0, 0.0),
            z0: 100.0,
            freq: UnitVal::new_scaled(280.0, Scale::Giga, Unit::Hz),
            r: UnitVal::new(100.0, Scale::Base, Unit::Ohm),
            c: UnitVal::new_scaled(0.0, Scale::Femto, Unit::Farad),
        }
    }
}

#[derive(Default, Debug)]
pub struct ImpedanceBuilder {
    kind: ImpedanceType,
    category: Option<ComplexNumberType>,
    mode: ImpedanceMode,
    ri: Complex64,
    mag: Option<f64>,
    ang: Option<f64>,
    r: UnitVal,
    c: UnitVal,
    z0: f64,
    freq: UnitVal,
}

impl ImpedanceBuilder {
    pub fn new() -> Self {
        ImpedanceBuilder::default()
    }

    pub fn kind(mut self, imp: ImpedanceType) -> Self {
        self.kind = imp;
        self
    }

    pub fn kind_str(mut self, imp: &str) -> Self {
        self.kind = match ImpedanceType::from_str(imp) {
            Ok(x) => x,
            Err(_) => self.kind,
        };
        self
    }

    pub fn category(mut self, cat: ComplexNumberType) -> Self {
        self.category = Some(cat);
        self
    }

    pub fn category_str(mut self, cat: &str) -> Self {
        self.category = match ComplexNumberType::from_str(cat) {
            Ok(x) => Some(x),
            Err(_) => self.category,
        };
        self
    }

    pub fn mode(mut self, mode: ImpedanceMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn mode_str(mut self, mode: &str) -> Self {
        self.mode = match ImpedanceMode::from_str(mode) {
            Ok(x) => x,
            Err(_) => self.mode,
        };
        self
    }

    pub fn ri(mut self, ri: Complex64) -> Self {
        self.ri = ri;
        self
    }

    pub fn re(mut self, re: f64) -> Self {
        self.ri = c64(re, self.ri.im);
        self
    }

    pub fn im(mut self, im: f64) -> Self {
        self.ri = c64(self.ri.re, im);
        self
    }

    pub fn mag(mut self, mag: f64) -> Self {
        self.mag = Some(mag);
        self
    }

    pub fn db(mut self, db: f64) -> Self {
        self.mag = Some(10_f64.powf(db / 20.0));
        self
    }

    pub fn ang(mut self, ang: f64) -> Self {
        self.ang = Some(ang);
        self
    }

    pub fn x(mut self, x: f64) -> Self {
        match (self.kind, self.category) {
            (ImpedanceType::Gamma, None)
            | (ImpedanceType::Gamma, Some(ComplexNumberType::ReIm))
            | (ImpedanceType::Y, _)
            | (ImpedanceType::Z, _) => self.ri = c64(x, self.ri.im),
            (ImpedanceType::Gamma, Some(ComplexNumberType::MagAng)) => self.mag = Some(x),
            (ImpedanceType::Gamma, Some(ComplexNumberType::Db)) => {
                self.mag = Some(10_f64.powf(x / 20.0))
            }
            (ImpedanceType::Rc, _) => self.r = *self.r.set_val_scaled(x),
        }
        self
    }

    pub fn y(mut self, y: f64) -> Self {
        match (self.kind, self.category) {
            (ImpedanceType::Gamma, None)
            | (ImpedanceType::Gamma, Some(ComplexNumberType::ReIm))
            | (ImpedanceType::Y, _)
            | (ImpedanceType::Z, _) => self.ri = c64(self.ri.re, y),
            (ImpedanceType::Gamma, Some(ComplexNumberType::MagAng)) => self.ang = Some(y),
            (ImpedanceType::Gamma, Some(ComplexNumberType::Db)) => {
                self.ang = Some(10_f64.powf(y / 20.0))
            }
            (ImpedanceType::Rc, _) => self.c = *self.c.set_val_scaled(y),
        }
        self
    }

    pub fn res(mut self, res: UnitVal) -> Self {
        self.r = res;
        self
    }

    pub fn res_val(mut self, res: f64) -> Self {
        self.r.set_val(res);
        self
    }

    pub fn res_val_scaled(mut self, res: f64) -> Self {
        self.r.set_val_scaled(res);
        self
    }

    pub fn res_unit(mut self, unit: Scale) -> Self {
        self.r.set_scale(unit);
        self
    }

    pub fn res_unit_str(mut self, unit: &str) -> Self {
        self.r.set_scale(Scale::from_str(unit).unwrap());
        self
    }

    pub fn cap(mut self, cap: UnitVal) -> Self {
        self.c = cap;
        self
    }

    pub fn cap_val(mut self, cap: f64) -> Self {
        self.c.set_val(cap);
        self
    }

    pub fn cap_val_scaled(mut self, cap: f64) -> Self {
        self.c.set_val_scaled(cap);
        self
    }

    pub fn cap_unit(mut self, unit: Scale) -> Self {
        self.c.set_scale(unit);
        self
    }

    pub fn cap_unit_str(mut self, unit: &str) -> Self {
        self.c.set_scale(Scale::from_str(unit).unwrap());
        self
    }

    pub fn z0(mut self, z0: f64) -> Self {
        self.z0 = z0;
        self
    }

    pub fn freq(mut self, freq: UnitVal) -> Self {
        self.freq = freq;
        self
    }

    pub fn freq_val(mut self, freq: f64) -> Self {
        self.freq.set_val(freq);
        self
    }

    pub fn freq_val_scaled(mut self, freq: f64, unit: Scale) -> Self {
        self.freq = UnitValBuilder::new().val_scaled(freq, unit).build();
        self
    }

    pub fn build(mut self) -> Impedance {
        if self.mode == ImpedanceMode::Diff {
            self.z0 /= 2.0;
        }

        match self.kind {
            ImpedanceType::Gamma => {
                let g = match (self.category, self.mag, self.ang) {
                    (None, Some(mag), Some(ang))
                    | (Some(ComplexNumberType::MagAng), Some(mag), Some(ang))
                    | (Some(ComplexNumberType::Db), Some(mag), Some(ang)) => {
                        Complex64::from_polar(mag, ang * PI / 180.0)
                    }
                    _ => self.ri,
                };
                let z = gamma_to_z(g, self.z0);
                let (r, c) = z_to_rc(z, &self.freq);
                Impedance {
                    y: 1.0 / z,
                    z,
                    g,
                    r: UnitValBuilder::new().val(r).scale(Scale::Base).build(),
                    c: UnitValBuilder::new().val(c).scale(Scale::Femto).build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Y => {
                let y = match self.mode {
                    ImpedanceMode::Diff => 2.0 * self.ri,
                    ImpedanceMode::Se => self.ri,
                };
                let z = 1.0 / y;
                let (r, c) = z_to_rc(z, &self.freq);
                Impedance {
                    y,
                    z,
                    g: z_to_gamma(z, self.z0),
                    r: UnitValBuilder::new().val(r).scale(Scale::Base).build(),
                    c: UnitValBuilder::new().val(c).scale(Scale::Femto).build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Z => {
                let z = match self.mode {
                    ImpedanceMode::Diff => self.ri / 2.0,
                    ImpedanceMode::Se => self.ri,
                };
                let (r, c) = z_to_rc(z, &self.freq);
                Impedance {
                    y: 1.0 / z,
                    z,
                    g: z_to_gamma(z, self.z0),
                    r: UnitValBuilder::new().val(r).scale(Scale::Base).build(),
                    c: UnitValBuilder::new().val(c).scale(Scale::Femto).build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Rc => {
                match self.mode {
                    ImpedanceMode::Diff => {
                        self.r.set_val(self.r.val() / 2.0);
                        self.c.set_val(self.c.val() * 2.0);
                    }
                    ImpedanceMode::Se => (),
                }
                let z = rc_to_z(self.r.val(), self.c.val(), &self.freq);
                Impedance {
                    y: 1.0 / z,
                    z,
                    g: z_to_gamma(z, self.z0),
                    r: self.r,
                    c: self.c,
                    z0: self.z0,
                    freq: self.freq,
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ImpedanceType {
    Gamma,
    Y,
    #[default]
    Z,
    Rc,
}

impl FromStr for ImpedanceType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "g" | "G" | "gamma" | "Gamma" | "Γ" => Ok(ImpedanceType::Gamma),
            "y" | "Y" | "adm" | "admittance" => Ok(ImpedanceType::Y),
            "z" | "Z" | "imp" | "impedance" => Ok(ImpedanceType::Z),
            "rc" | "RC" | "rescap" | "ResCap" => Ok(ImpedanceType::Rc),
            _ => Err("ImpedanceType not recognized".to_string().into()),
        }
    }
}

impl fmt::Display for ImpedanceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ImpedanceType::Gamma => write!(f, "g"),
            ImpedanceType::Y => write!(f, "y"),
            ImpedanceType::Z => write!(f, "z"),
            ImpedanceType::Rc => write!(f, "rc"),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ImpedanceMode {
    #[default]
    Se,
    Diff,
}

impl FromStr for ImpedanceMode {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "se" => Ok(ImpedanceMode::Se),
            "diff" => Ok(ImpedanceMode::Diff),
            _ => Err("ImpedanceMode not recognized".to_string().into()),
        }
    }
}

impl fmt::Display for ImpedanceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ImpedanceMode::Se => write!(f, "se"),
            ImpedanceMode::Diff => write!(f, "diff"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct ComplexNumber {
    kind: ComplexNumberType,
    re: f64,
    im: f64,
}

impl ComplexNumber {
    pub fn convert(&self, kind: ComplexNumberType) -> ComplexNumber {
        match (self.kind, kind) {
            (ComplexNumberType::ReIm, ComplexNumberType::ReIm) => *self,
            (ComplexNumberType::ReIm, ComplexNumberType::MagAng) => {
                let val = c64(self.re, self.im);
                ComplexNumber {
                    kind: ComplexNumberType::MagAng,
                    re: val.norm(),
                    im: 180.0 / PI * val.arg(),
                }
            }
            (ComplexNumberType::ReIm, ComplexNumberType::Db) => {
                let val = c64(self.re, self.im);
                ComplexNumber {
                    kind: ComplexNumberType::Db,
                    re: 20.0 * val.norm().log10(),
                    im: 180.0 / PI * val.arg(),
                }
            }
            (ComplexNumberType::MagAng, ComplexNumberType::ReIm) => {
                let val = Complex64::from_polar(self.re, PI / 180.0 * self.im);
                ComplexNumber {
                    kind: ComplexNumberType::ReIm,
                    re: val.re,
                    im: val.im,
                }
            }
            (ComplexNumberType::MagAng, ComplexNumberType::MagAng) => *self,
            (ComplexNumberType::MagAng, ComplexNumberType::Db) => ComplexNumber {
                kind: ComplexNumberType::Db,
                re: 20.0 * self.re.log10(),
                im: self.im,
            },
            (ComplexNumberType::Db, ComplexNumberType::ReIm) => {
                let val = Complex64::from_polar(10_f64.powf(self.re / 20.0), PI / 180.0 * self.im);
                ComplexNumber {
                    kind: ComplexNumberType::ReIm,
                    re: val.re,
                    im: val.im,
                }
            }
            (ComplexNumberType::Db, ComplexNumberType::MagAng) => ComplexNumber {
                kind: ComplexNumberType::MagAng,
                re: 10_f64.powf(self.re / 20.0),
                im: self.im,
            },
            (ComplexNumberType::Db, ComplexNumberType::Db) => *self,
        }
    }

    pub fn ri(&self) -> Complex64 {
        match self.kind {
            ComplexNumberType::ReIm => c64(self.re, self.im),
            ComplexNumberType::MagAng => Complex64::from_polar(self.re, PI / 180.0 * self.im),
            ComplexNumberType::Db => {
                Complex64::from_polar(10_f64.powf(self.re / 20.0), PI / 180.0 * self.im)
            }
        }
    }

    pub fn mag(&self) -> f64 {
        match self.kind {
            ComplexNumberType::ReIm => c64(self.re, self.im).norm(),
            ComplexNumberType::MagAng => self.re,
            ComplexNumberType::Db => 10_f64.powf(self.re / 20.0),
        }
    }

    pub fn db(&self) -> f64 {
        match self.kind {
            ComplexNumberType::ReIm => 20.0 * c64(self.re, self.im).norm().log10(),
            ComplexNumberType::MagAng => 20.0 * self.re.log10(),
            ComplexNumberType::Db => self.re,
        }
    }

    pub fn ang(&self) -> f64 {
        match self.kind {
            ComplexNumberType::ReIm => c64(self.re, self.im).arg() * 180.0 / PI,
            ComplexNumberType::MagAng | ComplexNumberType::Db => self.im,
        }
    }
}

#[derive(Default)]
pub struct ComplexNumberBuilder {
    kind: ComplexNumberType,
    re: f64,
    im: f64,
}

impl ComplexNumberBuilder {
    pub fn new() -> Self {
        ComplexNumberBuilder::default()
    }

    pub fn kind(mut self, val: ComplexNumberType) -> Self {
        self.kind = val;
        self
    }

    pub fn kind_from_str(mut self, val: &str) -> Self {
        self.kind = ComplexNumberType::from_str(val).unwrap();
        self
    }

    pub fn ri(mut self, val: Complex64) -> Self {
        self.re = val.re;
        self.im = val.im;
        self
    }

    pub fn real(mut self, val: f64) -> Self {
        self.re = val;
        self
    }

    pub fn imag(mut self, val: f64) -> Self {
        self.im = val;
        self
    }

    pub fn mag(mut self, val: f64) -> Self {
        self.re = val;
        self
    }

    pub fn db(mut self, val: f64) -> Self {
        self.re = val;
        self
    }

    pub fn angle(mut self, val: f64) -> Self {
        self.im = val;
        self
    }

    pub fn build(self) -> ComplexNumber {
        ComplexNumber {
            kind: self.kind,
            re: self.re,
            im: self.im,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ComplexNumberType {
    #[default]
    ReIm,
    MagAng,
    Db,
}

impl ComplexNumberType {
    pub fn parse(&self, x: f64, y: f64) -> Complex64 {
        match self {
            ComplexNumberType::ReIm => c64(x, y),
            ComplexNumberType::MagAng => Complex64::from_polar(x, f64::to_radians(y)),
            ComplexNumberType::Db => {
                Complex64::from_polar(10f64.powf(x / 20.0), f64::to_radians(y))
            }
        }
    }
}

impl FromStr for ComplexNumberType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ri" | "RI" | "reim" => Ok(ComplexNumberType::ReIm),
            "ma" | "MA" | "magang" => Ok(ComplexNumberType::MagAng),
            "db" | "DB" | "dbang" => Ok(ComplexNumberType::Db),
            _ => Err("ComplexNumberType not recognized".to_string().into()),
        }
    }
}

impl fmt::Display for ComplexNumberType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ComplexNumberType::ReIm => write!(f, "ri"),
            ComplexNumberType::MagAng => write!(f, "ma"),
            ComplexNumberType::Db => write!(f, "db"),
        }
    }
}

#[cfg(test)]
mod impedance_tests {
    use super::*;
    use crate::util::{comp_c64, comp_f64};
    use float_cmp::F64Margin;

    #[test]
    fn test_impedance() {
        let z = c64(42.4, -19.6);
        let y = c64(0.01943242648676395, 0.008982914130673902);
        let gamma = c64(-0.37838109914277085, -0.18972099398313422);
        let r = UnitVal::new(51.46037735849057, Scale::Base, Unit::Ohm);
        let c = UnitVal::new(5.105982811667098e-15, Scale::Femto, Unit::Farad);
        let freq = UnitVal::new_scaled(27.0, Scale::Giga, Unit::Hz);
        let mut test = Impedance::default();

        comp_c64(
            &test.gamma(),
            &c64(0.0, 0.0),
            F64Margin::default(),
            "gamma()",
            "gamma",
        );
        comp_c64(&test.y(), &c64(0.01, 0.0), F64Margin::default(), "y()", "y");
        comp_c64(
            &test.z(),
            &c64(100.0, 0.0),
            F64Margin::default(),
            "z()",
            "z",
        );
        comp_f64(&test.r().val(), &100.0, F64Margin::default(), "r()", "r");
        comp_f64(&test.c().val(), &0.0, F64Margin::default(), "c()", "c");
        comp_f64(&test.z0(), &100.0, F64Margin::default(), "z0()", "z0");
        comp_f64(
            &test.freq().val(),
            &280.0e9,
            F64Margin::default(),
            "freq()",
            "freq",
        );

        test.set_z(z);
        let test_name = "set_z()";
        comp_c64(
            &test.gamma(),
            &gamma,
            F64Margin::default(),
            test_name,
            "gamma",
        );
        comp_c64(&test.y(), &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z(), &z, F64Margin::default(), test_name, "z");
        comp_f64(
            &test.r().val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c().val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        comp_f64(&test.z0(), &100.0, F64Margin::default(), test_name, "z0");
        comp_f64(
            &test.freq().val(),
            &280.0e9,
            F64Margin::default(),
            test_name,
            "freq",
        );

        let mut test = Impedance::default();
        test.set_y(y);
        let test_name = "set_y()";
        comp_c64(
            &test.gamma(),
            &gamma,
            F64Margin::default(),
            test_name,
            "gamma",
        );
        comp_c64(&test.y(), &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z(), &z, F64Margin::default(), test_name, "z");
        comp_f64(
            &test.r().val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c().val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        comp_f64(&test.z0(), &100.0, F64Margin::default(), test_name, "z0");
        comp_f64(
            &test.freq().val(),
            &280.0e9,
            F64Margin::default(),
            test_name,
            "freq",
        );

        let mut test = Impedance::default();
        test.set_gamma(gamma);
        let test_name = "set_gamma()";
        comp_c64(
            &test.gamma(),
            &gamma,
            F64Margin::default(),
            test_name,
            "gamma",
        );
        comp_c64(&test.y(), &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z(), &z, F64Margin::default(), test_name, "z");
        comp_f64(
            &test.r().val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c().val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        comp_f64(&test.z0(), &100.0, F64Margin::default(), test_name, "z0");
        comp_f64(
            &test.freq().val(),
            &280.0e9,
            F64Margin::default(),
            test_name,
            "freq",
        );

        let mut test = Impedance::default();
        test.set_r(&r);
        let test_name = "set_r()";
        comp_c64(
            &test.gamma(),
            &c64(-0.3204773650247901, 0.0),
            F64Margin::default(),
            test_name,
            "gamma",
        );
        comp_c64(
            &test.y(),
            &c64(0.01943242648676395, 0.0),
            F64Margin::default(),
            test_name,
            "y",
        );
        comp_c64(
            &test.z(),
            &c64(51.46037735849057, 0.0),
            F64Margin::default(),
            test_name,
            "z",
        );
        comp_f64(
            &test.r().val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(&test.c().val(), &0.0, F64Margin::default(), test_name, "c");
        comp_f64(&test.z0(), &100.0, F64Margin::default(), test_name, "z0");
        comp_f64(
            &test.freq().val(),
            &280.0e9,
            F64Margin::default(),
            test_name,
            "freq",
        );

        let mut test = Impedance::default();
        test.set_c(&c);
        let test_name = "set_c()";
        comp_c64(
            &test.gamma(),
            &c64(-0.16786761794032898, -0.37374868666975763),
            F64Margin::default(),
            test_name,
            "gamma",
        );
        comp_c64(
            &test.y(),
            &c64(0.01, 0.008982914130673902),
            F64Margin::default(),
            test_name,
            "y",
        );
        comp_c64(
            &test.z(),
            &c64(55.342564690206515, -49.71375063833907),
            F64Margin::default(),
            test_name,
            "z",
        );
        comp_f64(
            &test.r().val(),
            &100.0,
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c().val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        comp_f64(&test.z0(), &100.0, F64Margin::default(), test_name, "z0");
        comp_f64(
            &test.freq().val(),
            &280.0e9,
            F64Margin::default(),
            test_name,
            "freq",
        );

        let mut test =
            Impedance::new_from_z(z, 100.0, &UnitVal::new_scaled(280.0, Scale::Giga, Unit::Hz));
        test.set_freq(&freq);
        let test_name = "set_freq()";
        comp_c64(
            &test.gamma(),
            &gamma,
            F64Margin::default(),
            test_name,
            "gamma",
        );
        comp_c64(&test.y(), &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z(), &z, F64Margin::default(), test_name, "z");
        comp_f64(
            &test.r().val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c().val(),
            &5.295093286173287e-14,
            F64Margin::default(),
            test_name,
            "c",
        );
        comp_f64(&test.z0(), &100.0, F64Margin::default(), test_name, "z0");
        comp_f64(
            &test.freq().val(),
            &27.0e9,
            F64Margin::default(),
            test_name,
            "freq",
        );
    }

    #[test]
    fn test_impedancebuilder() {
        let z = c64(42.4, -19.6);
        let y = c64(0.01943242648676395, 0.008982914130673902);
        let gamma = c64(-0.37838109914277085, -0.18972099398313422);
        let mag = gamma.norm();
        let db = 20.0 * mag.log10();
        let ang = gamma.arg() * 180.0 / PI;
        let r = UnitVal::new(51.46037735849057, Scale::Base, Unit::Ohm);
        let c = UnitVal::new(5.105982811667098e-15, Scale::Femto, Unit::Farad);
        let z0 = 100.0;
        let freq = UnitVal::new_scaled(280.0, Scale::Giga, Unit::Hz);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .ri(gamma)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(RI)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .mode(ImpedanceMode::Diff)
            .ri(gamma)
            .z0(z0 * 2.0)
            .freq(freq)
            .build();
        let test_name = "Gamma(RI)Diff";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .mag(mag)
            .ang(ang)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(MA)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .category(ComplexNumberType::MagAng)
            .mag(mag)
            .ang(ang)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(MA)MagAng";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .category(ComplexNumberType::ReIm)
            .mag(0.1)
            .ang(2.0)
            .ri(gamma)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(MA)ReIm";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .db(db)
            .ang(ang)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(dB)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .ri(c64(0.0, 0.0))
            .mag(mag)
            .ang(ang)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(MA2)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .ri(gamma)
            .mag(0.0)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(RI2)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Y)
            .ri(y)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Y";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Y)
            .mode(ImpedanceMode::Diff)
            .ri(y / 2.0)
            .z0(z0 * 2.0)
            .freq(freq)
            .build();
        let test_name = "Ydiff";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .ri(z)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Z";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .mode(ImpedanceMode::Diff)
            .ri(z * 2.0)
            .z0(z0 * 2.0)
            .freq(freq)
            .build();
        let test_name = "Zdiff";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Rc)
            .res(r)
            .cap(c)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "RC";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let r_diff = *r.clone().set_val(r.val() * 2.0);
        let c_diff = *c.clone().set_val(c.val() / 2.0);
        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Rc)
            .mode(ImpedanceMode::Diff)
            .res(r_diff)
            .cap(c_diff)
            .z0(z0 * 2.0)
            .freq(freq)
            .build();
        let test_name = "RCdiff";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .x(gamma.re)
            .y(gamma.im)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(RI)xy";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Gamma)
            .category(ComplexNumberType::MagAng)
            .x(mag)
            .y(ang)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Gamma(MA)xy";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Y)
            .x(y.re)
            .y(y.im)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Y(xy)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .x(z.re)
            .y(z.im)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "Z(xy)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);

        let test = ImpedanceBuilder::new()
            .kind(ImpedanceType::Rc)
            .x(r.val())
            .y(c.val())
            .cap_unit(Scale::Femto)
            .z0(z0)
            .freq(freq)
            .build();
        let test_name = "RC(xy)";
        comp_c64(&test.y, &y, F64Margin::default(), test_name, "y");
        comp_c64(&test.z, &z, F64Margin::default(), test_name, "z");
        comp_c64(&test.g, &gamma, F64Margin::default(), test_name, "g");
        comp_f64(
            &test.r.val(),
            &r.val(),
            F64Margin::default(),
            test_name,
            "r",
        );
        comp_f64(
            &test.c.val(),
            &c.val(),
            F64Margin::default(),
            test_name,
            "c",
        );
        assert_eq!(test.r.scale(), Scale::Base);
        assert_eq!(test.c.scale(), Scale::Femto);
        assert_eq!(test.z0, z0);
        assert_eq!(test.freq, freq);
    }

    #[test]
    fn test_impedancetype() {
        let gamma = ["g", "G", "gamma", "Gamma", "Γ"];
        let y = ["y", "Y", "adm", "admittance"];
        let z = ["z", "Z", "imp", "impedance"];
        let rc = ["rc", "RC", "rescap", "ResCap"];
        let nada = ["", "google", ".sfwe"];

        for val in gamma.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::Gamma);
        }
        for val in y.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::Y);
        }
        for val in z.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::Z);
        }
        for val in rc.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::Rc);
        }
        for val in nada.iter() {
            assert_eq!(
                ImpedanceType::from_str(val).unwrap_err().to_string(),
                "ImpedanceType not recognized".to_string()
            )
        }
    }

    #[test]
    fn test_complexnumber() {
        let val = c64(-0.37838109914277085, -0.18972099398313422);
        let mag = 0.42328041739069366;
        let db = -7.467436464485704;
        let ang = -153.3707274747144;
        let exemplar_ri = ComplexNumber {
            kind: ComplexNumberType::ReIm,
            re: val.re,
            im: val.im,
        };
        let exemplar_magang = ComplexNumber {
            kind: ComplexNumberType::MagAng,
            re: mag,
            im: ang,
        };
        let exemplar_db = ComplexNumber {
            kind: ComplexNumberType::Db,
            re: db,
            im: ang,
        };

        let test = ComplexNumberBuilder::new()
            .ri(val)
            .kind(ComplexNumberType::ReIm)
            .build();
        assert_eq!(test.convert(ComplexNumberType::ReIm), exemplar_ri);
        assert_eq!(
            test.convert(ComplexNumberType::MagAng).kind,
            exemplar_magang.kind
        );
        comp_f64(
            &test.convert(ComplexNumberType::MagAng).mag(),
            &exemplar_magang.re,
            F64Margin::default(),
            "convert",
            "ri_to_ma",
        );
        comp_f64(
            &test.convert(ComplexNumberType::MagAng).ang(),
            &exemplar_magang.im,
            F64Margin::default(),
            "convert",
            "ri_to_ma",
        );
        assert_eq!(test.convert(ComplexNumberType::Db).kind, exemplar_db.kind);
        comp_f64(
            &test.convert(ComplexNumberType::Db).db(),
            &exemplar_db.re,
            F64Margin::default(),
            "convert",
            "ri_to_db",
        );
        comp_f64(
            &test.convert(ComplexNumberType::Db).ang(),
            &exemplar_db.im,
            F64Margin::default(),
            "convert",
            "ri_to_db",
        );

        let test = ComplexNumberBuilder::new()
            .mag(mag)
            .angle(ang)
            .kind(ComplexNumberType::MagAng)
            .build();
        assert_eq!(test.convert(ComplexNumberType::ReIm).kind, exemplar_ri.kind);
        comp_c64(
            &test.convert(ComplexNumberType::ReIm).ri(),
            &val,
            F64Margin::default(),
            "convert",
            "ma_to_ri",
        );
        assert_eq!(test.convert(ComplexNumberType::MagAng), exemplar_magang);
        assert_eq!(test.convert(ComplexNumberType::Db).kind, exemplar_db.kind);
        comp_f64(
            &test.convert(ComplexNumberType::Db).re,
            &exemplar_db.re,
            F64Margin::default(),
            "convert",
            "ma_to_db",
        );
        comp_f64(
            &test.convert(ComplexNumberType::Db).im,
            &exemplar_db.im,
            F64Margin::default(),
            "convert",
            "ma_to_db",
        );

        let test = ComplexNumberBuilder::new()
            .db(db)
            .angle(ang)
            .kind(ComplexNumberType::Db)
            .build();
        assert_eq!(test.convert(ComplexNumberType::ReIm).kind, exemplar_ri.kind);
        comp_f64(
            &test.convert(ComplexNumberType::ReIm).re,
            &exemplar_ri.re,
            F64Margin::default(),
            "convert",
            "db_to_ri",
        );
        comp_f64(
            &test.convert(ComplexNumberType::ReIm).im,
            &exemplar_ri.im,
            F64Margin::default(),
            "convert",
            "db_to_ri",
        );
        assert_eq!(
            test.convert(ComplexNumberType::MagAng).kind,
            exemplar_magang.kind
        );
        comp_f64(
            &test.convert(ComplexNumberType::MagAng).re,
            &exemplar_magang.re,
            F64Margin::default(),
            "convert",
            "db_to_ma",
        );
        comp_f64(
            &test.convert(ComplexNumberType::MagAng).im,
            &exemplar_magang.im,
            F64Margin::default(),
            "convert",
            "db_to_ma",
        );
        assert_eq!(test.convert(ComplexNumberType::Db), exemplar_db);
    }

    #[test]
    fn test_complexnumberbuilder() {
        let exemplar_ri = c64(-0.37838109914277085, -0.18972099398313422);
        let exemplar_mag = 0.42328041739069366;
        let exemplar_db = -7.467436464485704;
        let exemplar_ang = -153.3707274747144;

        let test = ComplexNumberBuilder::new()
            .kind(ComplexNumberType::ReIm)
            .ri(exemplar_ri)
            .build();
        assert_eq!(
            test,
            ComplexNumber {
                kind: ComplexNumberType::ReIm,
                re: exemplar_ri.re,
                im: exemplar_ri.im
            }
        );

        let test = ComplexNumberBuilder::new()
            .kind(ComplexNumberType::MagAng)
            .mag(exemplar_mag)
            .angle(exemplar_ang)
            .build();
        assert_eq!(
            test,
            ComplexNumber {
                kind: ComplexNumberType::MagAng,
                re: exemplar_mag,
                im: exemplar_ang
            }
        );

        let test = ComplexNumberBuilder::new()
            .kind(ComplexNumberType::Db)
            .mag(exemplar_db)
            .angle(exemplar_ang)
            .build();
        assert_eq!(
            test,
            ComplexNumber {
                kind: ComplexNumberType::Db,
                re: exemplar_db,
                im: exemplar_ang
            }
        );
    }

    #[test]
    fn test_complexnumbertype() {
        let ri = ["ri", "reim"];
        let ma = ["ma", "magang"];
        let db = ["db", "dbang"];
        let nada = ["", "google", ".sfwe"];

        for val in ri.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap(),
                ComplexNumberType::ReIm
            );
        }
        for val in ma.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap(),
                ComplexNumberType::MagAng
            );
        }
        for val in db.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap(),
                ComplexNumberType::Db
            );
        }
        for val in nada.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap_err().to_string(),
                "ComplexNumberType not recognized".to_string()
            )
        }
    }
}
