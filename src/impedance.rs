use crate::math::*;
use crate::scale::Scale;
use crate::unit::{Unit, UnitValBuilder, UnitValue};
use num::complex::{Complex64, c64};
use serde::Serialize;
use std::f64::consts::PI;
use std::fmt;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct Impedance {
    mode: ImpedanceMode, // differential or single-ended input
    y: Complex64,
    z: Complex64,
    g: Complex64,
    rp: UnitValue,
    cp: UnitValue,
    rs: UnitValue,
    cs: UnitValue,
    z0: f64,
    freq: UnitValue,
}

impl Impedance {
    pub fn new_from_gamma(
        gamma: Complex64,
        z0: f64,
        freq: &UnitValue,
        mode: ImpedanceMode,
    ) -> Self {
        let mut out = Impedance::default();
        if mode == ImpedanceMode::Diff {
            out.mode = ImpedanceMode::Diff;
        }
        out.set_gamma(gamma);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_y(y: Complex64, z0: f64, freq: &UnitValue, mode: ImpedanceMode) -> Self {
        let mut out = Impedance::default();
        if mode == ImpedanceMode::Diff {
            out.mode = ImpedanceMode::Diff;
        }
        out.set_y(y);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_z(z: Complex64, z0: f64, freq: &UnitValue, mode: ImpedanceMode) -> Self {
        let mut out = Impedance::default();
        if mode == ImpedanceMode::Diff {
            out.mode = ImpedanceMode::Diff;
        }
        out.set_z(z);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_rpcp(
        rp: &UnitValue,
        cp: &UnitValue,
        z0: f64,
        freq: &UnitValue,
        mode: ImpedanceMode,
    ) -> Self {
        let mut out = Impedance::default();
        if mode == ImpedanceMode::Diff {
            out.mode = ImpedanceMode::Diff;
        }
        out.set_rp(rp);
        out.set_cp(cp);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_rscs(
        rs: &UnitValue,
        cs: &UnitValue,
        z0: f64,
        freq: &UnitValue,
        mode: ImpedanceMode,
    ) -> Self {
        let mut out = Impedance::default();
        if mode == ImpedanceMode::Diff {
            out.mode = ImpedanceMode::Diff;
        }
        out.set_rs(rs);
        out.set_cs(cs);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn mode(&self) -> ImpedanceMode {
        self.mode
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

    pub fn rp(&self) -> UnitValue {
        self.rp
    }

    pub fn cp(&self) -> UnitValue {
        self.cp
    }

    pub fn rs(&self) -> UnitValue {
        self.rs
    }

    pub fn cs(&self) -> UnitValue {
        self.cs
    }

    pub fn z0(&self) -> f64 {
        self.z0
    }

    pub fn freq(&self) -> UnitValue {
        self.freq
    }

    pub fn diff_to_se(&mut self) -> &Self {
        if self.mode == ImpedanceMode::Se {
            ()
        } else {
            self.mode = ImpedanceMode::Se;
            self.set_z(self.z / 2.0);
            self.set_z0(self.z0 / 2.0);
        }

        self
    }

    pub fn se_to_diff(&mut self) -> &Self {
        if self.mode == ImpedanceMode::Se {
            self.mode = ImpedanceMode::Diff;
            self.set_z(self.z * 2.0);
            self.set_z0(self.z0 * 2.0);
        } else {
            ()
        }

        self
    }

    pub fn set_gamma(&mut self, g: Complex64) -> &Self {
        self.g = g;
        self.z = gamma_to_z(self.g, self.z0);
        self.y = z_to_y(self.z);
        let (rp, cp) = z_to_rpcp(self.z, &self.freq);
        self.rp.set_val(rp);
        self.cp.set_val(cp);
        let (rs, cs) = z_to_rscs(self.z, &self.freq);
        self.rs.set_val(rs);
        self.cs.set_val(cs);
        self
    }

    pub fn set_y(&mut self, y: Complex64) -> &Self {
        self.y = y;
        self.z = y_to_z(self.y);
        self.g = z_to_gamma(self.z, self.z0);
        let (rp, cp) = z_to_rpcp(self.z, &self.freq);
        self.rp.set_val(rp);
        self.cp.set_val(cp);
        let (rs, cs) = z_to_rscs(self.z, &self.freq);
        self.rs.set_val(rs);
        self.cs.set_val(cs);
        self
    }

    pub fn set_z(&mut self, z: Complex64) -> &Self {
        self.z = z;
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        let (rp, cp) = z_to_rpcp(self.z, &self.freq);
        self.rp.set_val(rp);
        self.cp.set_val(cp);
        let (rs, cs) = z_to_rscs(self.z, &self.freq);
        self.rs.set_val(rs);
        self.cs.set_val(cs);
        self
    }

    pub fn set_z0(&mut self, z0: f64) -> &Self {
        self.z0 = z0;
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_rp(&mut self, rp: &UnitValue) -> &Self {
        self.rp = *rp;
        let (rs, cs) = rpcp_to_rscs(self.rp.val(), self.cp.val(), &self.freq);
        self.rs.set_val(rs);
        self.cs.set_val(cs);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), &self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_cp(&mut self, cp: &UnitValue) -> &Self {
        self.cp = *cp;
        let (rs, cs) = rpcp_to_rscs(self.rp.val(), self.cp.val(), &self.freq);
        self.rs.set_val(rs);
        self.cs.set_val(cs);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), &self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_rs(&mut self, rs: &UnitValue) -> &Self {
        self.rs = *rs;
        let (rp, cp) = rscs_to_rpcp(self.rs.val(), self.cs.val(), &self.freq);
        self.rp.set_val(rp);
        self.cp.set_val(cp);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), &self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_cs(&mut self, cs: &UnitValue) -> &Self {
        self.cs = *cs;
        let (rp, cp) = rscs_to_rpcp(self.rs.val(), self.cs.val(), &self.freq);
        self.rp.set_val(rp);
        self.cp.set_val(cp);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), &self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_freq(&mut self, freq: &UnitValue) -> &Self {
        let z = rpcp_to_z(self.rp.val(), self.cp.val(), &self.freq);
        self.freq = *freq;
        let (rp, cp) = z_to_rpcp(z, &self.freq);
        self.rp.set_val(rp);
        self.cp.set_val(cp);
        let (rs, cs) = z_to_rscs(z, &self.freq);
        self.rs.set_val(rs);
        self.cs.set_val(cs);
        self
    }
}

impl Default for Impedance {
    fn default() -> Self {
        Self {
            mode: ImpedanceMode::Se,
            y: c64(0.02, 0.0),
            z: c64(50.0, 0.0),
            g: c64(0.0, 0.0),
            z0: 50.0,
            freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
            rp: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
            cp: UnitValue::new_scaled(0.0, Scale::Femto, Unit::Farad),
            rs: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
            cs: UnitValue::new_scaled(0.0, Scale::Femto, Unit::Farad),
        }
    }
}

#[derive(Debug)]
pub struct ImpedanceBuilder {
    kind: ImpedanceType,
    category: Option<ComplexNumberType>,
    mode: ImpedanceMode, // differential or single-ended input
    ri: Complex64,       // real/imaginary form of complex input
    mag: Option<f64>,
    ang: Option<f64>,
    rp: UnitValue,
    cp: UnitValue,
    rs: UnitValue,
    cs: UnitValue,
    z0: f64,
    freq: UnitValue,
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
            (ImpedanceType::Rpcp, _) => self.rp = *self.rp.set_val_scaled(x),
            (ImpedanceType::Rscs, _) => self.rs = *self.rs.set_val_scaled(x),
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
            (ImpedanceType::Rpcp, _) => self.cp = *self.cp.set_val_scaled(y),
            (ImpedanceType::Rscs, _) => self.cs = *self.cs.set_val_scaled(y),
        }
        self
    }

    pub fn r_scale(mut self, scale: Scale) -> Self {
        self.rp.set_scale(scale);
        self.rs.set_scale(scale);
        self
    }

    pub fn r_scale_str(mut self, scale: &str) -> Self {
        self.rp.set_scale(Scale::from_str(scale).unwrap());
        self.rs.set_scale(Scale::from_str(scale).unwrap());
        self
    }

    pub fn c_scale(mut self, scale: Scale) -> Self {
        self.cp.set_scale(scale);
        self.cs.set_scale(scale);
        self
    }

    pub fn c_scale_str(mut self, scale: &str) -> Self {
        self.cp.set_scale(Scale::from_str(scale).unwrap());
        self.cs.set_scale(Scale::from_str(scale).unwrap());
        self
    }

    pub fn rp(mut self, res: UnitValue) -> Self {
        self.rp = res;
        self
    }

    pub fn rp_val(mut self, res: f64) -> Self {
        self.rp.set_val(res);
        self
    }

    pub fn rp_val_scaled(mut self, res: f64) -> Self {
        self.rp.set_val_scaled(res);
        self
    }

    pub fn cp(mut self, cap: UnitValue) -> Self {
        self.cp = cap;
        self
    }

    pub fn cp_val(mut self, cap: f64) -> Self {
        self.cp.set_val(cap);
        self
    }

    pub fn cp_val_scaled(mut self, cap: f64) -> Self {
        self.cp.set_val_scaled(cap);
        self
    }

    pub fn rs(mut self, res: UnitValue) -> Self {
        self.rs = res;
        self
    }

    pub fn rs_val(mut self, res: f64) -> Self {
        self.rs.set_val(res);
        self
    }

    pub fn rs_val_scaled(mut self, res: f64) -> Self {
        self.rs.set_val_scaled(res);
        self
    }

    pub fn cs(mut self, cap: UnitValue) -> Self {
        self.cs = cap;
        self
    }

    pub fn cs_val(mut self, cap: f64) -> Self {
        self.cs.set_val(cap);
        self
    }

    pub fn cs_val_scaled(mut self, cap: f64) -> Self {
        self.cs.set_val_scaled(cap);
        self
    }

    pub fn z0(mut self, z0: f64) -> Self {
        self.z0 = z0;
        self
    }

    pub fn freq(mut self, freq: UnitValue) -> Self {
        self.freq = freq;
        self
    }

    pub fn freq_val(mut self, freq: f64) -> Self {
        self.freq.set_val(freq);
        self
    }

    pub fn freq_val_scaled(mut self, freq: f64, scale: Scale) -> Self {
        self.freq = UnitValBuilder::new()
            .val_scaled(freq, scale)
            .unit(Unit::Hz)
            .build();
        self
    }

    pub fn build(self) -> Impedance {
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
                let (rp, cp) = z_to_rpcp(z, &self.freq);
                let (rs, cs) = z_to_rscs(z, &self.freq);
                Impedance {
                    mode: self.mode,
                    y: 1.0 / z,
                    z,
                    g,
                    rp: UnitValBuilder::new()
                        .val(rp)
                        .scale(self.rp.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cp: UnitValBuilder::new()
                        .val(cp)
                        .scale(self.cp.scale())
                        .unit(Unit::Farad)
                        .build(),
                    rs: UnitValBuilder::new()
                        .val(rs)
                        .scale(self.rs.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cs: UnitValBuilder::new()
                        .val(cs)
                        .scale(self.cs.scale())
                        .unit(Unit::Farad)
                        .build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Y => {
                let z = 1.0 / self.ri;
                let (rp, cp) = z_to_rpcp(z, &self.freq);
                let (rs, cs) = z_to_rscs(z, &self.freq);
                Impedance {
                    mode: self.mode,
                    y: self.ri,
                    z,
                    g: z_to_gamma(z, self.z0),
                    rp: UnitValBuilder::new()
                        .val(rp)
                        .scale(self.rp.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cp: UnitValBuilder::new()
                        .val(cp)
                        .scale(self.cp.scale())
                        .unit(Unit::Farad)
                        .build(),
                    rs: UnitValBuilder::new()
                        .val(rs)
                        .scale(self.rs.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cs: UnitValBuilder::new()
                        .val(cs)
                        .scale(self.cs.scale())
                        .unit(Unit::Farad)
                        .build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Z => {
                let (rp, cp) = z_to_rpcp(self.ri, &self.freq);
                let (rs, cs) = z_to_rscs(self.ri, &self.freq);
                Impedance {
                    mode: self.mode,
                    y: 1.0 / self.ri,
                    z: self.ri,
                    g: z_to_gamma(self.ri, self.z0),
                    rp: UnitValBuilder::new()
                        .val(rp)
                        .scale(self.rp.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cp: UnitValBuilder::new()
                        .val(cp)
                        .scale(self.cp.scale())
                        .unit(Unit::Farad)
                        .build(),
                    rs: UnitValBuilder::new()
                        .val(rs)
                        .scale(self.rs.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cs: UnitValBuilder::new()
                        .val(cs)
                        .scale(self.cs.scale())
                        .unit(Unit::Farad)
                        .build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Rpcp => {
                let z = rpcp_to_z(self.rp.val(), self.cp.val(), &self.freq);
                let (rs, cs) = z_to_rscs(z, &self.freq);
                let g = z_to_gamma(z, self.z0);
                Impedance {
                    mode: self.mode,
                    y: 1.0 / z,
                    z,
                    g,
                    rp: self.rp,
                    cp: self.cp,
                    rs: UnitValBuilder::new()
                        .val(rs)
                        .scale(self.rp.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cs: UnitValBuilder::new()
                        .val(cs)
                        .scale(self.cp.scale())
                        .unit(Unit::Farad)
                        .build(),
                    z0: self.z0,
                    freq: self.freq,
                }
            }
            ImpedanceType::Rscs => {
                let z = rscs_to_z(self.rs.val(), self.cs.val(), &self.freq);
                let (rp, cp) = z_to_rpcp(z, &self.freq);
                Impedance {
                    mode: self.mode,
                    y: 1.0 / z,
                    z,
                    g: z_to_gamma(z, self.z0),
                    rp: UnitValBuilder::new()
                        .val(rp)
                        .scale(self.rs.scale())
                        .unit(Unit::Ohm)
                        .build(),
                    cp: UnitValBuilder::new()
                        .val(cp)
                        .scale(self.cs.scale())
                        .unit(Unit::Farad)
                        .build(),
                    rs: self.rs,
                    cs: self.cs,
                    z0: self.z0,
                    freq: self.freq,
                }
            }
        }
    }
}

impl Default for ImpedanceBuilder {
    fn default() -> Self {
        Self {
            kind: ImpedanceType::Gamma,
            category: None,
            mode: ImpedanceMode::Se,
            ri: c64(0.0, 0.0),
            mag: None,
            ang: None,
            rp: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
            cp: UnitValue::new_scaled(0.0, Scale::Base, Unit::Farad),
            rs: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
            cs: UnitValue::new_scaled(0.0, Scale::Base, Unit::Farad),
            freq: UnitValue::new_scaled(1.0, Scale::Giga, Unit::Hz),
            z0: 50.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ImpedanceType {
    Gamma,
    Y,
    #[default]
    Z,
    Rpcp,
    Rscs,
}

impl FromStr for ImpedanceType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "g" | "G" | "gamma" | "Gamma" | "Γ" => Ok(ImpedanceType::Gamma),
            "y" | "Y" | "adm" | "admittance" => Ok(ImpedanceType::Y),
            "z" | "Z" | "imp" | "impedance" => Ok(ImpedanceType::Z),
            "rc" | "RC" | "rpcp" | "RpCp" | "RPCP" | "rescap" | "ResCap" => Ok(ImpedanceType::Rpcp),
            "rscs" | "RsCs" | "RSCS" => Ok(ImpedanceType::Rscs),
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
            ImpedanceType::Rpcp => write!(f, "rpcp"),
            ImpedanceType::Rscs => write!(f, "rscs"),
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
    use crate::util::{comp_f64, comp_num};
    use float_cmp::F64Margin;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-3,
        ulps: 10,
    };
    const CP: f64 = 2.952781875545368e-14;

    fn comp_impedance(exemplar: &Impedance, calc: &Impedance, margin: F64Margin, test_name: &str) {
        assert_eq!(exemplar.mode(), calc.mode());
        comp_num(&exemplar.gamma(), &calc.gamma(), margin, test_name, "gamma");
        comp_num(&exemplar.y(), &calc.y(), margin, test_name, "y");
        comp_num(&exemplar.z(), &calc.z(), margin, test_name, "z");
        comp_f64(
            &exemplar.rp().val(),
            &calc.rp().val(),
            margin,
            test_name,
            "rp",
        );
        assert_eq!(exemplar.rp().scale(), calc.rp().scale());
        comp_f64(
            &exemplar.cp().val(),
            &calc.cp().val(),
            margin,
            test_name,
            "cp",
        );
        assert_eq!(exemplar.cp().scale(), calc.cp().scale());
        comp_f64(
            &exemplar.rs().val(),
            &calc.rs().val(),
            margin,
            test_name,
            "rs",
        );
        assert_eq!(exemplar.rs().scale(), calc.rs().scale());
        comp_f64(
            &exemplar.cs().val(),
            &calc.cs().val(),
            margin,
            test_name,
            "cs",
        );
        assert_eq!(exemplar.cs().scale(), calc.cs().scale());
        comp_f64(&exemplar.z0(), &calc.z0(), margin, test_name, "z0");
        comp_f64(
            &exemplar.freq().val(),
            &calc.freq().val(),
            margin,
            test_name,
            "freq",
        );
    }

    mod impedance_struct_tests {
        use super::*;

        #[test]
        fn test_impedance() {
            let mut test = Impedance::default();
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.02, 0.0),
                    z: c64(50.0, 0.0),
                    g: c64(0.0, 0.0),
                    rp: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "impedance()",
            );

            // --------------------------
            // se_to_diff()
            // --------------------------
            test.set_z(c64(42.4, -19.6));
            test.se_to_diff();
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Diff,
                    y: c64(0.009716213243381976, 0.004491457065336951),
                    z: c64(84.8, -39.2),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(84.8, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                    z0: 100.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "se_to_diff()",
            );

            // --------------------------
            // diff_to_se()
            // --------------------------
            test.diff_to_se();
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "diff_to_se()",
            );

            // --------------------------
            // set_z()
            // --------------------------
            test.set_z(c64(42.4, -19.6));
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_z()",
            );

            // --------------------------
            // set_y()
            // --------------------------
            let mut test = Impedance::default();
            test.set_y(c64(0.01943242648676395, 0.008982914130673902));
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_y()",
            );

            // --------------------------
            // set_gamma()
            // --------------------------
            let mut test = Impedance::default();
            test.set_gamma(c64(-0.035651518955561144, -0.21968365553602814));
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_gamma()",
            );

            // --------------------------
            // set_rp()
            // --------------------------
            let mut test = Impedance::default();
            test.set_rp(&UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm));
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.0),
                    z: c64(51.46037735849057, 0.0),
                    g: c64(0.014393573102242738, 0.0),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_rp()",
            );

            // --------------------------
            // set_cp()
            // --------------------------
            let mut test = Impedance::default();
            test.set_cp(&UnitValue::new(
                5.105982811667098e-15,
                Scale::Femto,
                Unit::Farad,
            ));
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.02, 0.008982914130673902),
                    z: c64(41.60661910298355, -18.687434333487882),
                    g: c64(-0.04801159906098788, -0.21379075147581758),
                    rp: UnitValue::new(50.0, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(41.60661910298355, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(3.041672285766333e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_cp()",
            );

            // --------------------------
            // set_freq()
            // --------------------------
            let mut test = Impedance::new_from_z(
                c64(42.4, -19.6),
                50.0,
                &UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                ImpedanceMode::Se,
            );
            test.set_freq(&UnitValue::new_scaled(27.0, Scale::Giga, Unit::Hz));
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.295093286173287e-14, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(3.007463021388801e-13, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: UnitValue::new_scaled(27.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_freq()",
            );

            // --------------------------
            // set_z0()
            // --------------------------
            let mut test = Impedance::new_from_z(
                c64(42.4, -19.6),
                50.0,
                &UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                ImpedanceMode::Se,
            );
            test.set_z0(74.9);
            comp_impedance(
                &Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.24238004164471896, -0.20759291403441169),
                    rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 74.9,
                    freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                &test,
                F64Margin::default(),
                "set_z0()",
            );
        }
    }

    mod impedancebuilder_tests {
        use super::*;

        mod gamma_tests {
            use super::*;

            #[test]
            fn test_gamma_ri_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .ri(c64(-0.035651518955561144, -0.21968365553602814))
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(RI)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .ri(c64(-0.035651518955561144, -0.21968365553602814))
                    .mag(0.0)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(RI2)",
                );
            }

            #[test]
            fn test_gamma_ri_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .mode(ImpedanceMode::Diff)
                    .ri(c64(-0.035651518955561144, -0.21968365553602814))
                    .z0(100.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(RI)Diff",
                );
            }

            #[test]
            fn test_gamma_ri_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .x(-0.035651518955561144)
                    .y(-0.21968365553602814)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(RI)xy",
                );
            }

            #[test]
            fn test_gamma_ma() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .mag(0.22255772130732962)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(MA)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .category(ComplexNumberType::MagAng)
                    .mag(0.22255772130732962)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(MA)MagAng",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .category(ComplexNumberType::ReIm)
                    .mag(0.1)
                    .ang(2.0)
                    .ri(c64(-0.035651518955561144, -0.21968365553602814))
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(MA)ReIm",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .ri(c64(0.0, 0.0))
                    .mag(0.22255772130732962)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(MA2)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .category(ComplexNumberType::MagAng)
                    .x(0.22255772130732962)
                    .y(-99.21792403733895)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(MA)xy",
                );
            }

            #[test]
            fn test_gamma_db() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .db(-13.051146678449534)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Gamma(dB)",
                );
            }
        }

        mod y_tests {
            use super::*;

            #[test]
            fn test_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Y)
                    .ri(c64(0.01943242648676395, 0.008982914130673902))
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Y",
                );
            }

            #[test]
            fn test_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Y)
                    .mode(ImpedanceMode::Diff)
                    .ri(c64(0.009716213243381976, 0.004491457065336951))
                    .z0(100.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Ydiff",
                );
            }

            #[test]
            fn test_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Y)
                    .x(0.01943242648676395)
                    .y(0.008982914130673902)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Y(xy)",
                );
            }
        }

        mod z_tests {
            use super::*;

            #[test]
            fn test_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Z)
                    .ri(c64(42.4, -19.6))
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Z",
                );
            }

            #[test]
            fn test_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Z)
                    .mode(ImpedanceMode::Diff)
                    .ri(c64(84.8, -39.2))
                    .z0(100.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Zdiff",
                );
            }

            #[test]
            fn test_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Z)
                    .x(42.4)
                    .y(-19.6)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "Z(xy)",
                );
            }
        }

        mod rpcp_tests {
            use super::*;

            #[test]
            fn test_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rpcp)
                    .rp(UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm))
                    .cp(UnitValue::new(
                        5.105982811667098e-15,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RpCp",
                );
            }

            #[test]
            fn test_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rpcp)
                    .mode(ImpedanceMode::Diff)
                    .rp(UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm))
                    .cp(UnitValue::new(
                        2.552991405833549e-15,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(100.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RpCpdiff",
                );
            }

            #[test]
            fn test_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rpcp)
                    .c_scale(Scale::Femto)
                    .x(51.46037735849057)
                    .y(5.105982811667098)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RpCp(xy)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rpcp)
                    .x(51.46037735849057)
                    .y(5.105982811667098e-15)
                    .c_scale(Scale::Femto)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RpCp(xy2)",
                );
            }
        }

        mod rscs_tests {
            use super::*;

            #[test]
            fn test_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rscs)
                    .rs(UnitValue::new(42.4, Scale::Base, Unit::Ohm))
                    .cs(UnitValue::new(
                        2.900053627767772e-14,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RsCs",
                );
            }

            #[test]
            fn test_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rscs)
                    .mode(ImpedanceMode::Diff)
                    .rs(UnitValue::new(84.8, Scale::Base, Unit::Ohm))
                    .cs(UnitValue::new(
                        1.450026813883886e-14,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(100.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RsCsdiff",
                );
            }

            #[test]
            fn test_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Rscs)
                    .x(42.4)
                    .y(2.900053627767772e-14)
                    .c_scale(Scale::Femto)
                    .z0(50.0)
                    .freq(UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build();
                comp_impedance(
                    &Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: UnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: UnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: UnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: UnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: UnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    &test,
                    F64Margin::default(),
                    "RsCs(xy)",
                );
            }
        }
    }

    #[test]
    fn test_impedancetype() {
        let gamma = ["g", "G", "gamma", "Gamma", "Γ"];
        let y = ["y", "Y", "adm", "admittance"];
        let z = ["z", "Z", "imp", "impedance"];
        let rpcp = ["rc", "RC", "rpcp", "RpCp", "RPCP", "rescap", "ResCap"];
        let rscs = ["rscs", "RsCs", "RSCS"];
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
        for val in rpcp.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::Rpcp);
        }
        for val in rscs.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::Rscs);
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
        comp_num(
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
