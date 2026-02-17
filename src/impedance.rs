use crate::{
    math::*,
    num::RealScalar,
    scale::Scale,
    unit::{ScalarUnitValue, Unit, UnitValBuilder},
    util::ApproxEq,
};
use num_complex::{Complex, ComplexFloat};
use serde::Serialize;
use std::{fmt, str::FromStr};

#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct Impedance<T: RealScalar> {
    mode: ImpedanceMode, // differential or single-ended input
    y: Complex<T>,
    z: Complex<T>,
    g: Complex<T>,
    rp: ScalarUnitValue<T>,
    cp: ScalarUnitValue<T>,
    rs: ScalarUnitValue<T>,
    cs: ScalarUnitValue<T>,
    z0: T,
    freq: ScalarUnitValue<T>,
}

impl<T> Impedance<T>
where
    T: RealScalar + ApproxEq<Compare = T>,
{
    pub fn new_from_gamma(
        gamma: Complex<T>,
        z0: T,
        freq: ScalarUnitValue<T>,
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

    pub fn new_from_y(y: Complex<T>, z0: T, freq: ScalarUnitValue<T>, mode: ImpedanceMode) -> Self {
        let mut out = Impedance::default();
        if mode == ImpedanceMode::Diff {
            out.mode = ImpedanceMode::Diff;
        }
        out.set_y(y);
        out.set_z0(z0);
        out.set_freq(freq);

        out
    }

    pub fn new_from_z(z: Complex<T>, z0: T, freq: ScalarUnitValue<T>, mode: ImpedanceMode) -> Self {
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
        rp: ScalarUnitValue<T>,
        cp: ScalarUnitValue<T>,
        z0: T,
        freq: ScalarUnitValue<T>,
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
        rs: ScalarUnitValue<T>,
        cs: ScalarUnitValue<T>,
        z0: T,
        freq: ScalarUnitValue<T>,
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

    pub fn gamma(&self) -> Complex<T> {
        self.g
    }

    pub fn y(&self) -> Complex<T> {
        self.y
    }

    pub fn z(&self) -> Complex<T> {
        self.z
    }

    pub fn rp(&self) -> ScalarUnitValue<T> {
        self.rp
    }

    pub fn cp(&self) -> ScalarUnitValue<T> {
        self.cp
    }

    pub fn rs(&self) -> ScalarUnitValue<T> {
        self.rs
    }

    pub fn cs(&self) -> ScalarUnitValue<T> {
        self.cs
    }

    pub fn z0(&self) -> T {
        self.z0
    }

    pub fn freq(&self) -> ScalarUnitValue<T> {
        self.freq
    }

    pub fn diff_to_se(&mut self) -> &Self {
        if self.mode == ImpedanceMode::Se {
            ()
        } else {
            self.mode = ImpedanceMode::Se;
            self.set_z(self.z / T::C2);
            self.set_z0(self.z0 / T::C2);
        }

        self
    }

    pub fn se_to_diff(&mut self) -> &Self {
        if self.mode == ImpedanceMode::Se {
            self.mode = ImpedanceMode::Diff;
            self.set_z(self.z * T::C2);
            self.set_z0(self.z0 * 2.0.into());
        } else {
            ()
        }

        self
    }

    pub fn set_gamma(&mut self, g: Complex<T>) -> &Self {
        self.g = g;
        self.z = gamma_to_z(self.g, self.z0);
        self.y = z_to_y(self.z);
        let (rp, cp) = z_to_rpcp(self.z, self.freq);
        self.rp.set_val_inplace(rp);
        self.cp.set_val_inplace(cp);
        let (rs, cs) = z_to_rscs(self.z, self.freq);
        self.rs.set_val_inplace(rs);
        self.cs.set_val_inplace(cs);
        self
    }

    pub fn set_y(&mut self, y: Complex<T>) -> &Self {
        self.y = y;
        self.z = y_to_z(self.y);
        self.g = z_to_gamma(self.z, self.z0);
        let (rp, cp) = z_to_rpcp(self.z, self.freq);
        self.rp.set_val_inplace(rp);
        self.cp.set_val_inplace(cp);
        let (rs, cs) = z_to_rscs(self.z, self.freq);
        self.rs.set_val_inplace(rs);
        self.cs.set_val_inplace(cs);
        self
    }

    pub fn set_z(&mut self, z: Complex<T>) -> &Self {
        self.z = z;
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        let (rp, cp) = z_to_rpcp(self.z, self.freq);
        self.rp.set_val_inplace(rp);
        self.cp.set_val_inplace(cp);
        let (rs, cs) = z_to_rscs(self.z, self.freq);
        self.rs.set_val_inplace(rs);
        self.cs.set_val_inplace(cs);
        self
    }

    pub fn set_z0(&mut self, z0: T) -> &Self {
        self.z0 = z0;
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_rp(&mut self, rp: ScalarUnitValue<T>) -> &Self {
        self.rp = rp;
        let (rs, cs) = rpcp_to_rscs(self.rp.val(), self.cp.val(), self.freq);
        self.rs.set_val_inplace(rs);
        self.cs.set_val_inplace(cs);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_cp(&mut self, cp: ScalarUnitValue<T>) -> &Self {
        self.cp = cp;
        let (rs, cs) = rpcp_to_rscs(self.rp.val(), self.cp.val(), self.freq);
        self.rs.set_val_inplace(rs);
        self.cs.set_val_inplace(cs);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_rs(&mut self, rs: ScalarUnitValue<T>) -> &Self {
        self.rs = rs;
        let (rp, cp) = rscs_to_rpcp(self.rs.val(), self.cs.val(), self.freq);
        self.rp.set_val_inplace(rp);
        self.cp.set_val_inplace(cp);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_cs(&mut self, cs: ScalarUnitValue<T>) -> &Self {
        self.cs = cs;
        let (rp, cp) = rscs_to_rpcp(self.rs.val(), self.cs.val(), self.freq);
        self.rp.set_val_inplace(rp);
        self.cp.set_val_inplace(cp);
        self.z = rpcp_to_z(self.rp.val(), self.cp.val(), self.freq);
        self.y = z_to_y(self.z);
        self.g = z_to_gamma(self.z, self.z0);
        self
    }

    pub fn set_freq(&mut self, freq: ScalarUnitValue<T>) -> &Self {
        let z = rpcp_to_z(self.rp.val(), self.cp.val(), self.freq);
        self.freq = freq;
        let (rp, cp) = z_to_rpcp(z, self.freq);
        self.rp.set_val_inplace(rp);
        self.cp.set_val_inplace(cp);
        let (rs, cs) = z_to_rscs(z, self.freq);
        self.rs.set_val_inplace(rs);
        self.cs.set_val_inplace(cs);
        self
    }
}

impl<T: RealScalar> Default for Impedance<T> {
    fn default() -> Self {
        Self {
            mode: ImpedanceMode::Se,
            y: Complex::<T>::new(0.02.into(), T::ZERO),
            z: Complex::<T>::new(50.0.into(), T::ZERO),
            g: Complex::<T>::new(T::ZERO, T::ZERO),
            z0: 50.0.into(),
            freq: ScalarUnitValue::new_scaled(280.0.into(), Scale::Giga, Unit::Hz),
            rp: ScalarUnitValue::new(50.0.into(), Scale::Base, Unit::Ohm),
            cp: ScalarUnitValue::new_scaled(0.0.into(), Scale::Femto, Unit::Farad),
            rs: ScalarUnitValue::new(50.0.into(), Scale::Base, Unit::Ohm),
            cs: ScalarUnitValue::new_scaled(0.0.into(), Scale::Femto, Unit::Farad),
        }
    }
}

#[derive(Debug)]
pub struct ImpedanceBuilder<T: RealScalar> {
    kind: Option<ImpedanceType>,
    category: Option<ComplexNumberType>,
    mode: ImpedanceMode,    // differential or single-ended input
    ri: Option<Complex<T>>, // real/imaginary form of complex input
    mag: Option<T>,
    ang: Option<T>,
    rp: Option<ScalarUnitValue<T>>,
    cp: Option<ScalarUnitValue<T>>,
    rs: Option<ScalarUnitValue<T>>,
    cs: Option<ScalarUnitValue<T>>,
    z0: T,
    freq: Option<ScalarUnitValue<T>>,
}

impl<T> ImpedanceBuilder<T>
where
    T: RealScalar + ApproxEq<Compare = T>,
{
    pub fn new() -> Self {
        ImpedanceBuilder::default()
    }

    pub fn kind(mut self, imp: ImpedanceType) -> Self {
        self.kind = Some(imp);
        self
    }

    pub fn kind_str(mut self, imp: &str) -> Self {
        self.kind = match ImpedanceType::from_str(imp) {
            Ok(x) => Some(x),
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

    pub fn ri(mut self, ri: Complex<T>) -> Self {
        self.ri = Some(ri);
        self
    }

    pub fn re(mut self, re: T) -> Self {
        self.ri = Some(Complex::new(re, self.ri.unwrap_or(Complex::<T>::ZERO).im));
        self
    }

    pub fn im(mut self, im: T) -> Self {
        self.ri = Some(Complex::new(self.ri.unwrap_or(Complex::<T>::ZERO).re, im));
        self
    }

    pub fn mag(mut self, mag: T) -> Self {
        self.mag = Some(mag);
        self
    }

    pub fn db(mut self, db: T) -> Self {
        self.mag = Some(T::C10.powf(db / T::C20));
        self
    }

    pub fn ang(mut self, ang: T) -> Self {
        self.ang = Some(ang);
        self
    }

    // pub fn x(mut self, x: T) -> Self {
    //     match (self.kind, self.category) {
    //         (Some(ImpedanceType::Gamma), None)
    //         | (Some(ImpedanceType::Gamma), Some(ComplexNumberType::ReIm))
    //         | (Some(ImpedanceType::Y), _)
    //         | (Some(ImpedanceType::Z), _) => self.ri = c64(x, self.ri.im),
    //         (Some(ImpedanceType::Gamma), Some(ComplexNumberType::MagAng)) => self.mag = Some(x),
    //         (Some(ImpedanceType::Gamma), Some(ComplexNumberType::Db)) => {
    //             self.mag = Some(10_f64.powf(x / 20.0))
    //         }
    //         (Some(ImpedanceType::RpCp), _) => self.rp = *self.rp.set_val_scaled(x),
    //         (Some(ImpedanceType::RsCs), _) => self.rs = *self.rs.set_val_scaled(x),
    //     }
    //     self
    // }

    // pub fn y(mut self, y: f64) -> Self {
    //     match (self.kind, self.category) {
    //         (Some(ImpedanceType::Gamma), None)
    //         | (Some(ImpedanceType::Gamma), Some(ComplexNumberType::ReIm))
    //         | (Some(ImpedanceType::Y), _)
    //         | (Some(ImpedanceType::Z), _) => self.ri = c64(self.ri.re, y),
    //         (Some(ImpedanceType::Gamma), Some(ComplexNumberType::MagAng)) => self.ang = Some(y),
    //         (Some(ImpedanceType::Gamma), Some(ComplexNumberType::Db)) => {
    //             self.ang = Some(10_f64.powf(y / 20.0))
    //         }
    //         (Some(ImpedanceType::RpCp), _) => self.cp = *self.cp.set_val_scaled(y),
    //         (Some(ImpedanceType::RsCs), _) => self.cs = *self.cs.set_val_scaled(y),
    //     }
    //     self
    // }

    pub fn r_scale(self, scale: Scale) -> Self {
        match self.rp {
            Some(mut x) => x.set_scale(scale),
            None => (),
        }
        match self.rs {
            Some(mut x) => x.set_scale(scale),
            None => (),
        }
        self
    }

    pub fn r_scale_str(self, scale: &str) -> Self {
        match self.rp {
            Some(mut x) => x.set_scale(Scale::from_str(scale).unwrap()),
            None => (),
        }
        match self.rs {
            Some(mut x) => x.set_scale(Scale::from_str(scale).unwrap()),
            None => (),
        }
        self
    }

    pub fn c_scale(self, scale: Scale) -> Self {
        match self.cp {
            Some(mut x) => x.set_scale(scale),
            None => (),
        }
        match self.cs {
            Some(mut x) => x.set_scale(scale),
            None => (),
        }
        self
    }

    pub fn c_scale_str(self, scale: &str) -> Self {
        match self.cp {
            Some(mut x) => x.set_scale(Scale::from_str(scale).unwrap()),
            None => (),
        }
        match self.cs {
            Some(mut x) => x.set_scale(Scale::from_str(scale).unwrap()),
            None => (),
        }
        self
    }

    pub fn rp(mut self, res: ScalarUnitValue<T>) -> Self {
        self.rp = Some(res);
        self
    }

    pub fn rp_val(self, res: T) -> Self {
        match self.rp {
            Some(mut x) => x.set_val_inplace(res),
            None => (),
        }
        self
    }

    pub fn rp_val_scaled(self, res: T) -> Self {
        match self.rp {
            Some(mut x) => x.set_val_scaled_inplace(res),
            None => (),
        }
        self
    }

    pub fn cp(mut self, cap: ScalarUnitValue<T>) -> Self {
        self.cp = Some(cap);
        self
    }

    pub fn cp_val(self, cap: T) -> Self {
        match self.cp {
            Some(mut x) => x.set_val_inplace(cap),
            None => (),
        }
        self
    }

    pub fn cp_val_scaled(self, cap: T) -> Self {
        match self.cp {
            Some(mut x) => x.set_val_scaled_inplace(cap),
            None => (),
        }
        self
    }

    pub fn rs(mut self, res: ScalarUnitValue<T>) -> Self {
        self.rs = Some(res);
        self
    }

    pub fn rs_val(self, res: T) -> Self {
        match self.rs {
            Some(mut x) => x.set_val_inplace(res),
            None => (),
        }
        self
    }

    pub fn rs_val_scaled(self, res: T) -> Self {
        match self.rs {
            Some(mut x) => x.set_val_scaled_inplace(res),
            None => (),
        }
        self
    }

    pub fn cs(mut self, cap: ScalarUnitValue<T>) -> Self {
        self.cs = Some(cap);
        self
    }

    pub fn cs_val(self, cap: T) -> Self {
        match self.cs {
            Some(mut x) => x.set_val_inplace(cap),
            None => (),
        }
        self
    }

    pub fn cs_val_scaled(self, cap: T) -> Self {
        match self.cs {
            Some(mut x) => x.set_val_scaled_inplace(cap),
            None => (),
        }
        self
    }

    pub fn z0(mut self, z0: T) -> Self {
        self.z0 = z0;
        self
    }

    pub fn freq(mut self, freq: ScalarUnitValue<T>) -> Self {
        self.freq = Some(freq);
        self
    }

    pub fn freq_val(self, freq: T) -> Self {
        match self.freq {
            Some(mut x) => x.set_val_inplace(freq),
            None => (),
        }
        self
    }

    pub fn freq_val_scaled(mut self, freq: T, scale: Scale) -> Self {
        self.freq = Some(
            UnitValBuilder::new()
                .val_scaled(&freq, scale)
                .unit(Unit::Hz)
                .build()
                .unwrap(),
        );
        self
    }

    pub fn build(self) -> Result<Impedance<T>, String> {
        match self.kind {
            Some(ImpedanceType::Gamma) => {
                if (self.mag, self.ang, self.ri) == (None, None, None) {
                    return Err("mag/ang or re/im must be set for ImpedanceType::Gamma".into());
                }
            }
            Some(ImpedanceType::Y) => {
                if self.ri == None {
                    return Err("re/im must be set for ImpedanceType::Y".into());
                }
            }
            Some(ImpedanceType::Z) => {
                if self.ri == None {
                    return Err("re/im must be set for ImpedanceType::Z".into());
                }
            }
            Some(ImpedanceType::RpCp) => {
                if (self.rp, self.cp) == (None, None) {
                    return Err("rp/cp must be set for ImpedanceType::RpCp".into());
                }
            }
            Some(ImpedanceType::RsCs) => {
                if (self.rs, self.cs) == (None, None) {
                    return Err("rp/cp must be set for ImpedanceType::RsCs".into());
                }
            }
            None => return Err("kind must be set".into()),
        }
        if self.freq == None {
            return Err("freq must be set".into());
        }
        let freq = self.freq.unwrap();
        match self.kind {
            Some(ImpedanceType::Gamma) => {
                let g = match (self.category, self.mag, self.ang) {
                    (None, Some(mag), Some(ang))
                    | (Some(ComplexNumberType::MagAng), Some(mag), Some(ang))
                    | (Some(ComplexNumberType::Db), Some(mag), Some(ang)) => {
                        Complex::from_polar(mag, ang.to_radians())
                    }
                    _ => self.ri.unwrap(),
                };
                let z = gamma_to_z(g, self.z0);
                let (rp, cp) = z_to_rpcp(z, freq);
                let (rs, cs) = z_to_rscs(z, freq);
                Ok(Impedance {
                    mode: self.mode,
                    y: z.recip(),
                    z,
                    g,
                    rp: ScalarUnitValue::builder()
                        .val(&rp)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cp: ScalarUnitValue::builder()
                        .val(&cp)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    rs: ScalarUnitValue::builder()
                        .val(&rs)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cs: ScalarUnitValue::builder()
                        .val(&cs)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    z0: self.z0,
                    freq: freq,
                })
            }
            Some(ImpedanceType::Y) => {
                let z = self.ri.unwrap().recip();
                let (rp, cp) = z_to_rpcp(z, freq);
                let (rs, cs) = z_to_rscs(z, freq);
                Ok(Impedance {
                    mode: self.mode,
                    y: self.ri.unwrap(),
                    z,
                    g: z_to_gamma(z, self.z0),
                    rp: ScalarUnitValue::builder()
                        .val(&rp)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cp: ScalarUnitValue::builder()
                        .val(&cp)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    rs: ScalarUnitValue::builder()
                        .val(&rs)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cs: ScalarUnitValue::builder()
                        .val(&cs)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    z0: self.z0,
                    freq: freq,
                })
            }
            Some(ImpedanceType::Z) => {
                let (rp, cp) = z_to_rpcp(self.ri.unwrap(), freq);
                let (rs, cs) = z_to_rscs(self.ri.unwrap(), freq);
                Ok(Impedance {
                    mode: self.mode,
                    y: self.ri.unwrap().recip(),
                    z: self.ri.unwrap(),
                    g: z_to_gamma(self.ri.unwrap(), self.z0),
                    rp: ScalarUnitValue::builder()
                        .val(&rp)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cp: ScalarUnitValue::builder()
                        .val(&cp)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    rs: ScalarUnitValue::builder()
                        .val(&rs)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cs: ScalarUnitValue::builder()
                        .val(&cs)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    z0: self.z0,
                    freq: freq,
                })
            }
            Some(ImpedanceType::RpCp) => {
                let z = rpcp_to_z(self.rp.unwrap().val(), self.cp.unwrap().val(), freq);
                let (rs, cs) = z_to_rscs(z, freq);
                let g = z_to_gamma(z, self.z0);
                Ok(Impedance {
                    mode: self.mode,
                    y: z.recip(),
                    z,
                    g,
                    rp: self.rp.unwrap(),
                    cp: self.cp.unwrap(),
                    rs: ScalarUnitValue::builder()
                        .val(&rs)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cs: ScalarUnitValue::builder()
                        .val(&cs)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    z0: self.z0,
                    freq: freq,
                })
            }
            Some(ImpedanceType::RsCs) => {
                let z = rscs_to_z(self.rs.unwrap().val(), self.cs.unwrap().val(), freq);
                let (rp, cp) = z_to_rpcp(z, freq);
                Ok(Impedance {
                    mode: self.mode,
                    y: z.recip(),
                    z,
                    g: z_to_gamma(z, self.z0),
                    rp: ScalarUnitValue::builder()
                        .val(&rp)
                        .scale(Scale::Base)
                        .unit(Unit::Ohm)
                        .build()
                        .unwrap(),
                    cp: ScalarUnitValue::builder()
                        .val(&cp)
                        .scale(Scale::Femto)
                        .unit(Unit::Farad)
                        .build()
                        .unwrap(),
                    rs: self.rs.unwrap(),
                    cs: self.cs.unwrap(),
                    z0: self.z0,
                    freq: freq,
                })
            }
            None => Err("pertinent fields must be set".into()),
        }
    }
}

impl<T: RealScalar> Default for ImpedanceBuilder<T> {
    fn default() -> Self {
        Self {
            kind: None,
            category: None,
            mode: ImpedanceMode::Se,
            ri: None,
            mag: None,
            ang: None,
            rp: None,
            cp: None,
            rs: None,
            cs: None,
            z0: 50.0.into(),
            freq: None,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ImpedanceType {
    Gamma,
    Y,
    #[default]
    Z,
    RpCp,
    RsCs,
}

impl FromStr for ImpedanceType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "g" | "G" | "gamma" | "Gamma" | "Γ" => Ok(ImpedanceType::Gamma),
            "y" | "Y" | "adm" | "admittance" => Ok(ImpedanceType::Y),
            "z" | "Z" | "imp" | "impedance" => Ok(ImpedanceType::Z),
            "rc" | "RC" | "rpcp" | "RpCp" | "RPCP" | "rescap" | "ResCap" => Ok(ImpedanceType::RpCp),
            "rscs" | "RsCs" | "RSCS" => Ok(ImpedanceType::RsCs),
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
            ImpedanceType::RpCp => write!(f, "rpcp"),
            ImpedanceType::RsCs => write!(f, "rscs"),
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
pub struct ComplexNumber<T: RealScalar> {
    kind: ComplexNumberType,
    re: T,
    im: T,
}

impl<T: RealScalar> ComplexNumber<T> {
    pub fn convert(&self, kind: ComplexNumberType) -> ComplexNumber<T> {
        match (self.kind, kind) {
            (ComplexNumberType::ReIm, ComplexNumberType::ReIm) => *self,
            (ComplexNumberType::ReIm, ComplexNumberType::MagAng) => {
                let val = Complex::new(self.re, self.im);
                ComplexNumber {
                    kind: ComplexNumberType::MagAng,
                    re: val.norm(),
                    im: val.arg().to_degrees(),
                }
            }
            (ComplexNumberType::ReIm, ComplexNumberType::Db) => {
                let val = Complex::new(self.re, self.im);
                ComplexNumber {
                    kind: ComplexNumberType::Db,
                    re: T::C20 * val.norm().log10(),
                    im: val.arg().to_degrees(),
                }
            }
            (ComplexNumberType::MagAng, ComplexNumberType::ReIm) => {
                let val = Complex::from_polar(self.re, self.im.to_radians());
                ComplexNumber {
                    kind: ComplexNumberType::ReIm,
                    re: val.re,
                    im: val.im,
                }
            }
            (ComplexNumberType::MagAng, ComplexNumberType::MagAng) => *self,
            (ComplexNumberType::MagAng, ComplexNumberType::Db) => ComplexNumber {
                kind: ComplexNumberType::Db,
                re: T::C20 * self.re.log10(),
                im: self.im,
            },
            (ComplexNumberType::Db, ComplexNumberType::ReIm) => {
                let val =
                    Complex::from_polar(T::C10.powf(self.re / 20.0.into()), self.im.to_radians());
                ComplexNumber {
                    kind: ComplexNumberType::ReIm,
                    re: val.re,
                    im: val.im,
                }
            }
            (ComplexNumberType::Db, ComplexNumberType::MagAng) => ComplexNumber {
                kind: ComplexNumberType::MagAng,
                re: T::C10.powf(self.re / 20.0.into()),
                im: self.im,
            },
            (ComplexNumberType::Db, ComplexNumberType::Db) => *self,
        }
    }

    pub fn ri(&self) -> Complex<T> {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im),
            ComplexNumberType::MagAng => Complex::from_polar(self.re, self.im.to_radians()),
            ComplexNumberType::Db => {
                Complex::from_polar(T::C10.powf(self.re / 20.0.into()), self.im.to_radians())
            }
        }
    }

    pub fn mag(&self) -> T {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im).norm(),
            ComplexNumberType::MagAng => self.re,
            ComplexNumberType::Db => T::C10.powf(self.re / 20.0.into()),
        }
    }

    pub fn db(&self) -> T {
        match self.kind {
            ComplexNumberType::ReIm => T::C20 * Complex::new(self.re, self.im).norm().log10(),
            ComplexNumberType::MagAng => T::C20 * self.re.log10(),
            ComplexNumberType::Db => self.re,
        }
    }

    pub fn ang(&self) -> T {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im).arg().to_degrees(),
            ComplexNumberType::MagAng | ComplexNumberType::Db => self.im,
        }
    }
}

#[derive(Default)]
pub struct ComplexNumberBuilder<T: RealScalar> {
    kind: Option<ComplexNumberType>,
    re: Option<T>,
    im: Option<T>,
}

impl<T: RealScalar> ComplexNumberBuilder<T> {
    pub fn new() -> Self {
        ComplexNumberBuilder::default()
    }

    pub fn kind(mut self, val: ComplexNumberType) -> Self {
        self.kind = Some(val);
        self
    }

    pub fn kind_from_str(mut self, val: &str) -> Self {
        self.kind = Some(ComplexNumberType::from_str(val).unwrap());
        self
    }

    pub fn ri(mut self, val: Complex<T>) -> Self {
        self.re = Some(val.re);
        self.im = Some(val.im);
        self
    }

    pub fn real(mut self, val: T) -> Self {
        self.re = Some(val);
        self
    }

    pub fn imag(mut self, val: T) -> Self {
        self.im = Some(val);
        self
    }

    pub fn mag(mut self, val: T) -> Self {
        self.re = Some(val);
        self
    }

    pub fn db(mut self, val: T) -> Self {
        self.re = Some(val);
        self
    }

    pub fn angle(mut self, val: T) -> Self {
        self.im = Some(val);
        self
    }

    pub fn build(self) -> Result<ComplexNumber<T>, String> {
        let kind = self.kind.ok_or("kind must be set")?;
        let re = self.re.ok_or("re must be set")?;
        let im = self.im.ok_or("im must be set")?;
        Ok(ComplexNumber { kind, re, im })
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
    pub fn parse<T: RealScalar>(&self, x: T, y: T) -> Complex<T> {
        match self {
            ComplexNumberType::ReIm => Complex::new(x, y),
            ComplexNumberType::MagAng => Complex::from_polar(x, y.to_radians()),
            ComplexNumberType::Db => Complex::from_polar(T::C10.powf(x / T::C20), y.to_radians()),
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
    use crate::util::{ApproxEq, NumMargin};
    use num_complex::c64;

    const MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-3,
        relative: 1e-3,
        ulps: 10,
    };
    const CP: f64 = 2.952781875545368e-14;

    fn comp_impedance<T>(
        calc: Impedance<T>,
        exemplar: Impedance<T>,
        margin: NumMargin<T>,
        test_name: &str,
    ) where
        T: RealScalar + ApproxEq<Compare = T>,
    {
        assert_eq!(exemplar.mode(), calc.mode());
        calc.gamma()
            .assert_approx_eq(&exemplar.gamma(), margin, test_name, "gamma");
        calc.y()
            .assert_approx_eq(&exemplar.y(), margin, test_name, "y");
        calc.z()
            .assert_approx_eq(&exemplar.z(), margin, test_name, "z");
        calc.rp()
            .val()
            .assert_approx_eq(&exemplar.rp().val(), margin, test_name, "rp");
        assert_eq!(calc.rp().scale(), exemplar.rp().scale());
        calc.cp()
            .val()
            .assert_approx_eq(&exemplar.cp().val(), margin, test_name, "cp");
        assert_eq!(calc.cp().scale(), exemplar.cp().scale());
        calc.rs()
            .val()
            .assert_approx_eq(&exemplar.rs().val(), margin, test_name, "rs");
        assert_eq!(calc.rs().scale(), exemplar.rs().scale());
        calc.cs()
            .val()
            .assert_approx_eq(&exemplar.cs().val(), margin, test_name, "cs");
        assert_eq!(calc.cs().scale(), exemplar.cs().scale());
        calc.z0()
            .assert_approx_eq(&exemplar.z0(), margin, test_name, "z0");
        calc.freq()
            .val()
            .assert_approx_eq(&exemplar.freq().val(), margin, test_name, "freq");
    }

    mod impedance_struct_tests {
        use super::*;

        #[test]
        fn test_impedance() {
            let mut test = Impedance::default();
            comp_impedance(
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.02, 0.0),
                    z: c64(50.0, 0.0),
                    g: c64(0.0, 0.0),
                    rp: ScalarUnitValue::new(50.0, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(50.0, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                test,
                NumMargin::default(),
                "impedance()",
            );

            // --------------------------
            // se_to_diff()
            // --------------------------
            test.set_z(c64(42.4, -19.6));
            test.se_to_diff();
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Diff,
                    y: c64(0.009716213243381976, 0.004491457065336951),
                    z: c64(84.8, -39.2),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: ScalarUnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                    z0: 100.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "se_to_diff()",
            );

            // --------------------------
            // diff_to_se()
            // --------------------------
            test.diff_to_se();
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "diff_to_se()",
            );

            // --------------------------
            // set_z()
            // --------------------------
            test.set_z(c64(42.4, -19.6));
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "set_z()",
            );

            // --------------------------
            // set_y()
            // --------------------------
            let mut test = Impedance::default();
            test.set_y(c64(0.01943242648676395, 0.008982914130673902));
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "set_y()",
            );

            // --------------------------
            // set_gamma()
            // --------------------------
            let mut test = Impedance::default();
            test.set_gamma(c64(-0.035651518955561144, -0.21968365553602814));
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "set_gamma()",
            );

            // --------------------------
            // set_rp()
            // --------------------------
            let mut test = Impedance::default();
            test.set_rp(ScalarUnitValue::new(
                51.46037735849057,
                Scale::Base,
                Unit::Ohm,
            ));
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.0),
                    z: c64(51.46037735849057, 0.0),
                    g: c64(0.014393573102242738, 0.0),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(0.0, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "set_rp()",
            );

            // --------------------------
            // set_cp()
            // --------------------------
            let mut test = Impedance::default();
            test.set_cp(ScalarUnitValue::new(
                5.105982811667098e-15,
                Scale::Femto,
                Unit::Farad,
            ));
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.02, 0.008982914130673902),
                    z: c64(41.60661910298355, -18.687434333487882),
                    g: c64(-0.04801159906098788, -0.21379075147581758),
                    rp: ScalarUnitValue::new(50.0, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(41.60661910298355, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(3.041672285766333e-14, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "set_cp()",
            );

            // --------------------------
            // set_freq()
            // --------------------------
            let mut test = Impedance::new_from_z(
                c64(42.4, -19.6),
                50.0,
                ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                ImpedanceMode::Se,
            );
            test.set_freq(ScalarUnitValue::new_scaled(27.0, Scale::Giga, Unit::Hz));
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.035651518955561144, -0.21968365553602814),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.295093286173287e-14, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(3.007463021388801e-13, Scale::Femto, Unit::Farad),
                    z0: 50.0,
                    freq: ScalarUnitValue::new_scaled(27.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
                "set_freq()",
            );

            // --------------------------
            // set_z0()
            // --------------------------
            let mut test = Impedance::new_from_z(
                c64(42.4, -19.6),
                50.0,
                ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                ImpedanceMode::Se,
            );
            test.set_z0(74.9);
            comp_impedance(
                test,
                Impedance {
                    mode: ImpedanceMode::Se,
                    y: c64(0.01943242648676395, 0.008982914130673902),
                    z: c64(42.4, -19.6),
                    g: c64(-0.24238004164471896, -0.20759291403441169),
                    rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                    cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                    rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                    cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                    z0: 74.9,
                    freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                },
                NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Gamma(RI)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .ri(c64(-0.035651518955561144, -0.21968365553602814))
                    .mag(0.0)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Gamma(RI)Diff",
                );
            }

            #[test]
            fn test_gamma_ri_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .re(-0.035651518955561144)
                    .im(-0.21968365553602814)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Gamma(MA)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .category(ComplexNumberType::MagAng)
                    .mag(0.22255772130732962)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Gamma(MA)MagAng",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .category(ComplexNumberType::ReIm)
                    .mag(0.1)
                    .ang(2.0)
                    .ri(c64(-0.035651518955561144, -0.21968365553602814))
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Gamma(MA)ReIm",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .ri(c64(0.0, 0.0))
                    .mag(0.22255772130732962)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Gamma(MA2)",
                );

                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Gamma)
                    .category(ComplexNumberType::MagAng)
                    .mag(0.22255772130732962)
                    .ang(-99.21792403733895)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Ydiff",
                );
            }

            #[test]
            fn test_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Y)
                    .re(0.01943242648676395)
                    .im(0.008982914130673902)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
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
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Zdiff",
                );
            }

            #[test]
            fn test_xy() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::Z)
                    .re(42.4)
                    .im(-19.6)
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .c_scale(Scale::Femto)
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "Z(xy)",
                );
            }
        }

        mod rpcp_tests {
            use super::*;

            #[test]
            fn test_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::RpCp)
                    .rp(ScalarUnitValue::new(
                        51.46037735849057,
                        Scale::Base,
                        Unit::Ohm,
                    ))
                    .cp(ScalarUnitValue::new(
                        5.105982811667098e-15,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "RpCp",
                );
            }

            #[test]
            fn test_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::RpCp)
                    .mode(ImpedanceMode::Diff)
                    .rp(ScalarUnitValue::new(
                        102.92075471698114,
                        Scale::Base,
                        Unit::Ohm,
                    ))
                    .cp(ScalarUnitValue::new(
                        2.552991405833549e-15,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(100.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "RpCpdiff",
                );
            }
        }

        mod rscs_tests {
            use super::*;

            #[test]
            fn test_se() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::RsCs)
                    .rs(ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm))
                    .cs(ScalarUnitValue::new(
                        2.900053627767772e-14,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(50.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Se,
                        y: c64(0.01943242648676395, 0.008982914130673902),
                        z: c64(42.4, -19.6),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(51.46037735849057, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(5.105982811667098e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(42.4, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(2.900053627767772e-14, Scale::Femto, Unit::Farad),
                        z0: 50.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "RsCs",
                );
            }

            #[test]
            fn test_diff() {
                let test = ImpedanceBuilder::new()
                    .kind(ImpedanceType::RsCs)
                    .mode(ImpedanceMode::Diff)
                    .rs(ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm))
                    .cs(ScalarUnitValue::new(
                        1.450026813883886e-14,
                        Scale::Femto,
                        Unit::Farad,
                    ))
                    .z0(100.0)
                    .freq(ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz))
                    .build()
                    .unwrap();
                comp_impedance(
                    test,
                    Impedance {
                        mode: ImpedanceMode::Diff,
                        y: c64(0.009716213243381976, 0.004491457065336951),
                        z: c64(84.8, -39.2),
                        g: c64(-0.035651518955561144, -0.21968365553602814),
                        rp: ScalarUnitValue::new(102.92075471698114, Scale::Base, Unit::Ohm),
                        cp: ScalarUnitValue::new(2.552991405833549e-15, Scale::Femto, Unit::Farad),
                        rs: ScalarUnitValue::new(84.8, Scale::Base, Unit::Ohm),
                        cs: ScalarUnitValue::new(1.450026813883886e-14, Scale::Femto, Unit::Farad),
                        z0: 100.0,
                        freq: ScalarUnitValue::new_scaled(280.0, Scale::Giga, Unit::Hz),
                    },
                    NumMargin::default(),
                    "RsCsdiff",
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
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::RpCp);
        }
        for val in rscs.iter() {
            assert_eq!(ImpedanceType::from_str(val).unwrap(), ImpedanceType::RsCs);
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
            .build()
            .unwrap();
        assert_eq!(test.convert(ComplexNumberType::ReIm), exemplar_ri);
        assert_eq!(
            test.convert(ComplexNumberType::MagAng).kind,
            exemplar_magang.kind
        );
        test.convert(ComplexNumberType::MagAng)
            .mag()
            .assert_approx_eq(
                &exemplar_magang.re,
                NumMargin::default(),
                "convert",
                "ri_to_ma",
            );
        test.convert(ComplexNumberType::MagAng)
            .ang()
            .assert_approx_eq(
                &exemplar_magang.im,
                NumMargin::default(),
                "convert",
                "ri_to_ma",
            );
        assert_eq!(test.convert(ComplexNumberType::Db).kind, exemplar_db.kind);
        test.convert(ComplexNumberType::Db).db().assert_approx_eq(
            &exemplar_db.re,
            NumMargin::default(),
            "convert",
            "ri_to_db",
        );
        test.convert(ComplexNumberType::Db).ang().assert_approx_eq(
            &exemplar_db.im,
            NumMargin::default(),
            "convert",
            "ri_to_db",
        );

        let test = ComplexNumberBuilder::new()
            .mag(mag)
            .angle(ang)
            .kind(ComplexNumberType::MagAng)
            .build()
            .unwrap();
        assert_eq!(test.convert(ComplexNumberType::ReIm).kind, exemplar_ri.kind);
        test.convert(ComplexNumberType::ReIm).ri().assert_approx_eq(
            &val,
            NumMargin::default(),
            "convert",
            "ma_to_ri",
        );
        assert_eq!(test.convert(ComplexNumberType::MagAng), exemplar_magang);
        assert_eq!(test.convert(ComplexNumberType::Db).kind, exemplar_db.kind);
        test.convert(ComplexNumberType::Db).re.assert_approx_eq(
            &exemplar_db.re,
            NumMargin::default(),
            "convert",
            "ma_to_db",
        );
        test.convert(ComplexNumberType::Db).im.assert_approx_eq(
            &exemplar_db.im,
            NumMargin::default(),
            "convert",
            "ma_to_db",
        );

        let test = ComplexNumberBuilder::new()
            .db(db)
            .angle(ang)
            .kind(ComplexNumberType::Db)
            .build()
            .unwrap();
        assert_eq!(test.convert(ComplexNumberType::ReIm).kind, exemplar_ri.kind);
        test.convert(ComplexNumberType::ReIm).re.assert_approx_eq(
            &exemplar_ri.re,
            NumMargin::default(),
            "convert",
            "db_to_ri",
        );
        test.convert(ComplexNumberType::ReIm).im.assert_approx_eq(
            &exemplar_ri.im,
            NumMargin::default(),
            "convert",
            "db_to_ri",
        );
        assert_eq!(
            test.convert(ComplexNumberType::MagAng).kind,
            exemplar_magang.kind
        );
        test.convert(ComplexNumberType::MagAng).re.assert_approx_eq(
            &exemplar_magang.re,
            NumMargin::default(),
            "convert",
            "db_to_ma",
        );
        test.convert(ComplexNumberType::MagAng).im.assert_approx_eq(
            &exemplar_magang.im,
            NumMargin::default(),
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
            .build()
            .unwrap();
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
            .build()
            .unwrap();
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
            .build()
            .unwrap();
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
