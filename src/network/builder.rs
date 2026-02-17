use crate::{
    consts::MathConst,
    frequency::Frequency,
    impedance::ComplexNumberType,
    math::*,
    network::{Network, NetworkPoint, PortVal, WaveType, network_err_msg},
    num::{ComplexScalar, RealScalar},
    parameter::RFParameter,
    pts::{Points, Points1, Points2, Points3, Pts},
    unit::{Unit, UnitValue},
};
use ndarray::{OwnedRepr, prelude::*};
use ndarray_linalg::*;
use num_complex::{Complex, ComplexFloat};
use num_traits::{ConstZero, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use simple_error::SimpleError;
use std::{
    error::Error,
    f64::consts::PI,
    fmt, fs,
    iter::Iterator,
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
    process,
    process::Child,
    slice::Iter,
};

/// Builder design pattern for Network
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
#[derive(Clone, Default)]
pub struct NetworkBuilder<T, U>
where
    T: ComplexScalar,
    <T as ComplexFloat>::Real: RealScalar,
    U: UnitValue + Frequency<U>,
{
    name: Option<String>,
    comments: Option<String>,
    nports: Option<usize>,
    port_names: Option<Vec<String>>,
    ports: Option<Vec<PortVal>>,
    freq: Option<U>,
    npts: Option<usize>,
    dim: Option<(usize, usize, usize)>,
    z0: Option<Vec<T>>,
    param: Option<RFParameter>,
    net: Option<Points3<T>>,
    // a: Option<Points3<Complex<T>>>,
    // g: Option<Points3<Complex<T>>>,
    // h: Option<Points3<Complex<T>>>,
    // s: Option<Points3<Complex<T>>>,
    // s_power: Option<Points3<Complex<T>>>,
    // s_pseudo: Option<Points3<Complex<T>>>,
    // s_traveling: Option<Points3<Complex<T>>>,
    // t: Option<Points3<Complex<T>>>,
    // y: Option<Points3<Complex<T>>>,
    // z: Option<Points3<Complex<T>>>,
}

impl<T, U> NetworkBuilder<T, U>
where
    T: ComplexScalar,
    <T as ComplexFloat>::Real: RealScalar,
    U: UnitValue + Frequency<U>,
{
    pub fn new() -> Self {
        NetworkBuilder::default()
    }

    /// Provide name of Network
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Provide coments for Network
    pub fn comments(mut self, comments: &str) -> Self {
        self.comments = Some(comments.to_string());
        self
    }

    /// Provide names of ports for Network
    pub fn port_names(mut self, names: &Vec<&str>) -> Self {
        let mut port_names: Vec<String> = vec![];

        for name in names {
            port_names.push(name.to_string());
        }
        self.port_names = Some(port_names);

        self
    }

    /// Provide Frequency for Network
    pub fn freq(mut self, freq: &U) -> Self {
        self.freq = Some(freq.clone());

        self
    }

    /// Provide Z0 for Network
    pub fn z0(mut self, z0: &Vec<T>) -> Self {
        self.z0 = Some(z0.clone());
        self
    }

    // fn set_port_names(mut self) -> Self {
    //     if self.port_names.is_none() {
    //         self.port_names = Some(vec!["".to_string(); self.nports])
    //     }
    //     self
    // }

    // /// Provide ABCD parameters representation of Network
    // pub fn a(mut self, net: &Array3<Complex<T>>) -> Self {
    //     self.param = RFParameter::A;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide inverse hybrid parameters representation of Network
    // pub fn g(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::G;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide hybrid parameters representation of Network
    // pub fn h(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::H;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide S-parameter representation of Network
    // pub fn s(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::S;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide Power S-parameter representation of Network
    // pub fn s_power(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::SPower;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide Pseudo S-parameter representation of Network
    // pub fn s_pseudo(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::SPseudo;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide Traveling S-parameter representation of Network
    // pub fn s_traveling(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::STraveling;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide cascade parameters representation of Network
    // pub fn t(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::T;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide admittance parameters representation of Network
    // pub fn y(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::Y;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    // /// Provide impedance parameters representation of Network
    // pub fn z(mut self, net: Points<Complex64, Ix3>) -> Self {
    //     self.param = RFParameter::Z;
    //     self.nports = net.slice(s![0, .., ..]).nrows();
    //     self.net = Some(net);
    //     self.set_port_names()
    // }

    /// Provide RF parameters representation of Network
    pub fn net(mut self, net: &Points3<T>, param: RFParameter) -> Self {
        self.param = Some(param);
        self.nports = Some(net.slice(s![0, .., ..]).nrows());
        self.npts = Some(net.slice(s![.., 0, 0]).len());
        self.dim = Some(net.dim());
        self.net = Some(net.clone());
        self
    }

    pub fn build(mut self) -> Result<Network<T, U>, String> {
        let name = self.name.unwrap_or("".into());
        let comments = self.comments.unwrap_or("".into());
        let nports = self.nports.ok_or("number of ports must be defined")?;
        let port_names = self.port_names.unwrap_or({
            let mut names = vec![];
            for i in 0..nports {
                names.push(format!("Port{}", i + 1));
            }
            names
        });
        let freq = self.freq.ok_or("frequency must be defined")?;
        let npts = self.npts.ok_or("npts must be defined")?;
        let dim = self.dim.ok_or("dim must be defined")?;
        let z0 = self.z0.ok_or("z0 must be defined")?;
        let param = self.param.ok_or("param must be defined")?;
        let net = self.net.ok_or("net must be defined")?;

        if !net.slice(s![0, .., ..]).is_square() {
            return Err(format!(
                "Provided data is not square!\n{rows} rows\t\t{cols} cols",
                rows = net.slice(s![0, .., 0]).len(),
                cols = net.slice(s![0, 0, ..]).len()
            ));
        }
        if net.slice(s![0, .., 0]).len() != z0.len() {
            return Err(format!(
                "Number of ports and number of Z0 points do not match!\n{npts} network points\t\t{zpts} port impedance points",
                npts = net.slice(s![0, .., 0]).len(),
                zpts = z0.len()
            ));
        }
        if freq.npts() != net.slice(s![.., 0, 0]).len() {
            return Err(format!(
                "Number of frequency points does not match number of data points!\n{fpts} frequency points\t\t{npts} network points",
                fpts = freq.npts(),
                npts = net.slice(s![.., 0, 0]).len()
            ));
        }

        let mut ports: Vec<PortVal> = vec![];
        for i in 0..net.slice(s![0, .., ..]).nrows() {
            for j in 0..net.slice(s![0, .., ..]).ncols() {
                ports.push((i, j));
            }
        }

        let wave_type = WaveType::Power;
        let mut a: Option<Points3<T>> = None;
        let mut g: Option<Points3<T>> = None;
        let mut h: Option<Points3<T>> = None;
        let mut s: Option<Points3<T>> = None;
        let mut s_power: Option<Points3<T>> = None;
        let mut s_pseudo: Option<Points3<T>> = None;
        let mut s_traveling: Option<Points3<T>> = None;
        let mut t: Option<Points3<T>> = None;
        let mut y: Option<Points3<T>> = None;
        let mut z: Option<Points3<T>> = None;

        if nports == 2 {
            match param {
                RFParameter::A => {
                    a = Some(net.clone());
                    g = net.clone().a_to_g();
                    h = net.clone().a_to_h();
                    s = net.clone().a_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().a_to_t(&z0);
                    y = net.clone().a_to_y();
                    z = net.clone().a_to_z();
                }
                RFParameter::G => {
                    a = net.clone().g_to_a();
                    g = Some(net.clone());
                    h = net.clone().g_to_h();
                    s = net.clone().g_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().g_to_t(&z0);
                    y = net.clone().g_to_y();
                    z = net.clone().g_to_z();
                }
                RFParameter::H => {
                    a = net.clone().h_to_a();
                    g = net.clone().h_to_g();
                    h = Some(net.clone());
                    s = net.clone().h_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().h_to_t(&z0);
                    y = net.clone().h_to_y();
                    z = net.clone().h_to_z();
                }
                RFParameter::S | RFParameter::SPower => {
                    a = net.clone().s_to_a(&z0);
                    g = net.clone().s_to_g(&z0);
                    h = net.clone().s_to_h(&z0);
                    s = Some(net.clone());
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().s_to_t();
                    y = net.clone().s_to_y(&z0);
                    z = net.clone().s_to_z(&z0);
                }
                RFParameter::SPseudo => {
                    let wave_type = WaveType::Pseudo;
                    a = net.clone().s_to_a(&z0);
                    g = net.clone().s_to_g(&z0);
                    h = net.clone().s_to_h(&z0);
                    s = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Power);
                    s_power = s.clone();
                    s_pseudo = Some(net.clone());
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().s_to_t();
                    y = net.clone().s_to_y(&z0);
                    z = net.clone().s_to_z(&z0);
                }
                RFParameter::STraveling => {
                    let wave_type = WaveType::Traveling;
                    a = net.clone().s_to_a(&z0);
                    g = net.clone().s_to_g(&z0);
                    h = net.clone().s_to_h(&z0);
                    s = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Power);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = Some(net.clone());
                    t = net.clone().s_to_t();
                    y = net.clone().s_to_y(&z0);
                    z = net.clone().s_to_z(&z0);
                }
                RFParameter::T => {
                    a = net.clone().t_to_a(&z0);
                    g = net.clone().t_to_g(&z0);
                    h = net.clone().t_to_h(&z0);
                    s = net.clone().t_to_s();
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = Some(net.clone());
                    y = net.clone().t_to_y(&z0);
                    z = net.clone().t_to_z(&z0);
                }
                RFParameter::Y => {
                    a = net.clone().y_to_a();
                    g = net.clone().y_to_g();
                    h = net.clone().y_to_h();
                    s = net.clone().y_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().y_to_t(&z0);
                    y = Some(net.clone());
                    z = net.clone().y_to_z();
                }
                RFParameter::Z => {
                    a = net.clone().z_to_a();
                    g = net.clone().z_to_g();
                    h = net.clone().z_to_h();
                    s = net.clone().z_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    t = net.clone().z_to_t(&z0);
                    y = net.clone().z_to_y();
                    z = Some(net.clone());
                }
            }
        } else {
            match param {
                RFParameter::A | RFParameter::G | RFParameter::H | RFParameter::T => {
                    panic!("{}", network_err_msg(param, nports))
                }
                RFParameter::S | RFParameter::SPower => {
                    s = Some(net.clone());
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    y = net.clone().s_to_y(&z0);
                    z = net.clone().s_to_z(&z0);
                }
                RFParameter::SPseudo => {
                    let wave_type = WaveType::Pseudo;
                    s = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Power);
                    s_power = s.clone();
                    s_pseudo = Some(net.clone());
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    y = net.clone().s_to_y(&z0);
                    z = net.clone().s_to_z(&z0);
                }
                RFParameter::STraveling => {
                    let wave_type = WaveType::Traveling;
                    s = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Power);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = Some(net.clone());
                    y = net.clone().s_to_y(&z0);
                    z = net.clone().s_to_z(&z0);
                }
                RFParameter::Y => {
                    s = net.clone().y_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    y = Some(net.clone());
                    z = net.clone().y_to_z();
                }
                RFParameter::Z => {
                    s = net.clone().z_to_s(&z0);
                    s_power = s.clone();
                    s_pseudo = s.clone().unwrap().s_to_s(&z0, wave_type, WaveType::Pseudo);
                    s_traveling = s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Traveling);
                    y = net.clone().z_to_y();
                    z = Some(net.clone());
                }
            }
        }

        Ok(Network {
            name,
            comments,
            nports,
            port_names,
            ports,
            freq,
            npts,
            dim,
            z0,
            a,
            g,
            h,
            s,
            s_power,
            s_pseudo,
            s_traveling,
            t,
            y,
            z,
        })
    }
}

// impl Default for NetworkBuilder {
//     fn default() -> Self {
//         NetworkBuilder {
//             name: "".to_string(),
//             comments: "".to_string(),
//             nports: 0,
//             port_names: None,
//             ports: vec![],
//             freq: Frequency::default(),
//             npts: 0,
//             dim: (0, 0, 0),
//             z0: array![c64(50.0, 0.0)],
//             param: RFParameter::S,
//             net: None,
//             a: None,
//             g: None,
//             h: None,
//             s: None,
//             s_power: None,
//             s_pseudo: None,
//             s_traveling: None,
//             t: None,
//             y: None,
//             z: None,
//         }
//     }
// }

#[cfg(test)]
mod network_builder_tests {
    use super::*;
    use crate::{frequency::FrequencyBuilder, impedance::*, scale::*, unit::*, util::*};
    use num_complex::{Complex64, c64};
    use std::{collections::HashMap, hash::Hash};

    const MARGIN: NumMargin<f64> = NumMargin {
        // epsilon: f64::EPSILON,
        epsilon: 1e-12,
        relative: 1.0,
        ulps: 10,
    };

    fn compare_2ports(
        calc: &Network<Complex64, ArrayUnitValue<f64>>,
        exemplars: HashMap<RFParameter, Points<Complex64, Ix3>>,
    ) {
        let margin = MARGIN;
        let mut new_net = calc.clone();
        let x = Points::<Complex64, Ix3>::new(array![
            [
                [c64(1.0, 0.0), c64(2.0, 0.0)],
                [c64(3.0, 0.0), c64(4.0, 0.0)]
            ],
            [
                [c64(1.0, 0.0), c64(2.0, 0.0)],
                [c64(3.0, 0.0), c64(4.0, 0.0)]
            ],
            [
                [c64(1.0, 0.0), c64(2.0, 0.0)],
                [c64(3.0, 0.0), c64(4.0, 0.0)]
            ]
        ]);

        let param = RFParameter::A;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.a(), margin, "a()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.a(), margin, "a(x)");
        new_net.set_net(calc.a(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.a(),
            margin,
            "a(new_net)",
        );

        let param = RFParameter::G;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.g(), margin, "g()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.g(), margin, "g(x)");
        new_net.set_net(calc.g(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.g(),
            margin,
            "g(new_net)",
        );

        let param = RFParameter::H;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.h(), margin, "h()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.h(), margin, "h(x)");
        new_net.set_net(calc.h(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.h(),
            margin,
            "h(new_net)",
        );

        let param = RFParameter::S;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.s(), margin, "s()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.s(), margin, "s(x)");
        new_net.set_net(calc.s(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.s(),
            margin,
            "s(new_net)",
        );

        let param = RFParameter::SPower;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.s_power(),
            margin,
            "s_power()",
        );
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.s_power(), margin, "s_power(x)");
        new_net.set_net(calc.s_power(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.s_power(),
            margin,
            "s_power(new_net)",
        );

        let param = RFParameter::SPseudo;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.s_pseudo(),
            margin,
            "s_pseudo()",
        );
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.s_pseudo(), margin, "s_pseudo(x)");
        new_net.set_net(calc.s_pseudo(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.s_pseudo(),
            margin,
            "s_pseudo(new_net)",
        );

        let param = RFParameter::STraveling;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.s_traveling(),
            margin,
            "s_traveling()",
        );
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.s_traveling(), margin, "s_traveling(x)");
        new_net.set_net(calc.s_traveling(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.s_traveling(),
            margin,
            "s_traveling(new_net)",
        );

        let param = RFParameter::T;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.t(), margin, "t()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.t(), margin, "t(x)");
        new_net.set_net(calc.t(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.t(),
            margin,
            "t(new_net)",
        );

        let param = RFParameter::Y;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.y(), margin, "y()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.y(), margin, "y(x)");
        new_net.set_net(calc.y(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.y(),
            margin,
            "y(new_net)",
        );

        let param = RFParameter::Z;
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_pts_ix3(exemplars.get(&param).unwrap(), calc.z(), margin, "z()");
        new_net.set_net(&x, param);
        comp_pts_ix3(&x, new_net.z(), margin, "z(x)");
        new_net.set_net(calc.z(), param);
        comp_pts_ix3(
            exemplars.get(&param).unwrap(),
            new_net.z(),
            margin,
            "z(new_net)",
        );
    }

    #[test]
    fn network_builder() {
        let name = String::from("title");
        let comments = String::from("here are some comments\nand some more");
        let fdata = array![1.0, 2.0, 3.0];
        let z0 = vec![c64(50.0, 0.0), c64(50.0, 0.0)];
        let ndata = points![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)]
            ],
            [
                [c64(0.849, -0.496), c64(-6.688, 2.413)],
                [c64(0.014, 0.041), c64(0.494, -0.248)]
            ],
            [
                [c64(0.700, -0.648), c64(-0.724, 0.396)],
                [c64(0.026, 0.053), c64(0.435, -0.311)]
            ],
        ];
        let freq = ArrayUnitValue::builder()
            .freq_scaled(&fdata, Scale::Giga)
            .build()
            .unwrap();
        let nports: usize = 2;
        let npoints: usize = 3;
        let mut net = NetworkBuilder::<Complex64, ArrayUnitValue<f64>>::new()
            .freq(&freq)
            .z0(&z0)
            .net(&ndata, RFParameter::S)
            .name("title")
            .comments("here are some comments\nand some more")
            .build()
            .unwrap();

        assert_eq!(name, *net.name());
        assert_eq!(comments, *net.comments());
        assert_eq!(nports, net.nports());
        assert_eq!(npoints, net.npts());
        for i in 0..2 {
            assert_eq!(z0[i], net.z0()[i]);
            assert_eq!(z0[i], net.z0_at_port_idx(i).clone());
            assert_eq!(String::from(format!("Port{}", i + 1)), *net.port_name(i));
        }
        net.set_port_name(0, String::from("Port1"));
        net.set_port_name(1, String::from("Port2"));
        assert_eq!(String::from("Port1"), *net.port_name(0));
        assert_eq!(String::from("Port2"), *net.port_name(1));
        for i in 0..npoints {
            for j in 0..nports {
                for k in 0..nports {
                    assert_eq!(ndata[[i, j, k]], net.s()[[i, j, k]]);
                    assert_eq!(ndata[[i, j, k]], net.net(RFParameter::S)[[i, j, k]]);
                    assert_eq!(
                        ndata[[i, j, k]],
                        net.net_at_freq_idx(RFParameter::S, i)[[j, k]]
                    );
                    assert_eq!(
                        ndata[[i, j, k]],
                        net.net_at_port_idx(RFParameter::S, j, k)[i]
                    );
                    assert_eq!(ndata[[i, j, k]], net.net_at_idx(RFParameter::S, i, j, k));
                }
            }
        }

        let z0 = 50.0;
        let freq = ScalarUnitValue::builder()
            .val_scaled(&275.0, Scale::Giga)
            .build()
            .unwrap();
        let src = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(42.4)
            .im(-19.6)
            .z0(z0)
            .freq(freq)
            .build()
            .unwrap();
        let load = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(212.3)
            .im(43.2)
            .z0(z0)
            .freq(freq)
            .build()
            .unwrap();
        let net = NetworkBuilder::new()
            .freq(
                &ArrayUnitValue::builder()
                    .freq_scaled(&array![275.0], Scale::Giga)
                    .build()
                    .unwrap(),
            )
            .z0(&vec![src.z(), load.z()])
            .net(
                &points![[
                    [c64(0.0, 0.0), c64(1.0, 0.0)],
                    [c64(1.0, 0.0), c64(0.0, 0.0)]
                ]],
                RFParameter::S,
            )
            .build()
            .unwrap();
        println!("net: {:?}", net);
        assert_eq!(1, net.npts());
    }
}
