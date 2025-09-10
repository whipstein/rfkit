use crate::frequency::Frequency;
use crate::impedance::ComplexNumberType;
use crate::math::*;
use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use crate::network::{Network, NetworkPoint, PortVal, WaveType, network_err_msg};
use crate::parameter::RFParameter;
use crate::point::{Point, PointComplex, Pt};
use crate::points::{Points, Pointsf64, Pts};
use crate::unit::Unit;
use ndarray::OwnedRepr;
use ndarray::prelude::*;
use ndarray_linalg::*;
use num::complex::{Complex64, c64};
use num::zero;
use num_traits::{ConstZero, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use rug::az::UnwrappedAs;
use rug::ops::{Pow, PowAssign};
use rug::{Complex, Float};
use simple_error::SimpleError;
use std::error::Error;
use std::f64::consts::PI;
use std::iter::Iterator;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};
use std::process::Child;
use std::slice::Iter;
use std::{fmt, fs, mem, process};

/// Builder design pattern for Network
///
/// ## Example
/// ```
/// use ndarray::prelude::*;
/// use rfkit_base_ndarray::prelude::*;
///
/// let freq1 = FrequencyBuilder::new().freqs_scaled(array![1.0, 2.0, 3.0], Unit::Giga).build();
///
/// let freq2 = FrequencyBuilder::new().start_stop_step_scaled(1.0, 3.0, 1.0, Unit::Giga).build();
/// ```
#[derive(Clone)]
pub struct NetworkBuilder {
    name: String,
    comments: String,
    nports: usize,
    port_names: Option<Vec<String>>,
    ports: Vec<PortVal>,
    freq: Frequency,
    npts: usize,
    dim: (usize, usize, usize),
    z0: Array1<Complex64>,
    param: RFParameter,
    net: Option<Points>,
    a: Option<Points>,
    g: Option<Points>,
    h: Option<Points>,
    s: Option<Points>,
    s_power: Option<Points>,
    s_pseudo: Option<Points>,
    s_traveling: Option<Points>,
    t: Option<Points>,
    y: Option<Points>,
    z: Option<Points>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        NetworkBuilder::default()
    }

    /// Provide name of Network
    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Provide coments for Network
    pub fn comments(mut self, comments: &str) -> Self {
        self.comments = comments.to_string();
        self
    }

    /// Provide names of ports for Network
    pub fn port_names(mut self, names: Vec<&str>) -> Self {
        let mut port_names: Vec<String> = vec![];

        for name in names {
            port_names.push(name.to_string());
        }
        self.port_names = Some(port_names);

        self
    }

    /// Provide Frequency for Network
    pub fn freq(mut self, freq: Frequency) -> Self {
        self.freq = freq;

        self
    }

    /// Provide Z0 for Network
    pub fn z0(mut self, z0: Array1<Complex64>) -> Self {
        self.z0 = z0;
        self
    }

    /// Provide Z0 for Network
    pub fn z0_vec(mut self, z0: Vec<Complex64>) -> Self {
        self.z0 = Array1::from_shape_fn(z0.len(), |i| z0[i]);
        self
    }

    fn set_port_names(mut self) -> Self {
        if self.port_names.is_none() {
            self.port_names = Some(vec!["".to_string(); self.nports])
        }
        self
    }

    /// Provide ABCD parameters representation of Network
    pub fn a(mut self, net: Points) -> Self {
        self.param = RFParameter::A;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide inverse hybrid parameters representation of Network
    pub fn g(mut self, net: Points) -> Self {
        self.param = RFParameter::G;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide hybrid parameters representation of Network
    pub fn h(mut self, net: Points) -> Self {
        self.param = RFParameter::H;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide S-parameter representation of Network
    pub fn s(mut self, net: Points) -> Self {
        self.param = RFParameter::S;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide Power S-parameter representation of Network
    pub fn s_power(mut self, net: Points) -> Self {
        self.param = RFParameter::SPower;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide Pseudo S-parameter representation of Network
    pub fn s_pseudo(mut self, net: Points) -> Self {
        self.param = RFParameter::SPseudo;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide Traveling S-parameter representation of Network
    pub fn s_traveling(mut self, net: Points) -> Self {
        self.param = RFParameter::STraveling;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide cascade parameters representation of Network
    pub fn t(mut self, net: Points) -> Self {
        self.param = RFParameter::T;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide admittance parameters representation of Network
    pub fn y(mut self, net: Points) -> Self {
        self.param = RFParameter::Y;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide impedance parameters representation of Network
    pub fn z(mut self, net: Points) -> Self {
        self.param = RFParameter::Z;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    /// Provide RF parameters representation of Network
    pub fn params(mut self, net: Points, param: RFParameter) -> Self {
        self.param = param;
        self.nports = net.slice(s![0, .., ..]).nrows();
        self.net = Some(net);
        self.set_port_names()
    }

    pub fn build(mut self) -> Network {
        match self.net {
            None => panic!("No network parameters have been specified!"),
            Some(net) => {
                if !net.slice(s![0, .., ..]).is_square() {
                    panic!(
                        "{}",
                        format!(
                            "Provided data is not square!\n{rows} rows\t\t{cols} cols",
                            rows = net.slice(s![0, .., 0]).len(),
                            cols = net.slice(s![0, 0, ..]).len()
                        )
                    );
                }
                if net.slice(s![0, .., 0]).len() != self.z0.len() {
                    panic!(
                        "{}",
                        format!(
                            "Number of ports and number of Z0 points do not match!\n{npts} network points\t\t{zpts} port impedance points",
                            npts = net.slice(s![0, .., 0]).len(),
                            zpts = self.z0.len()
                        )
                    );
                }
                if self.freq.npts() != net.slice(s![.., 0, 0]).len() {
                    panic!(
                        "{}",
                        format!(
                            "Number of frequency points does not match number of data points!\n{fpts} frequency points\t\t{npts} network points",
                            fpts = self.freq.npts(),
                            npts = net.slice(s![.., 0, 0]).len()
                        )
                    );
                }

                let mut ports: Array1<PortVal> = Array1::<PortVal>::from_elem(
                    net.slice(s![0, .., ..]).nrows() * net.slice(s![0, .., ..]).ncols(),
                    (0, 0),
                );
                for i in 0..net.slice(s![0, .., ..]).nrows() {
                    for j in 0..net.slice(s![0, .., ..]).ncols() {
                        let idx = i * net.slice(s![0, .., ..]).ncols() + j;
                        ports[idx] = (i, j);
                    }
                }

                self.nports = net.slice(s![0, .., ..]).nrows();
                self.npts = net.slice(s![.., 0, 0]).len();
                self.dim = net.dim();

                let wave_type = WaveType::Power;
                if self.nports == 2 {
                    match self.param {
                        RFParameter::A => {
                            self.a = Some(net.clone());
                            self.g = net.clone().a_to_g();
                            self.h = net.clone().a_to_h();
                            self.s = net.clone().a_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().a_to_t(&self.z0);
                            self.y = net.clone().a_to_y();
                            self.z = net.clone().a_to_z();
                        }
                        RFParameter::G => {
                            self.a = net.clone().g_to_a();
                            self.g = Some(net.clone());
                            self.h = net.clone().g_to_h();
                            self.s = net.clone().g_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().g_to_t(&self.z0);
                            self.y = net.clone().g_to_y();
                            self.z = net.clone().g_to_z();
                        }
                        RFParameter::H => {
                            self.a = net.clone().h_to_a();
                            self.g = net.clone().h_to_g();
                            self.h = Some(net.clone());
                            self.s = net.clone().h_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().h_to_t(&self.z0);
                            self.y = net.clone().h_to_y();
                            self.z = net.clone().h_to_z();
                        }
                        RFParameter::S | RFParameter::SPower => {
                            self.a = net.clone().s_to_a(&self.z0);
                            self.g = net.clone().s_to_g(&self.z0);
                            self.h = net.clone().s_to_h(&self.z0);
                            self.s = Some(net.clone());
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().s_to_t();
                            self.y = net.clone().s_to_y(&self.z0);
                            self.z = net.clone().s_to_z(&self.z0);
                        }
                        RFParameter::SPseudo => {
                            let wave_type = WaveType::Pseudo;
                            self.a = net.clone().s_to_a(&self.z0);
                            self.g = net.clone().s_to_g(&self.z0);
                            self.h = net.clone().s_to_h(&self.z0);
                            self.s = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Power,
                            );
                            self.s_power = self.s.clone();
                            self.s_pseudo = Some(net.clone());
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().s_to_t();
                            self.y = net.clone().s_to_y(&self.z0);
                            self.z = net.clone().s_to_z(&self.z0);
                        }
                        RFParameter::STraveling => {
                            let wave_type = WaveType::Traveling;
                            self.a = net.clone().s_to_a(&self.z0);
                            self.g = net.clone().s_to_g(&self.z0);
                            self.h = net.clone().s_to_h(&self.z0);
                            self.s = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Power,
                            );
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = Some(net.clone());
                            self.t = net.clone().s_to_t();
                            self.y = net.clone().s_to_y(&self.z0);
                            self.z = net.clone().s_to_z(&self.z0);
                        }
                        RFParameter::T => {
                            self.a = net.clone().t_to_a(&self.z0);
                            self.g = net.clone().t_to_g(&self.z0);
                            self.h = net.clone().t_to_h(&self.z0);
                            self.s = net.clone().t_to_s();
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = Some(net.clone());
                            self.y = net.clone().t_to_y(&self.z0);
                            self.z = net.clone().t_to_z(&self.z0);
                        }
                        RFParameter::Y => {
                            self.a = net.clone().y_to_a();
                            self.g = net.clone().y_to_g();
                            self.h = net.clone().y_to_h();
                            self.s = net.clone().y_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().y_to_t(&self.z0);
                            self.y = Some(net.clone());
                            self.z = net.clone().y_to_z();
                        }
                        RFParameter::Z => {
                            self.a = net.clone().z_to_a();
                            self.g = net.clone().z_to_g();
                            self.h = net.clone().z_to_h();
                            self.s = net.clone().z_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.t = net.clone().z_to_t(&self.z0);
                            self.y = net.clone().z_to_y();
                            self.z = Some(net.clone());
                        }
                    }
                } else {
                    match self.param {
                        RFParameter::A | RFParameter::G | RFParameter::H | RFParameter::T => {
                            panic!("{}", network_err_msg(self.param, self.nports))
                        }
                        RFParameter::S | RFParameter::SPower => {
                            self.s = Some(net.clone());
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.y = net.clone().s_to_y(&self.z0);
                            self.z = net.clone().s_to_z(&self.z0);
                        }
                        RFParameter::SPseudo => {
                            let wave_type = WaveType::Pseudo;
                            self.s = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Power,
                            );
                            self.s_power = self.s.clone();
                            self.s_pseudo = Some(net.clone());
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.y = net.clone().s_to_y(&self.z0);
                            self.z = net.clone().s_to_z(&self.z0);
                        }
                        RFParameter::STraveling => {
                            let wave_type = WaveType::Traveling;
                            self.s = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Power,
                            );
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = Some(net.clone());
                            self.y = net.clone().s_to_y(&self.z0);
                            self.z = net.clone().s_to_z(&self.z0);
                        }
                        RFParameter::Y => {
                            self.s = net.clone().y_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.y = Some(net.clone());
                            self.z = net.clone().y_to_z();
                        }
                        RFParameter::Z => {
                            self.s = net.clone().z_to_s(&self.z0);
                            self.s_power = self.s.clone();
                            self.s_pseudo = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Pseudo,
                            );
                            self.s_traveling = self.s.clone().unwrap().s_to_s(
                                &self.z0,
                                wave_type,
                                WaveType::Traveling,
                            );
                            self.y = net.clone().z_to_y();
                            self.z = Some(net.clone());
                        }
                    }
                }

                Network {
                    name: self.name,
                    comments: self.comments,
                    nports: self.nports,
                    port_names: self.port_names.unwrap(),
                    ports,
                    freq: self.freq,
                    npts: self.npts,
                    dim: self.dim,
                    z0: self.z0,
                    a: self.a,
                    g: self.g,
                    h: self.h,
                    s: self.s,
                    s_power: None,
                    s_pseudo: None,
                    s_traveling: None,
                    t: self.t,
                    y: self.y,
                    z: self.z,
                }
            }
        }
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        NetworkBuilder {
            name: "".to_string(),
            comments: "".to_string(),
            nports: 0,
            port_names: None,
            ports: vec![],
            freq: Frequency::default(),
            npts: 0,
            dim: (0, 0, 0),
            z0: array![c64(50.0, 0.0)],
            param: RFParameter::S,
            net: None,
            a: None,
            g: None,
            h: None,
            s: None,
            s_power: None,
            s_pseudo: None,
            s_traveling: None,
            t: None,
            y: None,
            z: None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frequency::FrequencyBuilder;
    use crate::impedance::*;
    use crate::scale::*;
    use crate::unit::*;
    use crate::util::*;
    use float_cmp::F64Margin;
    use std::collections::HashMap;
    use std::hash::Hash;

    const MARGIN: F64Margin = F64Margin {
        // epsilon: f64::EPSILON,
        epsilon: 1e-12,
        ulps: 10,
    };

    fn compare_2ports(calc: &Network, exemplars: HashMap<RFParameter, Points>) {
        let margin = MARGIN;
        let mut new_net = calc.clone();
        let x = Points::new(array![
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
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.a(), margin, "a()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.a(), margin, "a(x)");
        new_net.set_net(calc.a(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.a(),
            margin,
            "a(new_net)",
        );

        let param = RFParameter::G;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.g(), margin, "g()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.g(), margin, "g(x)");
        new_net.set_net(calc.g(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.g(),
            margin,
            "g(new_net)",
        );

        let param = RFParameter::H;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.h(), margin, "h()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.h(), margin, "h(x)");
        new_net.set_net(calc.h(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.h(),
            margin,
            "h(new_net)",
        );

        let param = RFParameter::S;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.s(), margin, "s()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.s(), margin, "s(x)");
        new_net.set_net(calc.s(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.s(),
            margin,
            "s(new_net)",
        );

        let param = RFParameter::SPower;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.s_power(),
            margin,
            "s_power()",
        );
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.s_power(), margin, "s_power(x)");
        new_net.set_net(calc.s_power(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.s_power(),
            margin,
            "s_power(new_net)",
        );

        let param = RFParameter::SPseudo;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.s_pseudo(),
            margin,
            "s_pseudo()",
        );
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.s_pseudo(), margin, "s_pseudo(x)");
        new_net.set_net(calc.s_pseudo(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.s_pseudo(),
            margin,
            "s_pseudo(new_net)",
        );

        let param = RFParameter::STraveling;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.s_traveling(),
            margin,
            "s_traveling()",
        );
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.s_traveling(), margin, "s_traveling(x)");
        new_net.set_net(calc.s_traveling(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.s_traveling(),
            margin,
            "s_traveling(new_net)",
        );

        let param = RFParameter::T;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.t(), margin, "t()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.t(), margin, "t(x)");
        new_net.set_net(calc.t(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.t(),
            margin,
            "t(new_net)",
        );

        let param = RFParameter::Y;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.y(), margin, "y()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.y(), margin, "y(x)");
        new_net.set_net(calc.y(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.y(),
            margin,
            "y(new_net)",
        );

        let param = RFParameter::Z;
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &calc.net(param),
            margin,
            format!("net({})", param).as_str(),
        );
        comp_points_c64(&exemplars.get(&param).unwrap(), &calc.z(), margin, "z()");
        new_net.set_net(&x, param);
        comp_points_c64(&x, &new_net.z(), margin, "z(x)");
        new_net.set_net(calc.z(), param);
        comp_points_c64(
            &exemplars.get(&param).unwrap(),
            &new_net.z(),
            margin,
            "z(new_net)",
        );
    }

    #[test]
    fn network_builder() {
        let name = String::from("title");
        let comments = String::from("here are some comments\nand some more");
        let fdata = array![1.0, 2.0, 3.0];
        let z0 = array![c64(50.0, 0.0), c64(50.0, 0.0),];
        let ndata = Points::new(array![
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
        ]);
        let freq = FrequencyBuilder::new()
            .freqs_scaled(fdata, Scale::Giga)
            .build();
        let nports: usize = 2;
        let npoints: usize = 3;
        let mut net = NetworkBuilder::new()
            .freq(freq.clone())
            .z0(z0.clone())
            .s(ndata.clone())
            .name("title")
            .comments("here are some comments\nand some more")
            // .port_names(vec![""; 2])
            .build();

        assert_eq!(name, *net.name());
        assert_eq!(comments, *net.comments());
        assert_eq!(nports, net.nports());
        assert_eq!(npoints, net.npts());
        for i in 0..2 {
            assert_eq!(z0[i], net.z0()[i]);
            assert_eq!(z0[i], net.z0_at_port_idx(i).clone());
            assert_eq!(String::from(""), *net.port_name(i));
        }
        net.set_port_name(0, String::from("1"));
        net.set_port_name(1, String::from("2"));
        assert_eq!(String::from("1"), *net.port_name(0));
        assert_eq!(String::from("2"), *net.port_name(1));
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
        let freq = UnitValBuilder::new().val_scaled(275.0, Scale::Giga).build();
        let src = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(42.4)
            .im(-19.6)
            .z0(z0)
            .freq(freq)
            .build();
        let load = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(212.3)
            .im(43.2)
            .z0(z0)
            .freq(freq)
            .build();
        let net = NetworkBuilder::new()
            .freq(
                FrequencyBuilder::new()
                    .freqs_scaled(array![275.0], Scale::Giga)
                    .build(),
            )
            .z0(array![src.z(), load.z()])
            .s(Points::new(array![[
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ]]))
            .build();
        println!("net: {:?}", net);
        assert_eq!(1, net.npts());
    }
}
