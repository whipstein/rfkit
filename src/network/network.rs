use crate::frequency::Frequency;
use crate::impedance::ComplexNumberType;
use crate::math::*;
use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use crate::network::{NetworkPoint, PortPoints, PortPointsf64, PortVal, WaveType, network_err_msg};
use crate::parameter::RFParameter;
use crate::point::{Point, PointComplex, Pt};
use crate::points::{Points, Pointsf64, Pts};
use crate::scale::Scale;
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

/// This type defines an RF network.
///
/// The network contains all the information required to define a multi-port network. It includes direct storage off all network representation types (ABCD, g, h, S, T, Y & Z).
///
/// ## Example
/// ```
/// use ndarray::prelude::*;
/// use num::complex::c64;
/// use rfkit_base_ndarray::prelude::*;
///
/// let z0 = array![c64(50.0, 0.0), c64(50.0, 0.0),];
/// let fdata = array![1.0, 2.0, 3.0];
/// let ndata: Points = array![
///     [
///         [c64(0.958, -0.263), c64(-0.846, 0.158)],
///         [c64(0.004, 0.022), c64(0.544, -0.129)]
///     ],
///     [
///         [c64(0.849, -0.496), c64(-6.688, 2.413)],
///         [c64(0.014, 0.041), c64(0.494, -0.248)]
///     ],
///     [
///         [c64(0.700, -0.648), c64(-0.724, 0.396)],
///         [c64(0.026, 0.053), c64(0.435, -0.311)]
///     ],
/// ];
/// let freq = FrequencyBuilder::new()
///    .freqs_scaled(fdata, Scale::Giga)
///    .build();
/// let net = Network::new(freq.clone(), z0.clone(), RFParameter::S, ndata, "My Network".to_string(), "here are some comments\nand some more".to_string());
/// ```
#[derive(Clone)]
pub struct Network {
    pub(super) name: String,
    pub(super) comments: String,
    pub(super) nports: usize,
    pub(super) port_names: Vec<String>,
    pub(super) ports: Array1<PortVal>,
    pub(super) freq: Frequency,
    pub(super) npts: usize,
    pub(super) dim: (usize, usize, usize),
    pub(super) z0: Array1<Complex64>,
    pub(super) a: Option<Points>,
    pub(super) g: Option<Points>,
    pub(super) h: Option<Points>,
    pub(super) s: Option<Points>,
    pub(super) s_power: Option<Points>,
    pub(super) s_pseudo: Option<Points>,
    pub(super) s_traveling: Option<Points>,
    pub(super) t: Option<Points>,
    pub(super) y: Option<Points>,
    pub(super) z: Option<Points>,
}

impl Network {
    pub fn new(
        freq: Frequency,
        z0: Array1<Complex64>,
        param: RFParameter,
        net: Points,
        name: String,
        comments: String,
    ) -> Network {
        if !net.slice(s![0, .., ..]).is_square() {
            panic!(
                "{}",
                format!(
                    "Provided data is not square!\n{rows} rows\t\t{cols} cols",
                    rows = net.slice(s![0, .., ..]).nrows(),
                    cols = net.slice(s![0, .., ..]).ncols()
                )
            );
        }
        if net.slice(s![0, .., ..]).nrows() != z0.len() {
            panic!(
                "{}",
                format!(
                    "Number of ports and number of Z0 points do not match!\n{npts} network points\t\t{zpts} port impedance points",
                    npts = net.slice(s![0, .., ..]).nrows(),
                    zpts = z0.len()
                )
            );
        }
        if freq.npts() != net.slice(s![.., 0, 0]).len() {
            panic!(
                "{}",
                format!(
                    "Number of frequency points does not match number of data points!\n{fpts} frequency points\t\t{npts} network points",
                    fpts = freq.npts(),
                    npts = net.slice(s![.., 0, 0]).len()
                )
            );
        }

        let nports = net.slice(s![0, .., ..]).nrows();

        Network::new_w_port_names(
            freq,
            z0,
            param,
            net,
            name,
            comments,
            vec![String::from(""); nports],
        )
    }

    pub fn new_from(
        freq: Frequency,
        z0: Array1<Complex64>,
        param: RFParameter,
        format: ComplexNumberType,
        net: Vec<Vec<(f64, f64)>>,
        name: String,
        comments: String,
    ) -> Network {
        let tmp_net: Vec<&[(f64, f64)]> = net.iter().map(|x| x.as_slice()).collect();
        let new_net: &[&[(f64, f64)]] = &tmp_net;
        let net_recalc = match format {
            ComplexNumberType::Db => Points::from_db(new_net),
            ComplexNumberType::MagAng => Points::from_ma(new_net),
            ComplexNumberType::ReIm => Points::from_ri(new_net),
        };

        let nports = net_recalc.slice(s![0, .., 0]).len();

        Network::new_w_port_names(
            freq,
            z0,
            param,
            net_recalc,
            name,
            comments,
            vec![String::from(""); nports],
        )
    }

    pub fn new_w_port_names(
        freq: Frequency,
        z0: Array1<Complex64>,
        param: RFParameter,
        net: Points,
        name: String,
        comments: String,
        port_names: Vec<String>,
    ) -> Network {
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
        if net.slice(s![0, .., 0]).len() != z0.len() {
            panic!(
                "{}",
                format!(
                    "Number of ports and number of Z0 points do not match!\n{npts} network points\t\t{zpts} port impedance points",
                    npts = net.slice(s![0, .., 0]).len(),
                    zpts = z0.len()
                )
            );
        }
        if freq.npts() != net.slice(s![.., 0, 0]).len() {
            panic!(
                "{}",
                format!(
                    "Number of frequency points does not match number of data points!\n{fpts} frequency points\t\t{npts} network points",
                    fpts = freq.npts(),
                    npts = net.slice(s![.., 0, 0]).len()
                )
            );
        }

        let mut ports = Array1::<PortVal>::from_elem(
            net.slice(s![0, .., ..]).nrows() * net.slice(s![0, .., ..]).ncols(),
            (0, 0),
        );
        for i in 0..net.slice(s![0, .., ..]).nrows() {
            for j in 0..net.slice(s![0, .., ..]).ncols() {
                let idx = i * net.slice(s![0, .., ..]).ncols() + j;
                ports[idx] = (i, j);
            }
        }

        let mut out = Network {
            name,
            comments,
            nports: net.slice(s![0, .., ..]).nrows(),
            port_names,
            ports,
            freq,
            npts: net.slice(s![.., 0, 0]).len(),
            dim: net.dim(),
            z0: z0.clone(),
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
        };

        let wave_type = WaveType::Power;
        if out.nports == 2 {
            match param {
                RFParameter::A => {
                    out.a = Some(net.clone());
                    out.g = net.clone().a_to_g();
                    out.h = net.clone().a_to_h();
                    out.s = net.clone().a_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().a_to_t(&z0);
                    out.y = net.clone().a_to_y();
                    out.z = net.clone().a_to_z();
                }
                RFParameter::G => {
                    out.a = net.clone().g_to_a();
                    out.g = Some(net.clone());
                    out.h = net.clone().g_to_h();
                    out.s = net.clone().g_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().g_to_t(&z0);
                    out.y = net.clone().g_to_y();
                    out.z = net.clone().g_to_z();
                }
                RFParameter::H => {
                    out.a = net.clone().h_to_a();
                    out.g = net.clone().h_to_g();
                    out.h = Some(net.clone());
                    out.s = net.clone().h_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().h_to_t(&z0);
                    out.y = net.clone().h_to_y();
                    out.z = net.clone().h_to_z();
                }
                RFParameter::S | RFParameter::SPower => {
                    out.a = net.clone().s_to_a(&z0);
                    out.g = net.clone().s_to_g(&z0);
                    out.h = net.clone().s_to_h(&z0);
                    out.s = Some(net.clone());
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().s_to_t();
                    out.y = net.clone().s_to_y(&z0);
                    out.z = net.clone().s_to_z(&z0);
                }
                RFParameter::SPseudo => {
                    let wave_type = WaveType::Pseudo;
                    out.a = net.clone().s_to_a(&z0);
                    out.g = net.clone().s_to_g(&z0);
                    out.h = net.clone().s_to_h(&z0);
                    out.s = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Power);
                    out.s_power = out.s.clone();
                    out.s_pseudo = Some(net.clone());
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().s_to_t();
                    out.y = net.clone().s_to_y(&z0);
                    out.z = net.clone().s_to_z(&z0);
                }
                RFParameter::STraveling => {
                    let wave_type = WaveType::Traveling;
                    out.a = net.clone().s_to_a(&z0);
                    out.g = net.clone().s_to_g(&z0);
                    out.h = net.clone().s_to_h(&z0);
                    out.s = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Power);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling = Some(net.clone());
                    out.t = net.clone().s_to_t();
                    out.y = net.clone().s_to_y(&z0);
                    out.z = net.clone().s_to_z(&z0);
                }
                RFParameter::T => {
                    out.a = net.clone().t_to_a(&z0);
                    out.g = net.clone().t_to_g(&z0);
                    out.h = net.clone().t_to_h(&z0);
                    out.s = net.clone().t_to_s();
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = Some(net.clone());
                    out.y = net.clone().t_to_y(&z0);
                    out.z = net.clone().t_to_z(&z0);
                }
                RFParameter::Y => {
                    out.a = net.clone().y_to_a();
                    out.g = net.clone().y_to_g();
                    out.h = net.clone().y_to_h();
                    out.s = net.clone().y_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().y_to_t(&z0);
                    out.y = Some(net.clone());
                    out.z = net.clone().y_to_z();
                }
                RFParameter::Z => {
                    out.a = net.clone().z_to_a();
                    out.g = net.clone().z_to_g();
                    out.h = net.clone().z_to_h();
                    out.s = net.clone().z_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.t = net.clone().z_to_t(&z0);
                    out.y = net.clone().z_to_y();
                    out.z = Some(net.clone());
                }
            }
        } else {
            match param {
                RFParameter::A | RFParameter::G | RFParameter::H | RFParameter::T => {
                    panic!("{}", network_err_msg(param, out.nports))
                }
                RFParameter::S | RFParameter::SPower => {
                    out.s = Some(net.clone());
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.y = net.clone().s_to_y(&z0);
                    out.z = net.clone().s_to_z(&z0);
                }
                RFParameter::SPseudo => {
                    let wave_type = WaveType::Pseudo;
                    out.s = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Power);
                    out.s_power = out.s.clone();
                    out.s_pseudo = Some(net.clone());
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.y = net.clone().s_to_y(&z0);
                    out.z = net.clone().s_to_z(&z0);
                }
                RFParameter::STraveling => {
                    let wave_type = WaveType::Traveling;
                    out.s = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Power);
                    out.s_power = out.s.clone();
                    out.s_pseudo = Some(net.clone());
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.y = net.clone().s_to_y(&z0);
                    out.z = net.clone().s_to_z(&z0);
                }
                RFParameter::Y => {
                    out.s = net.clone().y_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.y = Some(net.clone());
                    out.z = net.clone().y_to_z();
                }
                RFParameter::Z => {
                    out.s = net.clone().z_to_s(&z0);
                    out.s_power = out.s.clone();
                    out.s_pseudo = out
                        .s
                        .clone()
                        .unwrap()
                        .s_to_s(&z0, wave_type, WaveType::Pseudo);
                    out.s_traveling =
                        out.s
                            .clone()
                            .unwrap()
                            .s_to_s(&z0, wave_type, WaveType::Traveling);
                    out.y = net.clone().z_to_y();
                    out.z = Some(net.clone());
                }
            }
        }

        out
    }

    pub fn a(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.a.as_ref().unwrap()
    }

    pub fn a_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_db(RFParameter::A)
    }

    pub fn a_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_deg(RFParameter::A)
    }

    pub fn a_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_im(RFParameter::A)
    }

    pub fn a_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_mag(RFParameter::A)
    }

    pub fn a_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_rad(RFParameter::A)
    }

    pub fn a_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_re(RFParameter::A)
    }

    pub fn chop_in_half(&self) -> Network {
        if self.nports() != 2 {
            panic!("Cannot chop_in_half network that is not 2 ports")
        } else if !self.is_reciprocal() {
            panic!("Cannot chop_in_half network that is not reciprocal")
        }

        let mut tmp: Vec<Complex64> = vec![];
        let mut out = Points::zeros(self.dim);
        println!("{:?}", out);
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = self.s().slice(s![i, .., ..]);
            let b11 = val[[0, 0]];
            let b12 = val[[0, 1]];
            let b21 = val[[1, 0]];
            let b22 = val[[1, 1]];
            let a11 = b11 / (1.0 + b12);
            tmp.push(b21 * (1.0 - b11 * b22 / (1.0 + b12).powi(2)));
            let a22 = b22 / (1.0 + b12);
            pt.assign(&array![[a11, c64::from(0.0)], [c64::from(0.0), a22]].view());
        }

        let tmp_array = sqrt_phase_unwrap(Array1::<Complex64>::from_vec(tmp));

        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt[[0, 1]] = tmp_array[i];
            pt[[1, 0]] = tmp_array[i];
        }

        Network::new(
            self.freq.clone(),
            self.z0.clone(),
            RFParameter::S,
            out,
            self.name.clone(),
            self.comments.clone(),
        )
    }

    pub fn comments(&self) -> &String {
        &self.comments
    }

    pub fn connect(&self, p1: usize, net: &Network, p2: usize) -> Network {
        let mut out = self.clone();
        let calc_net = self.s.clone().unwrap().connect(p1, net.s(), p2).unwrap();
        out.set_net(&calc_net, RFParameter::S);
        out
    }

    pub fn freq(&self) -> &Frequency {
        &self.freq
    }

    pub fn g(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.g.as_ref().unwrap()
    }

    pub fn g_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_db(RFParameter::G)
    }

    pub fn g_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_deg(RFParameter::G)
    }

    pub fn g_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_im(RFParameter::G)
    }

    pub fn g_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_mag(RFParameter::G)
    }

    pub fn g_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_rad(RFParameter::G)
    }

    pub fn g_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_re(RFParameter::G)
    }

    pub fn gd(&self, port: PortVal, scale: Scale) -> Array1<f64> {
        let mut out: Vec<f64> = vec![];

        let dphi = gradient(unwrap(&self.net_rad_at_port(RFParameter::S, port)));
        let dw = gradient(self.freq().w());

        if dphi.len() != dw.len() {
            panic!(
                "{}",
                format!(
                    "Length of dphi and dw do not match: dphi: {}\t\t dw: {}",
                    dphi.len(),
                    dw.len()
                )
            );
        }

        for i in 0..dphi.len() {
            out.push(scale.scale(-dphi[i] / dw[i]));
        }

        Array1::<f64>::from_vec(out)
    }

    pub fn h(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.h.as_ref().unwrap()
    }

    pub fn h_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_db(RFParameter::H)
    }

    pub fn h_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_deg(RFParameter::H)
    }

    pub fn h_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_im(RFParameter::H)
    }

    pub fn h_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_mag(RFParameter::H)
    }

    pub fn h_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_rad(RFParameter::H)
    }

    pub fn h_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_re(RFParameter::H)
    }

    pub fn is_passive(&self) -> bool {
        for val in self.passivity().iter() {
            if *val < 0.0 {
                return false;
            }
        }
        true
    }

    pub fn is_reciprocal(&self) -> bool {
        for pt in self.s().outer_iter() {
            if !Point::new(pt.to_owned()).is_reciprocal() {
                return false;
            }
        }
        true
    }

    pub fn k(&self) -> Array1<f64> {
        if self.nports != 2 {
            panic!(
                "{}",
                format!(
                    "k() does not exist for networks that are not 2 port. nports = {}",
                    self.nports
                )
            );
        }

        let s11 = self.s().slice(s![.., 0, 0]);
        let s12 = self.s().slice(s![.., 0, 1]);
        let s21 = self.s().slice(s![.., 1, 0]);
        let s22 = self.s().slice(s![.., 1, 1]);
        let d = &s11 * &s22 - &s21 * &s12;

        let mut val = Array1::<f64>::zeros(s11.len());
        azip!((index i, &s11 in &s11, &s12 in &s12, &s21 in &s21, &s22 in &s22, &d in &d) {
            val[i] = (1.0 - s11.abs().powi(2) - s22.abs().powi(2) + d.abs().powi(2)) / (2.0 * (s21 * s12).abs());
        });

        val
    }

    pub fn max_gain(&self) -> PortPointsf64 {
        if self.nports != 2 {
            panic!(
                "{}",
                format!(
                    "max_gain() does not exist for networks that are not 2 port. nports = {}",
                    self.nports
                )
            );
        }

        let mut k = self.k();

        for val in k.iter_mut() {
            if *val < 1.0 {
                *val = 1.0;
            }
        }

        &self.max_stable_gain()
            / (&k + Array1::<f64>::from_shape_fn(k.len(), |i| (k[i].powi(2) - 1.0).sqrt()))
    }

    pub fn max_stable_gain(&self) -> PortPointsf64 {
        if self.nports != 2 {
            panic!(
                "{}",
                format!(
                    "max_stable_gain() does not exist for networks that are not 2 port. nports = {}",
                    self.nports
                )
            );
        }

        &self.s_mag().slice(s![.., 1, 0]) / &self.s_mag().slice(s![.., 0, 1])
    }

    pub fn mu(&self) -> Array1<f64> {
        if self.nports != 2 {
            panic!(
                "{}",
                format!(
                    "mu() does not exist for networks that are not 2 port. nports = {}",
                    self.nports
                )
            );
        }

        let d = &self.s().slice(s![.., 0, 0]) * &self.s().slice(s![.., 1, 1])
            - &self.s().slice(s![.., 0, 1]) * &self.s().slice(s![.., 1, 0]);
        let d_abs = Array1::<f64>::from_shape_fn(self.npts, |i| d[i].abs());
        let denom_arg1 = &self.s().slice(s![.., 1, 1]) - &self.s_conj().slice(s![.., 0, 0]) * &d;
        let denom_arg2 = &self.s().slice(s![.., 0, 1]) * &self.s_conj().slice(s![.., 1, 0]);
        let denom =
            Array1::<f64>::from_shape_fn(self.npts, |i| denom_arg1[i].abs() + denom_arg2[i].abs());
        let num = &Array1::<f64>::ones(self.npts)
            - &Array1::<f64>::from_shape_fn(self.npts, |i| {
                self.s().slice(s![.., 0, 0])[i].powi(2).abs()
            });

        &num / &denom
    }

    pub fn mu_prime(&self) -> Array1<f64> {
        if self.nports != 2 {
            panic!(
                "{}",
                format!(
                    "mu_prime() does not exist for networks that are not 2 port. nports = {}",
                    self.nports
                )
            );
        }

        let d = &self.s().slice(s![.., 0, 0]) * &self.s().slice(s![.., 1, 1])
            - &self.s().slice(s![.., 0, 1]) * &self.s().slice(s![.., 1, 0]);
        let d_abs = Array1::<f64>::from_shape_fn(self.npts, |i| d[i].abs());
        let denom_arg1 = &self.s().slice(s![.., 0, 0]) - &self.s_conj().slice(s![.., 1, 1]) * &d;
        let denom_arg2 = &self.s().slice(s![.., 0, 1]) * &self.s_conj().slice(s![.., 1, 0]);
        let denom =
            Array1::<f64>::from_shape_fn(self.npts, |i| denom_arg1[i].abs() + denom_arg2[i].abs());
        let num = &Array1::<f64>::ones(self.npts)
            - &Array1::<f64>::from_shape_fn(self.npts, |i| {
                self.s().slice(s![.., 1, 1])[i].powi(2).abs()
            });

        &num / &denom
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn net(&self, param: RFParameter) -> &Points {
        match param {
            RFParameter::A => {
                if self.nports != 2 {
                    panic!(
                        "{}",
                        format!(
                            "ABCD parameters do not exist for network with {nports} port(s)",
                            nports = self.nports
                        )
                    );
                }
                self.a.as_ref().unwrap()
            }
            RFParameter::G => {
                if self.nports != 2 {
                    panic!(
                        "{}",
                        format!(
                            "Inverse Hybrid (g) parameters do not exist for network with {nports} port(s)",
                            nports = self.nports
                        )
                    );
                }
                self.g.as_ref().unwrap()
            }
            RFParameter::H => {
                if self.nports != 2 {
                    panic!(
                        "{}",
                        format!(
                            "Hybrid (h) parameters do not exist for network with {nports} port(s)",
                            nports = self.nports
                        )
                    );
                }
                self.h.as_ref().unwrap()
            }
            RFParameter::S => self.s.as_ref().unwrap(),
            RFParameter::SPower => self.s_power.as_ref().unwrap(),
            RFParameter::SPseudo => self.s_pseudo.as_ref().unwrap(),
            RFParameter::STraveling => self.s_traveling.as_ref().unwrap(),
            RFParameter::S => self.s.as_ref().unwrap(),
            RFParameter::T => self.t.as_ref().unwrap(),
            RFParameter::Y => self.y.as_ref().unwrap(),
            RFParameter::Z => self.z.as_ref().unwrap(),
        }
    }

    pub fn net_at_freq_idx(&self, param: RFParameter, idx: usize) -> Point {
        Point::new(self.net(param).slice(s![idx, .., ..]).to_owned())
    }

    pub fn net_at_port(&self, param: RFParameter, port: PortVal) -> PortPoints {
        self.net(param).slice(s![.., port.0, port.1]).to_owned()
    }

    pub fn net_at_port_conj(&self, param: RFParameter, port: PortVal) -> PortPoints {
        self.net_conj(param)
            .slice(s![.., port.0, port.1])
            .to_owned()
    }

    pub fn net_at_port_idx(&self, param: RFParameter, j: usize, k: usize) -> PortPoints {
        self.net(param).slice(s![.., j, k]).to_owned()
    }

    pub fn net_at_idx(&self, param: RFParameter, idx: usize, j: usize, k: usize) -> Complex64 {
        self.net_at_freq_idx(param, idx)[[j, k]]
    }

    pub fn net_conj(&self, param: RFParameter) -> Points {
        Points::from_shape_fn(self.net(param).dim(), |(i, j, k)| {
            self.net(param)[[i, j, k]].conj()
        })
    }

    pub fn net_db(&self, param: RFParameter) -> Pointsf64 {
        self.net(param).db()
    }

    pub fn net_deg(&self, param: RFParameter) -> Pointsf64 {
        self.net(param).deg()
    }

    pub fn net_im(&self, param: RFParameter) -> Pointsf64 {
        self.net(param).im()
    }

    pub fn net_mag(&self, param: RFParameter) -> Pointsf64 {
        self.net(param).mag()
    }

    pub fn net_rad(&self, param: RFParameter) -> Pointsf64 {
        self.net(param).rad()
    }

    pub fn net_rad_at_port(&self, param: RFParameter, port: PortVal) -> PortPointsf64 {
        self.net(param)
            .rad()
            .slice(s![.., port.0, port.1])
            .to_owned()
    }

    pub fn net_re(&self, param: RFParameter) -> Pointsf64 {
        self.net(param).re()
    }

    pub fn nports(&self) -> usize {
        self.nports
    }

    pub fn npts(&self) -> usize {
        self.npts
    }

    pub fn passivity(&self) -> Array1<f64> {
        if self.nports < 2 {
            panic!(
                "{}",
                format!(
                    "passivity() does not exist for network with 1 port. nports = {}",
                    self.nports
                )
            );
        }

        let mut out = vec![];

        for (i, pt) in self.s().outer_iter().enumerate() {
            let mut acc: Array2<Complex64> = pt.map(|x| x.conj()).t().to_owned();
            acc = acc.dot(&pt);

            let (mid, _) = (Array2::<Complex64>::eye(acc.nrows()) - acc).eig().unwrap();

            let mut val = mid[0].re;
            for x in mid.iter() {
                if x.re < val {
                    val = x.re;
                }
            }
            out.push(val);
        }

        Array1::<f64>::from_vec(out)
    }

    // pub fn port_iter(&self) -> Iter<'_, PortVal> {
    //     self.ports.iter()
    // }

    pub fn port_name(&self, n: usize) -> &String {
        &self.port_names[n]
    }

    pub fn port_names(&self) -> &Vec<String> {
        &self.port_names
    }

    pub fn ports(&self) -> &Array1<PortVal> {
        &self.ports
    }

    pub fn reciprocity(&self) -> Pointsf64 {
        self.s().reciprocity().unwrap()
    }

    pub fn s(&self) -> &Points {
        self.s.as_ref().unwrap()
    }

    pub fn s_power(&self) -> &Points {
        self.s_power.as_ref().unwrap()
    }

    pub fn s_pseudo(&self) -> &Points {
        self.s_pseudo.as_ref().unwrap()
    }

    pub fn s_traveling(&self) -> &Points {
        self.s_traveling.as_ref().unwrap()
    }

    pub fn s_conj(&self) -> Points {
        self.s.as_ref().unwrap().conj()
    }

    pub fn s_db(&self) -> Pointsf64 {
        self.net_db(RFParameter::S)
    }

    pub fn s_deg(&self) -> Pointsf64 {
        self.net_deg(RFParameter::S)
    }

    pub fn s_im(&self) -> Pointsf64 {
        self.net_im(RFParameter::S)
    }

    pub fn s_mag(&self) -> Pointsf64 {
        self.net_mag(RFParameter::S)
    }

    pub fn s_rad(&self) -> Pointsf64 {
        self.net_rad(RFParameter::S)
    }

    pub fn s_re(&self) -> Pointsf64 {
        self.net_re(RFParameter::S)
    }

    pub fn set_net(&mut self, net: &Points, param: RFParameter) {
        if net.slice(s![0, .., ..]).nrows() == 2 {
            match param {
                RFParameter::A => {
                    self.a = Some(net.clone());
                    self.g = net.clone().a_to_g();
                    self.h = net.clone().a_to_h();
                    self.s = net.clone().a_to_s(&self.z0);
                    self.t = net.clone().a_to_t(&self.z0);
                    self.y = net.clone().a_to_y();
                    self.z = net.clone().a_to_z();
                }
                RFParameter::G => {
                    self.a = net.clone().g_to_a();
                    self.g = Some(net.clone());
                    self.h = net.clone().g_to_h();
                    self.s = net.clone().g_to_s(&self.z0);
                    self.t = net.clone().g_to_t(&self.z0);
                    self.y = net.clone().g_to_y();
                    self.z = net.clone().g_to_z();
                }
                RFParameter::H => {
                    self.a = net.clone().h_to_a();
                    self.g = net.clone().h_to_g();
                    self.h = Some(net.clone());
                    self.s = net.clone().h_to_s(&self.z0);
                    self.t = net.clone().h_to_t(&self.z0);
                    self.y = net.clone().h_to_y();
                    self.z = net.clone().h_to_z();
                }
                RFParameter::S
                | RFParameter::SPower
                | RFParameter::SPseudo
                | RFParameter::STraveling => {
                    self.a = net.clone().s_to_a(&self.z0);
                    self.g = net.clone().s_to_g(&self.z0);
                    self.h = net.clone().s_to_h(&self.z0);
                    self.t = net.clone().s_to_t();
                }
                RFParameter::T => {
                    self.a = net.clone().t_to_a(&self.z0);
                    self.g = net.clone().t_to_g(&self.z0);
                    self.h = net.clone().t_to_h(&self.z0);
                    self.s = net.clone().t_to_s();
                    self.t = Some(net.clone());
                    self.y = net.clone().t_to_y(&self.z0);
                    self.z = net.clone().t_to_z(&self.z0);
                }
                RFParameter::Y => {
                    self.a = net.clone().y_to_a();
                    self.g = net.clone().y_to_g();
                    self.h = net.clone().y_to_h();
                    self.t = net.clone().y_to_t(&self.z0);
                }
                RFParameter::Z => {
                    self.a = net.clone().z_to_a();
                    self.g = net.clone().z_to_g();
                    self.h = net.clone().z_to_h();
                    self.t = net.clone().z_to_t(&self.z0);
                }
            }
        }

        let wave_type = WaveType::Power;
        match param {
            RFParameter::S | RFParameter::SPower => {
                self.s = Some(net.clone());
                self.s_power = self.s.clone();
                self.s_pseudo = net.clone().s_to_s(&self.z0, wave_type, WaveType::Pseudo);
                self.s_traveling = net.clone().s_to_s(&self.z0, wave_type, WaveType::Traveling);
                self.y = net.clone().s_to_y(&self.z0);
                self.z = net.clone().s_to_z(&self.z0);
            }
            RFParameter::SPseudo => {
                let wave_type = WaveType::Pseudo;
                self.s = net.clone().s_to_s(&self.z0, wave_type, WaveType::Power);
                self.s_power = self.s.clone();
                self.s_pseudo = Some(net.clone());
                self.s_traveling = net.clone().s_to_s(&self.z0, wave_type, WaveType::Traveling);
                self.y = net.clone().s_to_y(&self.z0);
                self.z = net.clone().s_to_z(&self.z0);
            }
            RFParameter::STraveling => {
                let wave_type = WaveType::Traveling;
                self.s = net.clone().s_to_s(&self.z0, wave_type, WaveType::Power);
                self.s_power = self.s.clone();
                self.s_pseudo = net.clone().s_to_s(&self.z0, wave_type, WaveType::Pseudo);
                self.s_traveling = Some(net.clone());
                self.y = net.clone().s_to_y(&self.z0);
                self.z = net.clone().s_to_z(&self.z0);
            }
            RFParameter::Y => {
                self.s = net.clone().y_to_s(&self.z0);
                self.s_power = Some(net.clone());
                self.s_pseudo = net.clone().s_to_s(&self.z0, wave_type, WaveType::Pseudo);
                self.s_traveling = net.clone().s_to_s(&self.z0, wave_type, WaveType::Traveling);
                self.y = Some(net.clone());
                self.z = net.clone().y_to_z();
            }
            RFParameter::Z => {
                self.s = net.clone().z_to_s(&self.z0);
                self.s_power = Some(net.clone());
                self.s_pseudo = net.clone().s_to_s(&self.z0, wave_type, WaveType::Pseudo);
                self.s_traveling = net.clone().s_to_s(&self.z0, wave_type, WaveType::Traveling);
                self.y = net.clone().z_to_y();
                self.z = Some(net.clone());
            }
            _ => (),
        }
    }

    pub fn set_port_name(&mut self, n: usize, name: String) {
        self.port_names[n] = name;
    }

    pub fn set_port_names(&mut self, names: Vec<String>) {
        self.port_names = names;
    }

    pub fn t(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.t.as_ref().unwrap()
    }

    pub fn t_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_db(RFParameter::T)
    }

    pub fn t_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_deg(RFParameter::T)
    }

    pub fn t_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_im(RFParameter::T)
    }

    pub fn t_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_mag(RFParameter::T)
    }

    pub fn t_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_rad(RFParameter::T)
    }

    pub fn t_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_re(RFParameter::T)
    }

    pub fn y(&self) -> &Points {
        self.y.as_ref().unwrap()
    }

    pub fn y_db(&self) -> Pointsf64 {
        self.net_db(RFParameter::Y)
    }

    pub fn y_deg(&self) -> Pointsf64 {
        self.net_deg(RFParameter::Y)
    }

    pub fn y_im(&self) -> Pointsf64 {
        self.net_im(RFParameter::Y)
    }

    pub fn y_mag(&self) -> Pointsf64 {
        self.net_mag(RFParameter::Y)
    }

    pub fn y_rad(&self) -> Pointsf64 {
        self.net_rad(RFParameter::Y)
    }

    pub fn y_re(&self) -> Pointsf64 {
        self.net_re(RFParameter::Y)
    }

    pub fn z(&self) -> &Points {
        self.z.as_ref().unwrap()
    }

    pub fn z_db(&self) -> Pointsf64 {
        self.net_db(RFParameter::Z)
    }

    pub fn z_deg(&self) -> Pointsf64 {
        self.net_deg(RFParameter::Z)
    }

    pub fn z_im(&self) -> Pointsf64 {
        self.net_im(RFParameter::Z)
    }

    pub fn z_mag(&self) -> Pointsf64 {
        self.net_mag(RFParameter::Z)
    }

    pub fn z_rad(&self) -> Pointsf64 {
        self.net_rad(RFParameter::Z)
    }

    pub fn z_re(&self) -> Pointsf64 {
        self.net_re(RFParameter::Z)
    }

    pub fn z0(&self) -> &Array1<Complex64> {
        &self.z0
    }

    pub fn z0_at_port_idx(&self, idx: usize) -> &c64 {
        &self.z0[idx]
    }

    pub fn z0_to_vector(&self) -> Vec<Complex64> {
        let mut out: Vec<Complex64> = vec![];

        for i in 0..self.z0.len() {
            out.push(self.z0[i]);
        }

        out
    }
}

impl Default for Network {
    fn default() -> Self {
        Network {
            name: "".to_string(),
            comments: "".to_string(),
            nports: 0,
            port_names: vec!["".to_string()],
            ports: array![(1, 1)],
            freq: Frequency::default(),
            npts: 0,
            dim: (0, 0, 0),
            z0: array![c64(50.0, 0.0)],
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

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {name}\nnports: {nports}\nnpoints: {npts}\nz0: {z0:?}",
            name = self.name,
            nports = self.nports(),
            npts = self.npts(),
            z0 = self.z0,
        )
    }
}

impl fmt::Debug for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Network")
            .field("name", &self.name)
            .field("comments", &self.comments)
            .field("nports", &self.nports)
            .field("port_names", &self.port_names)
            .field("freq", &self.freq)
            .field("npts", &self.npts)
            .field("z0", &self.z0)
            .field("a", &self.a)
            .field("g", &self.g)
            .field("h", &self.h)
            .field("s", &self.s)
            .field("t", &self.t)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frequency::FrequencyBuilder;
    use crate::impedance::*;
    use crate::network::NetworkBuilder;
    use crate::points;
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
    fn network_reciprocity() {
        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![[
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(-0.846, 0.158), c64(0.544, -0.129)],
            ]]),
            String::from(""),
            String::from(""),
        );

        let exemplar = Pointsf64::zeros((1, 2, 2));
        comp_points_f64(
            &exemplar,
            &calc.reciprocity(),
            F64Margin::default(),
            "reciprocity(1)",
        );
        assert_eq!(true, calc.is_reciprocal());

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![[
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ]]),
            String::from(""),
            String::from(""),
        );

        let exemplar = Pointsf64::new(array![
            [[0.0, 0.860811245279707], [0.860811245279707, 0.0],]
        ]);

        comp_points_f64(
            &exemplar,
            &calc.reciprocity(),
            F64Margin::default(),
            "reciprocity(2)",
        );
        assert_eq!(false, calc.is_reciprocal());
    }

    #[test]
    fn network_new() {
        let name = String::from("title");
        let comments = String::from("here are some comments\nand some more");
        let fdata = array![1.0, 2.0, 3.0];
        let z0 = array![c64(50.0, 0.0), c64(50.0, 0.0),];
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
        let freq = FrequencyBuilder::new()
            .freqs_scaled(fdata, Scale::Giga)
            .build();
        let nports: usize = 2;
        let npts: usize = 3;
        let mut net = Network::new(
            freq.clone(),
            z0.clone(),
            RFParameter::S,
            ndata.clone(),
            String::from("title"),
            String::from("here are some comments\nand some more"),
        );

        assert_eq!(name, net.name);
        assert_eq!(comments, net.comments);
        assert_eq!(nports, net.nports);
        assert_eq!(npts, net.npts);
        for i in 0..2 {
            assert_eq!(z0[i], net.z0[i]);
            assert_eq!(z0[i], net.z0_at_port_idx(i).clone());
            assert_eq!(String::from(""), *net.port_name(i));
        }
        net.set_port_name(0, String::from("1"));
        net.set_port_name(1, String::from("2"));
        assert_eq!(String::from("1"), *net.port_name(0));
        assert_eq!(String::from("2"), *net.port_name(1));
        for i in 0..npts {
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
    }

    #[test]
    fn network_builder() {
        let name = String::from("title");
        let comments = String::from("here are some comments\nand some more");
        let fdata = array![1.0, 2.0, 3.0];
        let z0 = array![c64(50.0, 0.0), c64(50.0, 0.0),];
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
        let freq = FrequencyBuilder::new()
            .freqs_scaled(fdata, Scale::Giga)
            .build();
        let nports: usize = 2;
        let npts: usize = 3;
        let mut net = NetworkBuilder::new()
            .freq(freq.clone())
            .z0(z0.clone())
            .s(ndata.clone())
            .name("title")
            .comments("here are some comments\nand some more")
            // .port_names(vec![""; 2])
            .build();

        assert_eq!(name, net.name);
        assert_eq!(comments, net.comments);
        assert_eq!(nports, net.nports);
        assert_eq!(npts, net.npts);
        for i in 0..2 {
            assert_eq!(z0[i], net.z0[i]);
            assert_eq!(z0[i], net.z0_at_port_idx(i).clone());
            assert_eq!(String::from(""), *net.port_name(i));
        }
        net.set_port_name(0, String::from("1"));
        net.set_port_name(1, String::from("2"));
        assert_eq!(String::from("1"), *net.port_name(0));
        assert_eq!(String::from("2"), *net.port_name(1));
        for i in 0..npts {
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
            .s(points![[
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ]])
            .build();
        println!("net: {:?}", net);
        assert_eq!(1, net.npts());
    }

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_a() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::A,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );
    }

    #[test]
    fn network_a_to_2port() {
        let param = RFParameter::A;
        // 2 PortVal, 3 Frequncy
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = exemplar;

        let exemplar_g = Points::new(array![
            [
                [
                    c64(-0.001979870974017487, 0.0224209748787405),
                    c64(-0.5458675431868223, 0.10971903563869077)
                ],
                [
                    c64(0.9706839268724422, 0.26648212188669346),
                    c64(-0.8633027773921836, -0.07207581467029678)
                ],
            ],
            [
                [
                    c64(-1.2323927201361262, 1.1168822069634479),
                    c64(-3.4584408230318258, 390.05113808702043)
                ],
                [
                    c64(0.39761214735276523, 0.1911185162508152),
                    c64(-0.7155757503688567, 1.2463556598814403)
                ],
            ],
            [
                [
                    c64(0.006345435959551723, 0.006314005745181268),
                    c64(-0.48829408817838926, 0.4721320825578742)
                ],
                [
                    c64(0.04686626414341519, 9.25064335148286e-4),
                    c64(-0.06661043300916777, 1.1371352153266978)
                ],
            ],
        ]);

        let exemplar_h = Points::new(array![
            [
                [
                    c64(-1.5375603451309596, -0.07416412595936357),
                    c64(0.9625186306094178, -0.22887701590328144)
                ],
                [
                    c64(-1.7403711725430853, -0.41269831113613603),
                    c64(-0.0021178781548226505, 0.03993895904049242)
                ],
            ],
            [
                [
                    c64(-0.008239667486325092, -6.00998985342353e-4),
                    c64(2.0291927440451167, -0.9540811006959242)
                ],
                [
                    c64(-7.910524066400982e-7, -0.002536004683241709),
                    c64(-0.008856852439349883, -0.0036009002998824815)
                ],
            ],
            [
                [
                    c64(-38.351941918327334, 36.03638546644967),
                    c64(31.378370866300532, -0.3315648611508466)
                ],
                [
                    c64(-1.5430289329650075, -1.5195786452299769),
                    c64(0.012353611578814082, 0.41338167219311783)
                ],
            ],
        ]);

        let exemplar_s = Points::new(array![
            [
                [
                    c64(-0.16238837756090263, -0.6619008482765142),
                    c64(0.3920347312080829, -0.4605997456773211)
                ],
                [
                    c64(1.0074484415236313, -0.4251768661867363),
                    c64(-0.5224983322826633, -0.35087953451103876)
                ],
            ],
            [
                [
                    c64(-0.9976821499977078, 0.019199564121015533),
                    c64(7.579337798670654, -1.0415161603990508)
                ],
                [
                    c64(-0.002578809275813293, 0.008259451506647513),
                    c64(2.25603355998508, 1.0176876789673772)
                ],
            ],
            [
                [
                    c64(0.42410885389642705, -0.3118690796565085),
                    c64(0.5297921201601938, -0.8378163586024442)
                ],
                [
                    c64(0.06678268810628303, -0.014837331705425892),
                    c64(-0.9768883751600902, 0.006192806903744692)
                ],
            ],
        ]);

        let exemplar_t = Points::new(array![
            [
                [c64(0.65946, -0.74758), c64(0.09853999999999996, -0.61542)],
                [
                    c64(0.31545999999999996, 0.48141999999999996),
                    c64(0.84254, 0.35557999999999995)
                ],
            ],
            [
                [
                    c64(36.61038, -284.98399),
                    c64(36.482620000000004, 109.40199000000001)
                ],
                [
                    c64(-34.56262, 283.93701),
                    c64(-34.44438, -110.31901000000002)
                ],
            ],
            [
                [c64(7.38842, -3.9153200000000004), c64(7.04058, -3.10568)],
                [
                    c64(13.95942, 3.00868),
                    c64(14.269580000000001, 3.1703200000000002)
                ],
            ],
        ]);

        let exemplar_y = Points::new(array![
            [
                [
                    c64(-0.6488713074472108, 0.031298266457849534),
                    c64(0.6173872684560134, -0.17863689312523628)
                ],
                [
                    c64(1.1421936598801101, 0.21331749203434683),
                    c64(-1.1503240265701788, 0.09603877517956474)
                ],
            ],
            [
                [
                    c64(-120.72185614425057, 8.805417593823217),
                    c64(236.5668320255695, -133.0463308778349)
                ],
                [
                    c64(0.02242607757067159, 0.30614422700467225),
                    c64(-0.3464501073954702, -0.6034302475961459)
                ],
            ],
            [
                [
                    c64(-0.013848003398757299, -0.013011909266055952),
                    c64(0.438842078293104, 0.4037010033048306)
                ],
                [
                    c64(0.0015952504537115827, 0.04112088271446354),
                    c64(-0.0513369885500035, -0.8763957069757803)
                ],
            ],
        ]);

        let exemplar_z = Points::new(array![
            [
                [
                    c64(-3.9080000000000004, -44.256),
                    c64(-6.988976000000001, -23.729132000000003)
                ],
                [
                    c64(8.000000000000002, -44.0),
                    c64(-1.3239999999999996, -24.968000000000004)
                ],
            ],
            [
                [
                    c64(-0.4455154518952489, -0.4037578874160514),
                    c64(-159.02701232436794, 172.37743628663426)
                ],
                [
                    c64(-0.09997674713938806, -0.24568529275914364),
                    c64(-96.89116746597642, 39.39271161774127)
                ],
            ],
            [
                [
                    c64(79.18805528134254, -78.79582099374794),
                    c64(1.4650241855873656, -75.86275501809806)
                ],
                [
                    c64(3.7841395195788086, -3.61961171437973),
                    c64(0.07222770648239558, -2.416913458374465)
                ],
            ],
        ]);

        let mut exemplars: HashMap<RFParameter, Points> = HashMap::new();
        exemplars.insert(RFParameter::A, exemplar_a);
        exemplars.insert(RFParameter::G, exemplar_g);
        exemplars.insert(RFParameter::H, exemplar_h);
        exemplars.insert(RFParameter::S, exemplar_s.clone());
        exemplars.insert(RFParameter::SPower, exemplar_s.clone());
        exemplars.insert(RFParameter::SPseudo, exemplar_s.clone());
        exemplars.insert(RFParameter::STraveling, exemplar_s.clone());
        exemplars.insert(RFParameter::T, exemplar_t);
        exemplars.insert(RFParameter::Y, exemplar_y);
        exemplars.insert(RFParameter::Z, exemplar_z);

        compare_2ports(&calc, exemplars);
    }

    #[test]
    fn network_chop_in_half() {
        let base = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![[
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(-0.846, 0.158), c64(0.544, -0.129)],
            ]]),
            String::from(""),
            String::from(""),
        );

        let exemplar = Points::new(array![[
            [
                c64(2.1770336894001643, -3.941372226787181),
                c64(1.2297289917865257, -2.9608419909240657)
            ],
            [
                c64(1.2297289917865257, -2.9608419909240657),
                c64(1.3022596548890717, -2.173746918652424)
            ],
        ]]);

        let calc = base.chop_in_half();

        comp_points_c64(&exemplar, &calc.s(), F64Margin::default(), "chop_in_half");
    }

    #[test]
    fn network_connect() {
        let param = RFParameter::S;
        let net_a = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);
        let net_b = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);
        let exemplar = Points::new(array![
            [
                [
                    c64(9.2834313172353100e-01, -2.7765074827479197e-01),
                    c64(8.4694042027146166e-01, -9.6177865696858533e-01)
                ],
                [
                    c64(-5.7791318002886909e-04, 6.4375771265646347e-04),
                    c64(5.2760946315286006e-01, -1.3782565996306906e-01)
                ],
            ],
            [
                [
                    c64(2.0291713136982503, -9.5410855647186832e-01),
                    c64(-6.6518543600706482e-03, 9.8280271970849521e-03)
                ],
                [
                    c64(-1.4935150750155130e-02, 5.4366422143152420e-03),
                    c64(3.4525206818300038, -3.9004984906103607e+02)
                ],
            ],
            [
                [
                    c64(3.2072157651372038e+01, 5.1525524871287964e-01),
                    c64(3.7256238556763940e+01, 5.2436485740325317e+01)
                ],
                [
                    c64(2.9343056016684363e-03, -2.6778112897451523e-03),
                    c64(5.1180048671532596e-01, -4.6914492462426050e-01)
                ],
            ],
        ]);

        let net = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            net_a.clone(),
            String::from(""),
            String::from(""),
        );
        let test = net.connect(1, &net.clone(), 0);

        let margin = F64Margin {
            epsilon: f64::EPSILON,
            ulps: 10,
        };
        comp_points_c64(&exemplar, &test.s(), margin, "connect(1, net, 0)");

        // let param = RFParameter::S;
        // let net_a = array![
        //     array![
        //         [c64(0.295, 0.538), c64(0.258, 0.045)],
        //         [c64(0.696, 0.365), c64(0.184, 0.240)],
        //     ],
        //     array![
        //         [c64(0.445, 0.079), c64(0.482, 0.702)],
        //         [c64(0.264, 0.799), c64(0.129, 0.365)],
        //     ],
        //     array![
        //         [c64(0.633, 0.542), c64(0.845, 0.609)],
        //         [c64(0.773, 0.512), c64(0.408, 0.647)],
        //     ],
        // ];
        // let net_b = array![
        //     array![
        //         [c64(0.679, 0.873), c64(0.634, 0.856)],
        //         [c64(0.333, 0.109), c64(0.419, 0.230)],
        //     ],
        //     array![
        //         [c64(0.522, 0.495), c64(0.262, 0.698)],
        //         [c64(0.757, 0.477), c64(0.506, 0.333)],
        //     ],
        //     array![
        //         [c64(0.906, 0.771), c64(0.756, 0.539)],
        //         [c64(0.531, 0.593), c64(0.025, 0.435)],
        //     ],
        // ];
        // let exemplar = array![
        //     array![
        //         [
        //             c64(0.2385338147292546, 0.7310321147793707),
        //             c64(0.04288150235867945, 0.24272357983452622)
        //         ],
        //         [
        //             c64(0.11267592460978103, 0.21563108786071733),
        //             c64(0.34178830524632003, 0.29311689022820475)
        //         ],
        //     ],
        //     array![
        //         [c64(-0.005, 0.050), c64(-0.412, 0.373)],
        //         [c64(-0.297, 0.588), c64(0.280, 0.313)],
        //     ],
        //     array![
        //         [c64(-0.044, 0.961), c64(-0.227, 0.630)],
        //         [c64(-0.257, 0.441), c64(-0.367, 0.435)],
        //     ],
        // ];

        // let net = Network::new(
        //     Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
        //     array![c64::from(50.0), c64::from(50.0),],
        //     param,
        //     net_a.clone(),
        //     String::from(""),
        //     String::from(""),
        // );
        // let test = net.connect(1, &net.clone(), 0);

        // let margin = F64Margin {
        //     epsilon: f64::EPSILON,
        //     ulps: 10,
        // };
        // comp_points_c64(&exemplar, &test.s(), margin, "connect(1, net2, 0)");
    }

    #[test]
    #[should_panic(expected = "G parameters do not exist for network with 1 port(s)")]
    fn network_g() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::G,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );
    }

    #[test]
    fn network_g_to_2port() {
        let param = RFParameter::G;
        // 2 PortVal, 3 Frequncy
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = Points::new(array![
            [
                [
                    c64(8.000000000000002, -44.0),
                    c64(-1.3239999999999996, -24.968000000000004)
                ],
                [
                    c64(-3.9080000000000004, -44.256),
                    c64(-6.988976000000001, -23.729132000000003)
                ],
            ],
            [
                [
                    c64(-0.09997674713938806, -0.24568529275914364),
                    c64(-96.89116746597642, 39.39271161774127)
                ],
                [
                    c64(-0.4455154518952489, -0.4037578874160514),
                    c64(-159.02701232436794, 172.37743628663426)
                ],
            ],
            [
                [
                    c64(3.7841395195788086, -3.61961171437973),
                    c64(0.07222770648239558, -2.416913458374465)
                ],
                [
                    c64(79.18805528134254, -78.79582099374794),
                    c64(1.4650241855873656, -75.86275501809806)
                ],
            ],
        ]);

        let exemplar_g = exemplar;

        let exemplar_h = Points::new(array![
            [
                [
                    c64(0.9833390626156057, 0.2338279001727904),
                    c64(1.4946015066547766, 0.43245298899150947)
                ],
                [
                    c64(0.011421435247022799, -0.03877831954295688),
                    c64(1.760808278638465, 0.3539213655183655)
                ],
            ],
            [
                [
                    c64(0.4035870224366391, 0.18975760273287182),
                    c64(0.0032113787399704777, 0.001806094940502799)
                ],
                [
                    c64(0.002891212495572124, 0.0031339317167705934),
                    c64(2.2730192220551214e-5, 0.0025635648542889945)
                ],
            ],
            [
                [
                    c64(0.031865528080802376, 3.3671249022544614e-4),
                    c64(1.2342387832408936, -1.1354048751434689)
                ],
                [
                    c64(-2.5446358440632175e-4, -0.013176784898676817),
                    c64(1.058426060538316, 1.0233933039404715)
                ],
            ],
        ]);

        let exemplar_s = Points::new(array![
            [
                [
                    c64(-0.9618560343904763, 0.010242229130219861),
                    c64(0.03351645414457177, 0.0026953265764029797)
                ],
                [
                    c64(-7.419179459091583e-5, 8.704739881097236e-4),
                    c64(-0.9783892386723618, -0.004309216661016642)
                ],
            ],
            [
                [
                    c64(-0.9839686731664778, 0.0074405462121711775),
                    c64(0.00672246279940744, 0.002694605457460855)
                ],
                [
                    c64(-0.0064396973743985455, -0.0053645841552183516),
                    c64(0.9655246034102899, -0.2516226596897059)
                ],
            ],
            [
                [
                    c64(-0.9981331657320738, 4.225025560859947e-5),
                    c64(0.0030558089317656383, -0.044974933250335386)
                ],
                [
                    c64(2.487748668520395e-4, 2.5220520949894193e-4),
                    c64(-0.9804900062487257, -0.018514921069408257)
                ],
            ],
        ]);

        let exemplar_t = Points::new(array![
            [
                [
                    c64(98.21875200000001, 1072.785114),
                    c64(105.18124800000001, 1096.014886)
                ],
                [
                    c64(-90.19227200000002, -1116.285754),
                    c64(-97.20772800000002, -1140.514246)
                ],
            ],
            [
                [
                    c64(-67.45669656371267, 95.76589556616143),
                    c64(89.63249241133573, -75.82368648811801)
                ],
                [
                    c64(69.29454316589282, -96.7994350912754),
                    c64(-91.67029250779464, 76.3658554277137)
                ],
            ],
            [
                [
                    c64(-1977.0775224580455, 1930.1785106120433),
                    c64(-1978.541102089503, 2005.992927360974)
                ],
                [
                    c64(1980.8602174234945, -1933.7497840572555),
                    c64(1982.3266861632117, -2009.6608773445212)
                ],
            ],
        ]);

        let exemplar_y = Points::new(array![
            [
                [
                    c64(0.9625186306094178, -0.22887701590328144),
                    c64(-1.5375603451309596, -0.07416412595936357)
                ],
                [
                    c64(0.0021178781548226505, -0.03993895904049242),
                    c64(1.7403711725430853, 0.41269831113613603)
                ],
            ],
            [
                [
                    c64(2.0291927440451167, -0.9540811006959242),
                    c64(-0.008239667486325092, -6.00998985342353e-4)
                ],
                [
                    c64(0.008856852439349883, 0.0036009002998824815),
                    c64(7.910524066400982e-7, 0.002536004683241709)
                ],
            ],
            [
                [
                    c64(31.378370866300532, -0.3315648611508466),
                    c64(-38.351941918327334, 36.03638546644967)
                ],
                [
                    c64(-0.012353611578814082, -0.41338167219311783),
                    c64(1.5430289329650075, 1.5195786452299769)
                ],
            ],
        ]);

        let exemplar_z = Points::new(array![
            [
                [
                    c64(0.9706839268724422, 0.26648212188669346),
                    c64(0.8633027773921836, 0.07207581467029678)
                ],
                [
                    c64(-0.001979870974017487, 0.0224209748787405),
                    c64(0.5458675431868223, -0.10971903563869077)
                ],
            ],
            [
                [
                    c64(0.39761214735276523, 0.1911185162508152),
                    c64(0.7155757503688567, -1.2463556598814403)
                ],
                [
                    c64(-1.2323927201361262, 1.1168822069634479),
                    c64(3.4584408230318444, -390.05113808702043)
                ],
            ],
            [
                [
                    c64(0.04686626414341519, 9.25064335148286e-4),
                    c64(0.06661043300916779, -1.1371352153266978)
                ],
                [
                    c64(0.006345435959551723, 0.006314005745181268),
                    c64(0.4882940881783893, -0.4721320825578742)
                ],
            ],
        ]);

        let mut exemplars: HashMap<RFParameter, Points> = HashMap::new();
        exemplars.insert(RFParameter::A, exemplar_a);
        exemplars.insert(RFParameter::G, exemplar_g);
        exemplars.insert(RFParameter::H, exemplar_h);
        exemplars.insert(RFParameter::S, exemplar_s.clone());
        exemplars.insert(RFParameter::SPower, exemplar_s.clone());
        exemplars.insert(RFParameter::SPseudo, exemplar_s.clone());
        exemplars.insert(RFParameter::STraveling, exemplar_s.clone());
        exemplars.insert(RFParameter::T, exemplar_t);
        exemplars.insert(RFParameter::Y, exemplar_y);
        exemplars.insert(RFParameter::Z, exemplar_z);

        compare_2ports(&calc, exemplars);
    }

    #[test]
    fn network_gd() {
        let margin = F64Margin {
            epsilon: 1e-15,
            ulps: 4,
        };
        let exemplar: Array1<f64> =
            array![-90.132597176469145, 49.955810487903721, 190.04421815227662];
        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [c64(0.958, -0.263), c64(-0.846, 0.158)],
                    [c64(0.004, 0.022), c64(0.544, -0.129)],
                ],
                [
                    [c64(2.043, -0.982), c64(-0.238, 3.249)],
                    [c64(-1.421, 3.492), c64(0.123, -394.321)],
                ],
                [
                    [c64(21.329, -0.421), c64(-0.942, 24.282)],
                    [c64(0.138, 0.132), c64(0.329, -0.324)],
                ],
            ]),
            String::from(""),
            String::from(""),
        );
        comp_array_f64(
            &exemplar,
            &calc.gd((1, 0), Scale::Pico),
            margin,
            "group_delay1((1,0), Scale::Pico)",
        );

        let exemplar: Array1<f64> = array![
            22.9888888888889,
            22.98888888888887,
            22.894444444444417,
            22.638888888888893,
            24.400000000000016,
            23.338888888888906,
            20.391666666666655,
            20.205555555555552,
            19.566666666666666,
            18.983333333333313,
            18.411111111111136,
            17.724999999999998,
            17.119444444444447,
            16.547222222222256,
            15.438888888888872,
            14.947222222222218,
            15.074999999999998,
            14.502777777777757,
            13.958333333333352,
            13.533333333333339,
            13.036111111111106,
            12.597222222222221,
            12.297222222222222,
            11.822222222222232,
            11.472222222222205,
            11.222222222222234,
            10.841666666666681,
            10.59166666666665,
            10.25833333333334,
            9.89722222222223,
            9.599999999999987,
            9.41944444444445,
            9.258333333333342,
            9.024999999999993,
            8.816666666666654,
            8.547222222222222,
            8.416666666666666,
            8.316666666666656,
            8.008333333333338,
            7.816666666666676,
            7.62222222222222,
            7.452777777777777,
            7.375000000000007,
            7.258333333333353,
            7.216666666666654,
            7.283333333333314,
            7.075000000000029,
            6.552777777777765,
            6.316666666666651,
            6.31388888888892,
            6.358333333333315,
            6.322222222222202,
            6.158333333333363,
            6.19166666666665,
            6.027777777777764,
            5.44722222222223,
            5.391666666666669,
            5.488888888888903,
            5.466666666666677,
            5.355555555555546,
            5.241666666666654,
            5.286111111111132,
            4.994444444444441,
            4.800000000000009,
            4.59166666666671,
            4.730555555555534,
            4.847222222222179,
            5.058333333333322,
            5.252777777777789,
            4.61111111111114,
            4.019444444444422,
            4.127777777777755,
            4.416666666666689,
            4.502777777777801,
            4.444444444444426,
            4.561111111111111,
            4.219444444444448,
            3.547222222222194,
            3.702777777777803,
            3.763888888888915,
            3.824999999999989,
            3.955555555555542,
            4.122222222222214,
            4.172222222222225,
            4.075000000000029,
            3.555555555555541,
            3.136111111111095,
            3.455555555555571,
            3.427777777777795,
            4.33888888888886,
            4.774999999999962,
            3.80833333333335,
            3.575000000000023,
            2.613888888888904,
            1.774999999999973,
            2.588888888888874,
            3.083333333333334,
            3.261111111111103,
            3.175000000000004,
            2.883333333333354,
            2.694444444444462,
            2.363888888888896,
            2.933333333333302,
            3.397222222222213,
            3.099999999999987,
            2.633333333333334,
            2.505555555555571,
            2.219444444444441,
            2.641666666666691,
            2.563888888888893,
            2.333333333333337,
            4.313888888888896,
            3.83611111111109,
            2.519444444444434,
            2.219444444444441,
            1.097222222222228,
            2.900000000000013,
            4.230555555555569,
            1.711111111111109,
            2.738888888888871,
            4.144444444444429,
            3.550000000000017,
            3.850000000000014,
            3.302777777777767,
            1.9861111111111,
            1.716666666666669,
            2.80833333333331,
            3.250000000000021,
            2.391666666666695,
            3.177777777777774,
            3.588888888888911,
            3.477777777777773,
            1.533333333333285,
            0.344444444444431,
            2.147222222222287,
            1.011111111111129,
            1.18055555555551,
            1.841666666666661,
            1.036111111111135,
            0.325000000000002,
            0.105555555555563,
            0.263888888888873,
            0.874999999999975,
            1.825000000000002,
            2.558333333333352,
            3.191666666666662,
            2.894444444444436,
            2.405555555555584,
            2.619444444444456,
            2.24166666666664,
            -0.816666666666696,
            -4.530555555555564,
            -4.330555555555556,
            -2.538888888888887,
            -1.266666666666679,
            0.527777777777742,
            1.552777777777799,
            1.011111111111129,
            0.33888888888889,
            0.258333333333332,
            0.477777777777766,
            0.766666666666678,
            1.141666666666698,
            0.544444444444441,
            -0.300000000000032,
            -0.280555555555567,
            -0.161111111111114,
            0.288888888888911,
            1.786111111111126,
            2.12222222222221,
            0.680555555555551,
            -0.141666666666687,
            0.286111111111106,
            1.074999999999993,
            1.041666666666676,
            0.930555555555589,
            0.452777777777801,
            0.511111111111088,
            0.519444444444435,
            0.630555555555575,
            0.436111111111098,
            0.761111111111109,
            1.375000000000025,
            0.419444444444448,
            0.863888888888866,
            2.113888888888864,
            0.816666666666696,
            -0.661111111111074,
            0.899999999999989,
            2.338888888888878,
            3.288888888888915,
            3.313888888888885,
            2.683333333333257,
            1.705555555555604,
            0.458333333333342,
            0.311111111111079,
            0.725000000000021,
            1.372222222222219,
            1.619444444444434,
            1.650000000000019,
            1.091666666666652,
            0.330555555555543,
            -0.02499999999997,
            -1.536111111111105,
            -2.299999999999975,
            -0.986111111111124,
            -1.036111111111099,
            -2.847222222222267,
            -2.697222222222233,
            -1.722222222222192,
            -1.252777777777768,
            -0.044444444444469,
            1.394444444444419,
            2.683333333333345,
            3.922222222222224,
            3.550000000000017,
            1.936111111111087,
            0.419444444444448,
            -2.163888888888875,
            -3.616666666666581,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.gd((1, 0), Scale::Pico),
            margin,
            "group_delay2((1,0), Scale::Pico)",
        );

        let exemplar: Array1<f64> = array![
            29.5388888888888596784634058095,
            26.0111111111110881427605181177,
            25.8111111111111116857075525184,
            29.1388888888889067643574746108,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![0.5, 1.0, 1.5, 2.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(0.9881388526914863, -0.13442709904013195),
                        c64(0.0010346705444205045, 0.011178864909012504)
                    ],
                    [
                        c64(-7.363542304219899, 0.6742816969789206),
                        c64(0.5574653418486702, -0.06665134724424635)
                    ],
                ],
                [
                    [
                        c64(0.9578079036840927, -0.2633207328693372),
                        c64(0.0037206104781559784, 0.021909191616475577)
                    ],
                    [
                        c64(-7.130124628011368, 1.3277987152036197),
                        c64(0.5435045929943587, -0.12869941397967788)
                    ],
                ],
                [
                    [
                        c64(0.9133108288727866, -0.38508398385543624),
                        c64(0.008042664765986755, 0.03190603796445517)
                    ],
                    [
                        c64(-6.9151682810378095, 1.800750901131042),
                        c64(0.5235871604669029, -0.18886435408156288)
                    ],
                ],
                [
                    [
                        c64(0.849070850314753, -0.49577931076259807),
                        c64(0.01381064392153511, 0.04080882571424955)
                    ],
                    [
                        c64(-6.688405272002992, 2.4133819411904995),
                        c64(0.4942211124266797, -0.24774648346309974)
                    ],
                ],
            ]),
            String::from("test"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:43 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 2.500\n\
            set_Vd(mA) = 11.250\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Vg(V) = -0.520\n\
            Vg(mA) = -0.001\n\
            Vd(V) = 2.500\n\
            Vd(mA) = 11.076",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.gd((1, 0), Scale::Pico),
            margin,
            "group_delay3((1,0), Scale::Pico)",
        );

        let exemplar: Array1<f64> = array![
            23.54444444444442,
            23.411111111111122,
            22.952777777777776,
            22.61944444444443,
            24.294444444444455,
            23.3361111111111,
            20.49722222222222,
            20.086111111111137,
            19.52222222222221,
            18.961111111111098,
            18.444444444444468,
            17.833333333333314,
            17.211111111111105,
            16.561111111111142,
            15.366666666666664,
            14.944444444444446,
            15.091666666666665,
            14.508333333333315,
            14.000000000000016,
            13.641666666666673,
            13.011111111111099,
            12.491666666666653,
            12.34444444444446,
            11.90277777777779,
            11.463888888888864,
            11.194444444444459,
            10.833333333333348,
            10.588888888888876,
            10.27500000000001,
            10.002777777777787,
            9.77222222222221,
            9.344444444444457,
            9.152777777777786,
            9.161111111111095,
            8.955555555555542,
            8.60555555555556,
            8.305555555555573,
            8.161111111111099,
            8.100000000000009,
            7.797222222222234,
            7.527777777777758,
            7.480555555555562,
            7.397222222222224,
            7.261111111111115,
            7.211111111111104,
            7.302777777777769,
            7.122222222222234,
            6.555555555555544,
            6.316666666666678,
            6.38333333333336,
            6.372222222222194,
            6.277777777777766,
            6.150000000000033,
            6.111111111111093,
            5.9861111111111,
            5.494444444444471,
            5.333333333333329,
            5.497222222222197,
            5.447222222222265,
            5.422222222222216,
            5.266666666666625,
            5.36666666666669,
            5.083333333333327,
            4.547222222222219,
            4.580555555555575,
            4.894444444444419,
            5.105555555555528,
            5.102777777777792,
            5.200000000000008,
            4.650000000000016,
            4.122222222222214,
            4.186111111111113,
            4.36111111111112,
            4.488888888888878,
            4.483333333333338,
            4.663888888888904,
            4.083333333333355,
            3.583333333333335,
            3.763888888888862,
            3.766666666666614,
            3.947222222222229,
            3.780555555555591,
            4.072222222222202,
            4.250000000000012,
            4.075000000000029,
            3.577777777777776,
            3.091666666666661,
            3.294444444444455,
            3.261111111111103,
            4.511111111111092,
            4.891666666666642,
            3.605555555555569,
            3.327777777777773,
            3.083333333333328,
            2.113888888888934,
            2.544444444444404,
            3.269444444444383,
            3.125000000000028,
            3.302777777777838,
            3.100000000000022,
            2.936111111111101,
            2.297222222222192,
            2.308333333333309,
            3.247222222222215,
            3.147222222222263,
            2.983333333333377,
            2.327777777777761,
            2.563888888888833,
            2.797222222222228,
            2.041666666666688,
            2.369444444444425,
            3.847222222222243,
            3.963888888888924,
            2.352777777777779,
            2.54722222222221,
            2.552777777777776,
            1.869444444444419,
            2.875000000000008,
            3.352777777777813,
            3.024999999999976,
            3.249999999999989,
            2.602777777777787,
            3.655555555555545,
            3.941666666666654,
            2.591666666666669,
            2.325000000000003,
            3.074999999999987,
            2.747222222222244,
            2.508333333333341,
            2.913888888888866,
            3.122222222222222,
            3.369444444444475,
            1.097222222222218,
            0.030555555555547,
            0.477777777777806,
            1.086111111111111,
            1.480555555555539,
            1.505555555555544,
            1.088888888888881,
            0.244444444444444,
            0.233333333333362,
            0.87777777777779,
            1.480555555555539,
            1.930555555555529,
            2.313888888888873,
            3.130555555555569,
            3.163888888888922,
            2.461111111111101,
            2.69166666666663,
            2.038888888888862,
            -1.213888888888909,
            -4.566666666666652,
            -4.786111111111127,
            -2.855555555555578,
            -1.124999999999994,
            0.505555555555577,
            1.711111111111109,
            0.905555555555566,
            0.250000000000021,
            0.183333333333315,
            0.430555555555561,
            1.141666666666652,
            0.922222222222189,
            0.124999999999993,
            -0.397222222222213,
            -0.647222222222234,
            -0.180555555555543,
            0.441666666666679,
            1.677777777777792,
            2.075000000000005,
            0.919444444444454,
            0.552777777777788,
            0.197222222222202,
            0.616666666666687,
            0.719444444444445,
            0.641666666666645,
            0.488888888888888,
            0.333333333333349,
            0.891666666666678,
            1.23888888888888,
            0.555555555555547,
            0.708333333333327,
            0.822222222222237,
            0.447222222222224,
            0.875000000000019,
            2.202777777777804,
            0.313888888888884,
            -1.944444444444398,
            1.605555555555581,
            2.988888888888847,
            3.222222222222209,
            3.597222222222257,
            3.14444444444436,
            2.847222222222231,
            0.336111111111155,
            -0.236111111111133,
            0.313888888888884,
            1.266666666666691,
            2.588888888888899,
            1.538888888888881,
            0.908333333333336,
            1.197222222222215,
            0.083333333333328,
            -1.713888888888915,
            -2.716666666666645,
            -0.327777777777773,
            -0.983333333333353,
            -2.72777777777778,
            -1.733333333333309,
            -3.255555555555562,
            -1.869444444444455,
            1.033333333333309,
            2.083333333333352,
            2.438888888888901,
            3.702777777777785,
            4.619444444444469,
            1.724999999999929,
            -0.463888888888918,
            -2.466666666666642,
            -4.922222222222105,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.gd((0, 1), Scale::Pico),
            margin,
            "group_delay4((1,0), Scale::Pico)",
        );

        let exemplar: Array1<f64> = array![
            13.666666666666593,
            13.650000000000007,
            13.611111111111148,
            15.374999999999998,
            15.338888888888913,
            13.394444444444446,
            13.361111111111054,
            13.36388888888886,
            13.26666666666672,
            13.027777777777784,
            12.755555555555528,
            12.697222222222234,
            12.619444444444447,
            12.283333333333344,
            11.530555555555551,
            11.458333333333341,
            11.877777777777789,
            11.861111111111095,
            11.694444444444452,
            11.302777777777749,
            11.044444444444435,
            10.886111111111108,
            10.441666666666668,
            10.400000000000023,
            10.422222222222215,
            10.083333333333352,
            9.933333333333335,
            9.80277777777775,
            9.691666666666661,
            9.477777777777792,
            9.238888888888885,
            9.14722222222224,
            8.883333333333342,
            8.777777777777766,
            8.774999999999988,
            8.416666666666671,
            8.308333333333342,
            8.199999999999983,
            8.036111111111115,
            8.041666666666668,
            8.022222222222213,
            7.713888888888904,
            7.638888888888897,
            7.788888888888894,
            7.652777777777765,
            7.6361111111111,
            7.322222222222246,
            6.836111111111103,
            6.736111111111097,
            6.736111111111129,
            6.669444444444436,
            6.694444444444436,
            6.908333333333354,
            6.799999999999987,
            6.09166666666666,
            5.430555555555571,
            5.977777777777776,
            6.538888888888877,
            6.125000000000015,
            6.097222222222222,
            5.930555555555548,
            5.5027777777778,
            5.405555555555548,
            5.447222222222221,
            5.622222222222233,
            5.494444444444427,
            5.405555555555548,
            5.397222222222211,
            5.288888888888885,
            5.338888888888905,
            5.0111111111111,
            4.922222222222221,
            5.405555555555575,
            5.677777777777741,
            5.433333333333334,
            5.369444444444514,
            4.938888888888871,
            4.230555555555512,
            4.405555555555572,
            4.991666666666671,
            4.927777777777771,
            4.675000000000021,
            5.06111111111111,
            5.33888888888887,
            4.82500000000002,
            4.138888888888907,
            4.005555555555533,
            4.563888888888864,
            4.455555555555565,
            3.913888888888911,
            6.841666666666618,
            7.149999999999992,
            4.452777777777778,
            4.272222222222251,
            4.244444444444492,
            3.933333333333304,
            4.124999999999981,
            4.469444444444453,
            3.630555555555557,
            3.988888888888912,
            3.708333333333345,
            2.305555555555521,
            2.766666666666647,
            3.416666666666678,
            3.838888888888896,
            4.244444444444474,
            3.79722222222225,
            3.363888888888881,
            3.341666666666628,
            3.20277777777778,
            3.17777777777781,
            3.655555555555581,
            3.330555555555579,
            3.530555555555519,
            4.347222222222172,
            3.416666666666678,
            3.036111111111123,
            3.213888888888897,
            2.819444444444455,
            4.552777777777756,
            4.927777777777765,
            3.924999999999994,
            4.030555555555559,
            4.238888888888915,
            4.330555555555556,
            4.15277777777774,
            3.508333333333319,
            3.119444444444452,
            4.427777777777771,
            4.188888888888939,
            3.605555555555569,
            2.363888888888861,
            1.130555555555534,
            -0.402777777777789,
            -0.558333333333293,
            2.302777777777791,
            2.705555555555553,
            1.358333333333318,
            0.816666666666661,
            0.758333333333338,
            -0.430555555555566,
            -0.436111111111107,
            0.136111111111109,
            -0.56666666666667,
            0.119444444444416,
            1.386111111111143,
            1.836111111111137,
            1.999999999999988,
            2.505555555555546,
            2.861111111111092,
            3.216666666666668,
            3.755555555555532,
            3.716666666666674,
            3.238888888888938,
            2.694444444444436,
            2.208333333333288,
            2.147222222222251,
            2.163888888888945,
            1.824999999999984,
            0.902777777777725,
            0.577777777777788,
            1.425000000000022,
            1.336111111111096,
            0.716666666666639,
            0.730555555555562,
            0.955555555555577,
            1.50833333333335,
            1.733333333333327,
            2.249999999999974,
            2.349999999999996,
            1.48611111111113,
            1.391666666666683,
            1.702777777777781,
            2.352777777777802,
            2.047222222222194,
            1.672222222222183,
            1.87222222222226,
            1.252777777777768,
            1.069444444444417,
            1.66666666666671,
            2.316666666666634,
            1.900000000000001,
            1.111111111111152,
            1.994444444444412,
            1.666666666666639,
            0.294444444444455,
            1.663888888888904,
            2.555555555555532,
            1.936111111111124,
            1.113888888888887,
            0.736111111111103,
            2.108333333333322,
            2.297222222222205,
            2.577777777777781,
            1.930555555555548,
            1.144444444444469,
            2.569444444444434,
            3.158333333333345,
            3.302777777777802,
            1.680555555555495,
            0.763888888888879,
            0.894444444444484,
            1.26388888888892,
            2.11944444444444,
            2.455555555555476,
            2.688888888888922,
            2.855555555555578,
            4.008333333333324,
            6.363888888888896,
            8.30277777777779,
            7.783333333333391,
            2.805555555555512,
            -5.525000000000035,
            -10.788888888888897,
            -9.719444444444443,
            -5.952777777777831,
            -2.297222222222205,
            -0.716666666666639,
            -0.874999999999984,
            -0.605555555555522,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::Db,
            vec![
                vec![
                    (-0.007, -2.703),
                    (-33.308, 87.205),
                    (-33.310, 87.223),
                    (0.002, -2.522),
                ],
                vec![
                    (-0.014, -5.380),
                    (-27.320, 84.747),
                    (-27.319, 84.763),
                    (-0.006, -5.048),
                ],
                vec![
                    (-0.020, -8.054),
                    (-23.824, 82.282),
                    (-23.822, 82.309),
                    (-0.011, -7.535),
                ],
                vec![
                    (-0.030, -10.721),
                    (-21.356, 79.850),
                    (-21.355, 79.863),
                    (-0.022, -10.008),
                ],
                vec![
                    (-0.070, -12.934),
                    (-19.767, 76.759),
                    (-19.766, 76.774),
                    (-0.073, -12.097),
                ],
                vec![
                    (-0.102, -15.528),
                    (-18.235, 74.354),
                    (-18.233, 74.341),
                    (-0.086, -14.420),
                ],
                vec![
                    (-0.122, -18.012),
                    (-16.968, 71.927),
                    (-16.964, 71.952),
                    (-0.118, -16.765),
                ],
                vec![
                    (-0.156, -20.492),
                    (-15.876, 69.541),
                    (-15.872, 69.531),
                    (-0.143, -19.112),
                ],
                vec![
                    (-0.185, -23.013),
                    (-14.932, 67.149),
                    (-14.928, 67.141),
                    (-0.181, -21.370),
                ],
                vec![
                    (-0.224, -25.457),
                    (-14.103, 64.781),
                    (-14.100, 64.755),
                    (-0.214, -23.659),
                ],
                vec![
                    (-0.265, -27.878),
                    (-13.365, 62.489),
                    (-13.360, 62.451),
                    (-0.244, -25.890),
                ],
                vec![
                    (-0.302, -30.240),
                    (-12.705, 60.203),
                    (-12.703, 60.163),
                    (-0.284, -28.109),
                ],
                vec![
                    (-0.344, -32.603),
                    (-12.109, 57.908),
                    (-12.107, 57.880),
                    (-0.322, -30.298),
                ],
                vec![
                    (-0.387, -34.903),
                    (-11.574, 55.657),
                    (-11.572, 55.620),
                    (-0.364, -32.496),
                ],
                vec![
                    (-0.429, -37.197),
                    (-11.087, 53.446),
                    (-11.086, 53.458),
                    (-0.407, -34.613),
                ],
                vec![
                    (-0.463, -39.356),
                    (-10.675, 51.542),
                    (-10.671, 51.469),
                    (-0.446, -36.438),
                ],
                vec![
                    (-0.515, -41.554),
                    (-10.268, 49.344),
                    (-10.270, 49.333),
                    (-0.488, -38.514),
                ],
                vec![
                    (-0.562, -43.746),
                    (-9.897, 47.250),
                    (-9.897, 47.193),
                    (-0.536, -40.558),
                ],
                vec![
                    (-0.610, -45.867),
                    (-9.555, 45.112),
                    (-9.555, 45.063),
                    (-0.582, -42.571),
                ],
                vec![
                    (-0.660, -47.981),
                    (-9.247, 43.031),
                    (-9.245, 42.983),
                    (-0.630, -44.513),
                ],
                vec![
                    (-0.704, -50.043),
                    (-8.963, 41.024),
                    (-8.964, 40.994),
                    (-0.679, -46.423),
                ],
                vec![
                    (-0.757, -52.064),
                    (-8.703, 39.062),
                    (-8.702, 39.007),
                    (-0.720, -48.330),
                ],
                vec![
                    (-0.799, -54.087),
                    (-8.460, 37.129),
                    (-8.460, 37.075),
                    (-0.767, -50.166),
                ],
                vec![
                    (-0.851, -56.013),
                    (-8.237, 35.258),
                    (-8.238, 35.248),
                    (-0.800, -51.986),
                ],
                vec![
                    (-0.898, -57.881),
                    (-8.025, 33.392),
                    (-8.025, 33.331),
                    (-0.839, -53.849),
                ],
                vec![
                    (-0.939, -59.713),
                    (-7.832, 31.566),
                    (-7.832, 31.496),
                    (-0.882, -55.676),
                ],
                vec![
                    (-0.982, -61.565),
                    (-7.656, 29.737),
                    (-7.658, 29.701),
                    (-0.927, -57.406),
                ],
                vec![
                    (-1.021, -63.357),
                    (-7.495, 27.999),
                    (-7.495, 27.920),
                    (-0.968, -59.135),
                ],
                vec![
                    (-1.056, -65.122),
                    (-7.345, 26.233),
                    (-7.349, 26.172),
                    (-1.010, -60.814),
                ],
                vec![
                    (-1.093, -66.877),
                    (-7.211, 24.513),
                    (-7.210, 24.431),
                    (-1.045, -62.470),
                ],
                vec![
                    (-1.130, -68.611),
                    (-7.084, 22.839),
                    (-7.085, 22.760),
                    (-1.077, -64.114),
                ],
                vec![
                    (-1.166, -70.296),
                    (-6.964, 21.187),
                    (-6.963, 21.105),
                    (-1.113, -65.762),
                ],
                vec![
                    (-1.199, -71.933),
                    (-6.856, 19.554),
                    (-6.855, 19.467),
                    (-1.149, -67.361),
                ],
                vec![
                    (-1.228, -73.544),
                    (-6.761, 17.969),
                    (-6.761, 17.907),
                    (-1.184, -68.913),
                ],
                vec![
                    (-1.258, -75.146),
                    (-6.668, 16.427),
                    (-6.669, 16.307),
                    (-1.209, -70.427),
                ],
                vec![
                    (-1.293, -76.757),
                    (-6.583, 14.877),
                    (-6.587, 14.748),
                    (-1.235, -71.896),
                ],
                vec![
                    (-1.316, -78.293),
                    (-6.503, 13.359),
                    (-6.507, 13.277),
                    (-1.261, -73.391),
                ],
                vec![
                    (-1.339, -79.774),
                    (-6.431, 11.869),
                    (-6.434, 11.757),
                    (-1.286, -74.841),
                ],
                vec![
                    (-1.362, -81.225),
                    (-6.369, 10.428),
                    (-6.372, 10.325),
                    (-1.308, -76.268),
                ],
                vec![
                    (-1.387, -82.666),
                    (-6.312, 8.931),
                    (-6.317, 8.864),
                    (-1.330, -77.697),
                ],
                vec![
                    (-1.405, -84.120),
                    (-6.257, 7.509),
                    (-6.262, 7.430),
                    (-1.347, -79.106),
                ],
                vec![
                    (-1.419, -85.518),
                    (-6.208, 6.101),
                    (-6.213, 5.976),
                    (-1.365, -80.482),
                ],
                vec![
                    (-1.429, -86.867),
                    (-6.165, 4.717),
                    (-6.169, 4.653),
                    (-1.384, -81.874),
                ],
                vec![
                    (-1.447, -88.250),
                    (-6.127, 3.340),
                    (-6.132, 3.226),
                    (-1.400, -83.269),
                ],
                vec![
                    (-1.465, -89.627),
                    (-6.089, 1.959),
                    (-6.092, 1.849),
                    (-1.410, -84.648),
                ],
                vec![
                    (-1.477, -90.985),
                    (-6.063, 0.594),
                    (-6.066, 0.471),
                    (-1.424, -85.964),
                ],
                vec![
                    (-1.481, -92.320),
                    (-6.052, -0.794),
                    (-6.055, -0.900),
                    (-1.441, -87.215),
                ],
                vec![
                    (-1.488, -93.626),
                    (-6.056, -2.034),
                    (-6.060, -2.165),
                    (-1.466, -88.427),
                ],
                vec![
                    (-1.504, -94.858),
                    (-6.038, -3.236),
                    (-6.039, -3.361),
                    (-1.474, -89.665),
                ],
                vec![
                    (-1.520, -96.087),
                    (-6.023, -4.482),
                    (-6.024, -4.590),
                    (-1.486, -90.966),
                ],
                vec![
                    (-1.522, -97.307),
                    (-6.009, -5.661),
                    (-6.008, -5.786),
                    (-1.490, -92.174),
                ],
                vec![
                    (-1.518, -98.492),
                    (-6.002, -6.874),
                    (-6.003, -6.991),
                    (-1.485, -93.349),
                ],
                vec![
                    (-1.517, -99.685),
                    (-5.991, -8.055),
                    (-5.995, -8.196),
                    (-1.487, -94.556),
                ],
                vec![
                    (-1.539, -100.827),
                    (-5.988, -9.359),
                    (-5.989, -9.478),
                    (-1.508, -95.800),
                ],
                vec![
                    (-1.549, -101.915),
                    (-6.011, -10.524),
                    (-6.016, -10.644),
                    (-1.520, -96.906),
                ],
                vec![
                    (-1.535, -103.121),
                    (-6.030, -11.560),
                    (-6.035, -11.671),
                    (-1.513, -97.983),
                ],
                vec![
                    (-1.527, -104.305),
                    (-6.029, -12.498),
                    (-6.032, -12.599),
                    (-1.483, -99.050),
                ],
                vec![
                    (-1.536, -105.323),
                    (-6.025, -13.680),
                    (-6.030, -13.823),
                    (-1.482, -100.338),
                ],
                vec![
                    (-1.543, -106.386),
                    (-6.035, -14.827),
                    (-6.036, -14.953),
                    (-1.508, -101.481),
                ],
                vec![
                    (-1.532, -107.502),
                    (-6.043, -15.925),
                    (-6.049, -16.028),
                    (-1.519, -102.578),
                ],
                vec![
                    (-1.523, -108.630),
                    (-6.073, -17.002),
                    (-6.074, -17.148),
                    (-1.520, -103.680),
                ],
                vec![
                    (-1.517, -109.630),
                    (-6.085, -18.045),
                    (-6.088, -18.163),
                    (-1.507, -104.645),
                ],
                vec![
                    (-1.517, -110.526),
                    (-6.108, -19.017),
                    (-6.115, -19.129),
                    (-1.502, -105.568),
                ],
                vec![
                    (-1.523, -111.551),
                    (-6.133, -19.965),
                    (-6.132, -20.109),
                    (-1.491, -106.576),
                ],
                vec![
                    (-1.520, -112.602),
                    (-6.146, -20.916),
                    (-6.148, -21.090),
                    (-1.497, -107.712),
                ],
                vec![
                    (-1.505, -113.579),
                    (-6.160, -21.929),
                    (-6.164, -22.133),
                    (-1.493, -108.803),
                ],
                vec![
                    (-1.496, -114.428),
                    (-6.187, -22.916),
                    (-6.190, -23.068),
                    (-1.490, -109.691),
                ],
                vec![
                    (-1.492, -115.261),
                    (-6.202, -23.910),
                    (-6.201, -24.079),
                    (-1.478, -110.630),
                ],
                vec![
                    (-1.479, -116.159),
                    (-6.233, -24.888),
                    (-6.235, -25.011),
                    (-1.486, -111.506),
                ],
                vec![
                    (-1.450, -117.227),
                    (-6.269, -25.884),
                    (-6.268, -25.983),
                    (-1.472, -112.436),
                ],
                vec![
                    (-1.415, -118.310),
                    (-6.304, -26.742),
                    (-6.300, -26.933),
                    (-1.449, -113.459),
                ],
                vec![
                    (-1.406, -119.173),
                    (-6.318, -27.627),
                    (-6.319, -27.787),
                    (-1.436, -114.468),
                ],
                vec![
                    (-1.416, -120.155),
                    (-6.333, -28.537),
                    (-6.336, -28.705),
                    (-1.442, -115.341),
                ],
                vec![
                    (-1.412, -121.119),
                    (-6.354, -29.574),
                    (-6.357, -29.733),
                    (-1.429, -116.311),
                ],
                vec![
                    (-1.375, -122.107),
                    (-6.378, -30.598),
                    (-6.386, -30.749),
                    (-1.411, -117.337),
                ],
                vec![
                    (-1.348, -122.983),
                    (-6.417, -31.525),
                    (-6.420, -31.689),
                    (-1.405, -118.305),
                ],
                vec![
                    (-1.340, -123.747),
                    (-6.459, -32.495),
                    (-6.472, -32.682),
                    (-1.397, -119.061),
                ],
                vec![
                    (-1.351, -124.603),
                    (-6.527, -33.267),
                    (-6.525, -33.467),
                    (-1.388, -119.865),
                ],
                vec![
                    (-1.339, -125.547),
                    (-6.566, -34.045),
                    (-6.567, -34.205),
                    (-1.374, -120.594),
                ],
                vec![
                    (-1.313, -126.503),
                    (-6.593, -34.907),
                    (-6.594, -35.053),
                    (-1.332, -121.429),
                ],
                vec![
                    (-1.287, -127.265),
                    (-6.622, -35.784),
                    (-6.631, -36.002),
                    (-1.303, -122.280),
                ],
                vec![
                    (-1.265, -127.923),
                    (-6.640, -36.573),
                    (-6.655, -36.827),
                    (-1.275, -123.026),
                ],
                vec![
                    (-1.276, -128.868),
                    (-6.676, -37.454),
                    (-6.683, -37.685),
                    (-1.248, -124.052),
                ],
                vec![
                    (-1.276, -129.931),
                    (-6.720, -38.438),
                    (-6.734, -38.649),
                    (-1.253, -124.964),
                ],
                vec![
                    (-1.258, -130.950),
                    (-6.775, -39.360),
                    (-6.782, -39.607),
                    (-1.251, -125.870),
                ],
                vec![
                    (-1.250, -131.674),
                    (-6.823, -40.202),
                    (-6.838, -40.386),
                    (-1.227, -126.610),
                ],
                vec![
                    (-1.245, -132.330),
                    (-6.872, -40.937),
                    (-6.868, -41.097),
                    (-1.203, -127.499),
                ],
                vec![
                    (-1.264, -133.188),
                    (-6.919, -41.663),
                    (-6.921, -41.828),
                    (-1.186, -128.275),
                ],
                vec![
                    (-1.241, -133.962),
                    (-6.979, -42.437),
                    (-6.979, -42.740),
                    (-1.181, -129.251),
                ],
                vec![
                    (-1.244, -134.873),
                    (-7.035, -43.207),
                    (-7.046, -43.432),
                    (-1.178, -130.173),
                ],
                vec![
                    (-1.233, -135.444),
                    (-7.079, -43.963),
                    (-7.079, -44.149),
                    (-1.163, -130.927),
                ],
                vec![
                    (-1.204, -137.506),
                    (-7.295, -45.675),
                    (-7.287, -45.895),
                    (-1.120, -132.380),
                ],
                vec![
                    (-1.196, -138.266),
                    (-7.331, -46.540),
                    (-7.340, -46.723),
                    (-1.101, -132.999),
                ],
                vec![
                    (-1.184, -139.237),
                    (-7.406, -47.224),
                    (-7.392, -47.498),
                    (-1.081, -133.941),
                ],
                vec![
                    (-1.156, -139.806),
                    (-7.439, -47.942),
                    (-7.442, -48.261),
                    (-1.066, -134.851),
                ],
                vec![
                    (-1.141, -140.499),
                    (-7.512, -48.823),
                    (-7.507, -49.026),
                    (-1.057, -135.763),
                ],
                vec![
                    (-1.151, -141.070),
                    (-7.544, -49.468),
                    (-7.552, -49.677),
                    (-1.033, -136.450),
                ],
                vec![
                    (-1.139, -141.988),
                    (-7.619, -50.241),
                    (-7.621, -50.511),
                    (-1.060, -137.163),
                ],
                vec![
                    (-1.126, -142.624),
                    (-7.678, -51.043),
                    (-7.661, -51.286),
                    (-1.045, -138.047),
                ],
                vec![
                    (-1.090, -143.317),
                    (-7.754, -51.827),
                    (-7.754, -51.818),
                    (-1.059, -138.695),
                ],
                vec![
                    (-1.090, -143.804),
                    (-7.812, -52.472),
                    (-7.801, -52.722),
                    (-0.973, -139.574),
                ],
                vec![
                    (-1.082, -144.444),
                    (-7.886, -53.018),
                    (-7.910, -53.153),
                    (-0.968, -140.498),
                ],
                vec![
                    (-1.091, -145.084),
                    (-7.937, -53.366),
                    (-7.948, -53.552),
                    (-0.981, -140.970),
                ],
                vec![
                    (-1.073, -145.770),
                    (-7.979, -53.983),
                    (-7.999, -54.149),
                    (-0.994, -141.596),
                ],
                vec![
                    (-1.063, -146.276),
                    (-8.043, -54.559),
                    (-8.026, -54.782),
                    (-1.008, -142.399),
                ],
                vec![
                    (-1.042, -146.836),
                    (-8.078, -55.281),
                    (-8.074, -55.531),
                    (-1.001, -143.284),
                ],
                vec![
                    (-1.033, -147.673),
                    (-8.132, -56.080),
                    (-8.110, -56.310),
                    (-0.981, -143.668),
                ],
                vec![
                    (-1.042, -148.451),
                    (-8.160, -56.728),
                    (-8.174, -56.898),
                    (-0.965, -144.102),
                ],
                vec![
                    (-1.024, -148.900),
                    (-8.249, -57.285),
                    (-8.226, -57.521),
                    (-0.941, -144.796),
                ],
                vec![
                    (-0.995, -149.415),
                    (-8.286, -57.910),
                    (-8.272, -58.101),
                    (-0.938, -145.596),
                ],
                vec![
                    (-0.989, -150.029),
                    (-8.310, -58.664),
                    (-8.327, -58.674),
                    (-0.907, -146.361),
                ],
                vec![
                    (-0.991, -150.499),
                    (-8.371, -59.225),
                    (-8.362, -59.245),
                    (-0.863, -146.713),
                ],
                vec![
                    (-1.005, -150.999),
                    (-8.404, -59.719),
                    (-8.422, -59.990),
                    (-0.836, -147.464),
                ],
                vec![
                    (-0.991, -151.557),
                    (-8.484, -60.340),
                    (-8.449, -60.444),
                    (-0.834, -148.351),
                ],
                vec![
                    (-0.992, -152.156),
                    (-8.521, -61.244),
                    (-8.512, -61.261),
                    (-0.813, -149.016),
                ],
                vec![
                    (-0.994, -152.469),
                    (-8.563, -61.662),
                    (-8.583, -62.009),
                    (-0.825, -149.822),
                ],
                vec![
                    (-0.996, -152.775),
                    (-8.620, -62.219),
                    (-8.625, -62.491),
                    (-0.844, -150.308),
                ],
                vec![
                    (-0.998, -153.136),
                    (-8.656, -62.852),
                    (-8.703, -63.102),
                    (-0.866, -150.828),
                ],
                vec![
                    (-0.956, -153.739),
                    (-8.707, -63.513),
                    (-8.698, -63.648),
                    (-0.869, -151.174),
                ],
                vec![
                    (-0.964, -154.217),
                    (-8.762, -64.131),
                    (-8.749, -64.117),
                    (-0.803, -151.983),
                ],
                vec![
                    (-0.949, -154.618),
                    (-8.809, -64.854),
                    (-8.772, -65.287),
                    (-0.774, -152.579),
                ],
                vec![
                    (-0.920, -155.032),
                    (-8.834, -65.647),
                    (-8.851, -65.891),
                    (-0.753, -153.294),
                ],
                vec![
                    (-0.903, -155.591),
                    (-8.860, -66.464),
                    (-8.894, -66.700),
                    (-0.741, -154.031),
                ],
                vec![
                    (-0.907, -156.265),
                    (-8.938, -67.399),
                    (-8.959, -67.342),
                    (-0.720, -154.847),
                ],
                vec![
                    (-0.896, -156.802),
                    (-9.031, -67.658),
                    (-9.033, -68.226),
                    (-0.692, -155.857),
                ],
                vec![
                    (-0.856, -157.201),
                    (-9.084, -68.597),
                    (-9.143, -68.901),
                    (-0.706, -156.318),
                ],
                vec![
                    (-0.874, -157.546),
                    (-9.202, -69.240),
                    (-9.200, -69.721),
                    (-0.727, -156.732),
                ],
                vec![
                    (-0.922, -158.150),
                    (-9.258, -69.906),
                    (-9.279, -70.164),
                    (-0.807, -157.422),
                ],
                vec![
                    (-0.947, -158.704),
                    (-9.308, -70.926),
                    (-9.333, -70.844),
                    (-0.835, -158.104),
                ],
                vec![
                    (-0.895, -159.033),
                    (-9.413, -71.526),
                    (-9.401, -71.758),
                    (-0.804, -158.610),
                ],
                vec![
                    (-0.854, -159.318),
                    (-9.522, -72.082),
                    (-9.547, -72.352),
                    (-0.752, -158.691),
                ],
                vec![
                    (-0.838, -159.873),
                    (-9.683, -72.618),
                    (-9.692, -73.056),
                    (-0.775, -159.259),
                ],
                vec![
                    (-0.849, -160.766),
                    (-9.746, -73.258),
                    (-9.704, -73.203),
                    (-0.850, -160.019),
                ],
                vec![
                    (-0.846, -161.398),
                    (-9.808, -73.008),
                    (-9.869, -73.463),
                    (-0.825, -161.307),
                ],
                vec![
                    (-0.756, -161.361),
                    (-9.948, -73.728),
                    (-9.931, -73.058),
                    (-0.713, -161.524),
                ],
                vec![
                    (-0.761, -162.046),
                    (-10.012, -74.283),
                    (-9.980, -73.262),
                    (-0.727, -162.222),
                ],
                vec![
                    (-0.760, -162.772),
                    (-10.097, -74.685),
                    (-10.090, -73.887),
                    (-0.778, -163.039),
                ],
                vec![
                    (-0.759, -163.211),
                    (-10.197, -74.987),
                    (-10.190, -74.236),
                    (-0.827, -163.492),
                ],
                vec![
                    (-0.743, -163.534),
                    (-10.288, -75.174),
                    (-10.306, -74.376),
                    (-0.845, -163.829),
                ],
                vec![
                    (-0.718, -164.022),
                    (-10.406, -75.414),
                    (-10.419, -74.530),
                    (-0.869, -164.190),
                ],
                vec![
                    (-0.713, -164.578),
                    (-10.501, -75.436),
                    (-10.514, -74.649),
                    (-0.882, -164.538),
                ],
                vec![
                    (-0.721, -164.938),
                    (-10.563, -75.321),
                    (-10.556, -74.375),
                    (-0.872, -164.822),
                ],
                vec![
                    (-0.740, -165.270),
                    (-10.616, -75.215),
                    (-10.615, -74.492),
                    (-0.882, -164.933),
                ],
                vec![
                    (-0.727, -165.532),
                    (-10.636, -75.164),
                    (-10.632, -74.424),
                    (-0.853, -165.150),
                ],
                vec![
                    (-0.704, -166.050),
                    (-10.623, -75.065),
                    (-10.634, -74.288),
                    (-0.849, -165.342),
                ],
                vec![
                    (-0.701, -166.646),
                    (-10.619, -75.230),
                    (-10.623, -74.467),
                    (-0.840, -165.459),
                ],
                vec![
                    (-0.696, -167.209),
                    (-10.617, -75.602),
                    (-10.618, -74.787),
                    (-0.809, -165.838),
                ],
                vec![
                    (-0.716, -167.810),
                    (-10.631, -75.931),
                    (-10.637, -75.128),
                    (-0.774, -166.238),
                ],
                vec![
                    (-0.741, -168.274),
                    (-10.612, -76.177),
                    (-10.622, -75.507),
                    (-0.745, -166.740),
                ],
                vec![
                    (-0.764, -168.664),
                    (-10.584, -76.727),
                    (-10.596, -76.030),
                    (-0.731, -167.341),
                ],
                vec![
                    (-0.765, -168.885),
                    (-10.590, -77.366),
                    (-10.609, -76.537),
                    (-0.708, -167.767),
                ],
                vec![
                    (-0.772, -169.145),
                    (-10.629, -77.899),
                    (-10.640, -77.188),
                    (-0.683, -168.348),
                ],
                vec![
                    (-0.750, -169.444),
                    (-10.667, -78.526),
                    (-10.673, -77.889),
                    (-0.677, -169.053),
                ],
                vec![
                    (-0.736, -169.760),
                    (-10.715, -79.048),
                    (-10.721, -78.526),
                    (-0.659, -169.607),
                ],
                vec![
                    (-0.729, -169.988),
                    (-10.774, -79.624),
                    (-10.776, -79.055),
                    (-0.645, -170.279),
                ],
                vec![
                    (-0.726, -170.132),
                    (-10.837, -80.185),
                    (-10.837, -79.496),
                    (-0.652, -170.933),
                ],
                vec![
                    (-0.700, -170.491),
                    (-10.903, -80.593),
                    (-10.902, -79.850),
                    (-0.681, -171.309),
                ],
                vec![
                    (-0.666, -170.890),
                    (-10.965, -81.065),
                    (-10.973, -80.269),
                    (-0.694, -171.561),
                ],
                vec![
                    (-0.659, -171.335),
                    (-11.040, -81.499),
                    (-11.050, -80.629),
                    (-0.707, -171.762),
                ],
                vec![
                    (-0.626, -171.867),
                    (-11.113, -81.675),
                    (-11.129, -80.926),
                    (-0.693, -172.083),
                ],
                vec![
                    (-0.607, -172.055),
                    (-11.179, -81.751),
                    (-11.191, -80.954),
                    (-0.685, -172.412),
                ],
                vec![
                    (-0.604, -172.623),
                    (-11.229, -82.010),
                    (-11.237, -81.134),
                    (-0.692, -172.867),
                ],
                vec![
                    (-0.581, -173.185),
                    (-11.233, -82.302),
                    (-11.249, -81.467),
                    (-0.727, -173.284),
                ],
                vec![
                    (-0.555, -173.548),
                    (-11.270, -82.562),
                    (-11.284, -81.615),
                    (-0.719, -173.825),
                ],
                vec![
                    (-0.529, -174.035),
                    (-11.295, -82.750),
                    (-11.294, -81.725),
                    (-0.699, -174.420),
                ],
                vec![
                    (-0.493, -174.425),
                    (-11.329, -82.919),
                    (-11.331, -81.878),
                    (-0.714, -174.634),
                ],
                vec![
                    (-0.474, -174.909),
                    (-11.354, -83.028),
                    (-11.362, -82.069),
                    (-0.722, -174.945),
                ],
                vec![
                    (-0.463, -175.232),
                    (-11.350, -83.348),
                    (-11.367, -82.421),
                    (-0.717, -175.277),
                ],
                vec![
                    (-0.432, -175.548),
                    (-11.368, -83.709),
                    (-11.363, -82.693),
                    (-0.739, -175.601),
                ],
                vec![
                    (-0.393, -176.181),
                    (-11.396, -84.163),
                    (-11.398, -83.231),
                    (-0.752, -175.840),
                ],
                vec![
                    (-0.385, -176.735),
                    (-11.445, -84.475),
                    (-11.452, -83.539),
                    (-0.736, -176.095),
                ],
                vec![
                    (-0.378, -177.418),
                    (-11.479, -84.621),
                    (-11.486, -83.766),
                    (-0.728, -176.458),
                ],
                vec![
                    (-0.357, -178.081),
                    (-11.505, -84.919),
                    (-11.501, -84.040),
                    (-0.745, -176.797),
                ],
                vec![
                    (-0.373, -178.550),
                    (-11.532, -85.329),
                    (-11.527, -84.379),
                    (-0.737, -177.162),
                ],
                vec![
                    (-0.398, -179.219),
                    (-11.555, -85.876),
                    (-11.537, -84.887),
                    (-0.712, -177.517),
                ],
                vec![
                    (-0.406, -179.387),
                    (-11.592, -86.304),
                    (-11.596, -85.116),
                    (-0.697, -178.130),
                ],
                vec![
                    (-0.388, -179.751),
                    (-11.608, -86.455),
                    (-11.617, -85.489),
                    (-0.708, -178.407),
                ],
                vec![
                    (-0.351, 179.972),
                    (-11.662, -86.823),
                    (-11.666, -85.790),
                    (-0.737, -178.606),
                ],
                vec![
                    (-0.309, 179.552),
                    (-11.715, -87.019),
                    (-11.706, -85.940),
                    (-0.744, -178.758),
                ],
                vec![
                    (-0.265, 179.038),
                    (-11.732, -87.209),
                    (-11.729, -86.175),
                    (-0.744, -178.952),
                ],
                vec![
                    (-0.268, 178.546),
                    (-11.731, -87.406),
                    (-11.728, -86.540),
                    (-0.754, -179.276),
                ],
                vec![
                    (-0.257, 178.084),
                    (-11.753, -87.778),
                    (-11.749, -87.009),
                    (-0.785, -179.361),
                ],
                vec![
                    (-0.252, 177.567),
                    (-11.757, -88.069),
                    (-11.774, -87.224),
                    (-0.760, -179.478),
                ],
                vec![
                    (-0.264, 177.133),
                    (-11.802, -88.337),
                    (-11.792, -87.409),
                    (-0.743, -179.786),
                ],
                vec![
                    (-0.284, 176.662),
                    (-11.827, -88.625),
                    (-11.825, -87.942),
                    (-0.737, 179.938),
                ],
                vec![
                    (-0.281, 176.114),
                    (-11.838, -88.888),
                    (-11.841, -88.009),
                    (-0.737, 179.650),
                ],
                vec![
                    (-0.284, 175.439),
                    (-11.849, -89.096),
                    (-11.828, -88.048),
                    (-0.699, 179.298),
                ],
                vec![
                    (-0.315, 174.679),
                    (-11.814, -89.502),
                    (-11.828, -88.608),
                    (-0.682, 178.625),
                ],
                vec![
                    (-0.346, 174.031),
                    (-11.835, -89.972),
                    (-11.824, -88.968),
                    (-0.720, 178.373),
                ],
                vec![
                    (-0.338, 173.956),
                    (-11.827, -90.109),
                    (-11.831, -89.305),
                    (-0.713, 178.598),
                ],
                vec![
                    (-0.273, 173.583),
                    (-11.855, -90.206),
                    (-11.862, -89.369),
                    (-0.648, 178.595),
                ],
                vec![
                    (-0.266, 173.005),
                    (-11.846, -90.457),
                    (-11.845, -89.570),
                    (-0.609, 178.294),
                ],
                vec![
                    (-0.232, 172.456),
                    (-11.843, -90.902),
                    (-11.855, -90.128),
                    (-0.612, 177.855),
                ],
                vec![
                    (-0.223, 172.030),
                    (-11.847, -91.056),
                    (-11.850, -90.397),
                    (-0.575, 177.768),
                ],
                vec![
                    (-0.227, 171.491),
                    (-11.866, -91.933),
                    (-11.857, -91.056),
                    (-0.561, 177.593),
                ],
                vec![
                    (-0.211, 170.829),
                    (-11.908, -92.002),
                    (-11.879, -91.092),
                    (-0.489, 177.305),
                ],
                vec![
                    (-0.233, 170.119),
                    (-11.866, -92.554),
                    (-11.893, -91.468),
                    (-0.476, 176.579),
                ],
                vec![
                    (-0.204, 169.606),
                    (-11.915, -92.825),
                    (-11.903, -92.017),
                    (-0.448, 176.373),
                ],
                vec![
                    (-0.235, 168.777),
                    (-11.933, -93.385),
                    (-11.912, -92.605),
                    (-0.420, 175.689),
                ],
                vec![
                    (-0.271, 167.832),
                    (-11.947, -93.891),
                    (-11.939, -93.206),
                    (-0.472, 175.246),
                ],
                vec![
                    (-0.306, 167.102),
                    (-11.957, -94.149),
                    (-11.954, -93.210),
                    (-0.448, 175.324),
                ],
                vec![
                    (-0.258, 166.964),
                    (-11.928, -94.426),
                    (-11.944, -93.481),
                    (-0.419, 175.523),
                ],
                vec![
                    (-0.229, 165.980),
                    (-11.950, -94.223),
                    (-11.941, -93.532),
                    (-0.339, 175.625),
                ],
                vec![
                    (-0.213, 165.242),
                    (-11.921, -94.718),
                    (-11.931, -93.936),
                    (-0.281, 175.172),
                ],
                vec![
                    (-0.198, 164.306),
                    (-11.915, -95.109),
                    (-11.905, -94.295),
                    (-0.217, 174.767),
                ],
                vec![
                    (-0.188, 163.445),
                    (-11.865, -95.670),
                    (-11.879, -94.820),
                    (-0.205, 174.577),
                ],
                vec![
                    (-0.181, 162.394),
                    (-11.874, -96.178),
                    (-11.869, -95.263),
                    (-0.135, 174.270),
                ],
                vec![
                    (-0.234, 161.079),
                    (-11.820, -96.781),
                    (-11.834, -95.848),
                    (-0.079, 173.670),
                ],
                vec![
                    (-0.311, 159.862),
                    (-11.797, -97.531),
                    (-11.792, -96.706),
                    (-0.026, 173.252),
                ],
                vec![
                    (-0.473, 158.035),
                    (-11.767, -99.069),
                    (-11.777, -98.139),
                    (0.031, 172.884),
                ],
                vec![
                    (-0.865, 156.756),
                    (-11.916, -100.603),
                    (-11.916, -99.695),
                    (0.055, 172.980),
                ],
                vec![
                    (-1.383, 157.170),
                    (-12.183, -101.839),
                    (-12.185, -100.941),
                    (0.148, 172.980),
                ],
                vec![
                    (-1.786, 159.984),
                    (-12.561, -101.601),
                    (-12.577, -100.705),
                    (0.301, 173.255),
                ],
                vec![
                    (-1.716, 163.563),
                    (-12.816, -99.966),
                    (-12.805, -98.952),
                    (0.502, 172.921),
                ],
                vec![
                    (-1.393, 165.231),
                    (-12.859, -97.814),
                    (-12.853, -96.821),
                    (0.657, 171.744),
                ],
                vec![
                    (-1.030, 165.580),
                    (-12.700, -96.201),
                    (-12.669, -95.453),
                    (0.747, 170.716),
                ],
                vec![
                    (-0.759, 164.875),
                    (-12.514, -95.622),
                    (-12.508, -94.678),
                    (0.780, 169.512),
                ],
                vec![
                    (-0.585, 164.044),
                    (-12.361, -95.610),
                    (-12.335, -94.626),
                    (0.755, 168.895),
                ],
                vec![
                    (-0.439, 163.431),
                    (-12.203, -95.395),
                    (-12.209, -94.420),
                    (0.799, 168.418),
                ],
                vec![
                    (-0.318, 162.847),
                    (-12.067, -95.224),
                    (-12.068, -94.311),
                    (0.841, 167.973),
                ],
            ],
            String::from("test_3"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:48 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Vg(V) = -1.500\n\
            Vg(mA) = -0.000\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.gd((1, 0), Scale::Pico),
            margin,
            "group_delay5((1,0), Scale::Pico)",
        );
    }

    #[test]
    #[should_panic(expected = "H parameters do not exist for network with 1 port(s)")]
    fn network_h() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::H,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );
    }

    #[test]
    fn network_h_to_2port() {
        let param = RFParameter::H;
        // 2 PortVal, 3 Frequncy
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = Points::new(array![
            [
                [
                    c64(6.988976000000001, 23.729132000000003),
                    c64(3.9080000000000004, 44.256)
                ],
                [
                    c64(1.3239999999999996, 24.968000000000004),
                    c64(-8.000000000000002, 44.0)
                ],
            ],
            [
                [
                    c64(159.02701232436794, -172.37743628663426),
                    c64(0.4455154518952489, 0.4037578874160514)
                ],
                [
                    c64(96.89116746597642, -39.39271161774127),
                    c64(0.09997674713938806, 0.24568529275914364)
                ],
            ],
            [
                [
                    c64(-1.4650241855873656, 75.86275501809806),
                    c64(-79.18805528134254, 78.79582099374794)
                ],
                [
                    c64(-0.07222770648239558, 2.416913458374465),
                    c64(-3.7841395195788086, 3.61961171437973)
                ],
            ],
        ]);

        let exemplar_g = Points::new(array![
            [
                [
                    c64(0.9833390626156057, 0.2338279001727904),
                    c64(1.4946015066547766, 0.43245298899150947)
                ],
                [
                    c64(0.011421435247022799, -0.03877831954295688),
                    c64(1.760808278638465, 0.3539213655183655)
                ],
            ],
            [
                [
                    c64(0.4035870224366391, 0.18975760273287182),
                    c64(0.0032113787399704777, 0.001806094940502799)
                ],
                [
                    c64(0.002891212495572124, 0.0031339317167705934),
                    c64(2.2730192220551214e-5, 0.0025635648542889945)
                ],
            ],
            [
                [
                    c64(0.031865528080802376, 3.3671249022544614e-4),
                    c64(1.2342387832408936, -1.1354048751434689)
                ],
                [
                    c64(-2.5446358440632175e-4, -0.013176784898676817),
                    c64(1.058426060538316, 1.0233933039404715)
                ],
            ],
        ]);

        let exemplar_h = exemplar;

        let exemplar_s = Points::new(array![
            [
                [
                    c64(-0.9621821592738881, -0.00885792224980608),
                    c64(-0.058317980244527175, -0.0026106380721724093)
                ],
                [
                    c64(7.507205978746872e-5, -0.001514866113737432),
                    c64(-0.9325997615470685, 0.015370145386657447)
                ],
            ],
            [
                [
                    c64(-0.9213517886382416, -0.03523260573923969),
                    c64(-3.1620396207720044e-4, -2.8884670276948945e-5)
                ],
                [
                    c64(3.378125320769322e-4, 1.4459578826854345e-4),
                    c64(-0.9999999092802149, 1.0146807004710654e-4)
                ],
            ],
            [
                [
                    c64(-0.2334048539059044, -0.009454925542048681),
                    c64(-0.8983610210834206, 0.8817869100934754)
                ],
                [
                    c64(-5.110454447550917e-4, -0.009879263991141238),
                    c64(-0.9459287648258871, 0.05037575636851103)
                ],
            ],
        ]);

        let exemplar_t = Points::new(array![
            [
                [
                    c64(-33.64459199999999, -590.777994),
                    c64(-25.56643199999999, -633.8928740000001)
                ],
                [
                    c64(40.55540799999999, 613.622006),
                    c64(32.63356799999999, 658.5071260000001)
                ],
            ],
            [
                [
                    c64(-2342.720147268176, 898.74787736772),
                    c64(-2342.8112137062776, 898.5102672327091)
                ],
                [
                    c64(2501.738249283506, -1071.1333888121026),
                    c64(2501.8471363396834, -1070.879628361595)
                ],
            ],
            [
                [
                    c64(-0.027008637709772073, -21.469611303060216),
                    c64(2.1733697762421857, -23.513306597564988)
                ],
                [
                    c64(0.14574555774925743, 95.75644990128332),
                    c64(-5.222155067456402, 100.951978035538)
                ],
            ],
        ]);

        let exemplar_y = Points::new(array![
            [
                [
                    c64(0.9706839268724422, 0.26648212188669346),
                    c64(0.8633027773921836, 0.07207581467029678)
                ],
                [
                    c64(-0.001979870974017487, 0.0224209748787405),
                    c64(0.5458675431868223, -0.10971903563869077)
                ],
            ],
            [
                [
                    c64(0.39761214735276523, 0.1911185162508152),
                    c64(0.7155757503688567, -1.2463556598814403)
                ],
                [
                    c64(-1.2323927201361262, 1.1168822069634479),
                    c64(3.4584408230318444, -390.05113808702043)
                ],
            ],
            [
                [
                    c64(0.04686626414341519, 9.25064335148286e-4),
                    c64(0.06661043300916779, -1.1371352153266978)
                ],
                [
                    c64(0.006345435959551723, 0.006314005745181268),
                    c64(0.4882940881783893, -0.4721320825578742)
                ],
            ],
        ]);

        let exemplar_z = Points::new(array![
            [
                [
                    c64(0.9625186306094178, -0.22887701590328144),
                    c64(-1.5375603451309596, -0.07416412595936357)
                ],
                [
                    c64(0.0021178781548226505, -0.03993895904049242),
                    c64(1.7403711725430853, 0.41269831113613603)
                ],
            ],
            [
                [
                    c64(2.0291927440451167, -0.9540811006959242),
                    c64(-0.008239667486325092, -6.00998985342353e-4)
                ],
                [
                    c64(0.008856852439349883, 0.0036009002998824815),
                    c64(7.910524066400982e-7, 0.002536004683241709)
                ],
            ],
            [
                [
                    c64(31.378370866300532, -0.3315648611508466),
                    c64(-38.351941918327334, 36.03638546644967)
                ],
                [
                    c64(-0.012353611578814082, -0.41338167219311783),
                    c64(1.5430289329650075, 1.5195786452299769)
                ],
            ],
        ]);

        let mut exemplars: HashMap<RFParameter, Points> = HashMap::new();
        exemplars.insert(RFParameter::A, exemplar_a);
        exemplars.insert(RFParameter::G, exemplar_g);
        exemplars.insert(RFParameter::H, exemplar_h);
        exemplars.insert(RFParameter::S, exemplar_s.clone());
        exemplars.insert(RFParameter::SPower, exemplar_s.clone());
        exemplars.insert(RFParameter::SPseudo, exemplar_s.clone());
        exemplars.insert(RFParameter::STraveling, exemplar_s.clone());
        exemplars.insert(RFParameter::T, exemplar_t);
        exemplars.insert(RFParameter::Y, exemplar_y);
        exemplars.insert(RFParameter::Z, exemplar_z);

        compare_2ports(&calc, exemplars);
    }

    #[test]
    fn network_k() {
        let margin = F64Margin {
            epsilon: 1e-12,
            ulps: 4,
        };

        let exemplar: Array1<f64> = array![
            1.6755019249092201e-01,
            2.5496798991797386e+04,
            -2.6328854393938734e+01,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [c64(0.958, -0.263), c64(-0.846, 0.158)],
                    [c64(0.004, 0.022), c64(0.544, -0.129)],
                ],
                [
                    [c64(2.043, -0.982), c64(-0.238, 3.249)],
                    [c64(-1.421, 3.492), c64(0.123, -394.321)],
                ],
                [
                    [c64(21.329, -0.421), c64(-0.942, 24.282)],
                    [c64(0.138, 0.132), c64(0.329, -0.324)],
                ],
            ]),
            String::from(""),
            String::from(""),
        );
        comp_array_f64(&exemplar, &calc.k(), margin, "k(1)");

        // test.s2p
        let exemplar: Array1<f64> = array![
            0.024742798510455,
            0.026829555623131,
            0.010327404229363,
            0.019609843571699,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![0.5, 1.0, 1.5, 2.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(0.9881388526914863, -0.13442709904013195),
                        c64(0.0010346705444205045, 0.011178864909012504)
                    ],
                    [
                        c64(-7.363542304219899, 0.6742816969789206),
                        c64(0.5574653418486702, -0.06665134724424635)
                    ],
                ],
                [
                    [
                        c64(0.9578079036840927, -0.2633207328693372),
                        c64(0.0037206104781559784, 0.021909191616475577)
                    ],
                    [
                        c64(-7.130124628011368, 1.3277987152036197),
                        c64(0.5435045929943587, -0.12869941397967788)
                    ],
                ],
                [
                    [
                        c64(0.9133108288727866, -0.38508398385543624),
                        c64(0.008042664765986755, 0.03190603796445517)
                    ],
                    [
                        c64(-6.9151682810378095, 1.800750901131042),
                        c64(0.5235871604669029, -0.18886435408156288)
                    ],
                ],
                [
                    [
                        c64(0.849070850314753, -0.49577931076259807),
                        c64(0.01381064392153511, 0.04080882571424955)
                    ],
                    [
                        c64(-6.688405272002992, 2.4133819411904995),
                        c64(0.4942211124266797, -0.24774648346309974)
                    ],
                ],
            ]),
            String::from("test"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:43 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 2.500\n\
            set_Vd(mA) = 11.250\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Vg(V) = -0.520\n\
            Vg(mA) = -0.001\n\
            Vd(V) = 2.500\n\
            Vd(mA) = 11.076",
            ),
        );
        comp_array_f64(&exemplar, &calc.k(), margin, "k(2)");

        // test_2.s2p
        let exemplar: Array1<f64> = array![
            0.999898913654026,
            1.000010680478558,
            1.000186508060361,
            1.000303615413879,
            1.000428833054145,
            1.001581502316822,
            1.00148573456698,
            1.001421859962389,
            1.001404720346898,
            1.0014343273456,
            1.001374163081635,
            1.001377097831651,
            1.00139673910043,
            1.001411693469628,
            1.001413793237645,
            1.00145187027887,
            1.001564242468825,
            1.001670150595793,
            1.001744364814426,
            1.001870881600432,
            1.001966957807714,
            1.002074666619497,
            1.002096651571681,
            1.002132765688713,
            1.002266685418098,
            1.00236200170342,
            1.002415839638551,
            1.002484815245632,
            1.002547717983076,
            1.002706169890626,
            1.002773804164093,
            1.002847566987383,
            1.003008115940121,
            1.003004241151958,
            1.00314444756832,
            1.003308564047208,
            1.003327225410743,
            1.00344836347437,
            1.003493454860496,
            1.003669034580217,
            1.003745084930169,
            1.003785414493464,
            1.003808985351502,
            1.003745339747983,
            1.003900500781349,
            1.004012844778619,
            1.004062497625225,
            1.004398984402274,
            1.004368136889469,
            1.00443132375476,
            1.004616149245176,
            1.004842399757907,
            1.004959295162926,
            1.004945633813386,
            1.005398978715728,
            1.00564615545579,
            1.005569208120365,
            1.005862364959928,
            1.005763357292168,
            1.006073828314022,
            1.006410165148961,
            1.006749073096247,
            1.006817086848425,
            1.007063774383798,
            1.007179228087291,
            1.007615355666958,
            1.0079419644119,
            1.007847984621936,
            1.007392386935432,
            1.007690200981694,
            1.008284572757275,
            1.008177638433809,
            1.008056598751744,
            1.008053922966132,
            1.009128807352613,
            1.010130646897936,
            1.009982191798923,
            1.010140857557463,
            1.010183363477247,
            1.010158641638911,
            1.0102344275253,
            1.009913156238003,
            1.010528542196402,
            1.009625195892039,
            1.009643000757466,
            1.009371720561737,
            1.009996926380428,
            1.011083821531034,
            1.012043071006805,
            1.0126235204628,
            1.016249606512045,
            1.015677016481509,
            1.013857729009161,
            1.013777912580565,
            1.015590888057388,
            1.017750364342563,
            1.017587588351623,
            1.014318549696929,
            1.012734853725099,
            1.014142473224681,
            1.014806081078364,
            1.018397188773165,
            1.017293013444415,
            1.016119001549195,
            1.015874810864433,
            1.015959308494161,
            1.015382742567051,
            1.014775156326193,
            1.0154084139164,
            1.01526167995389,
            1.015515175047598,
            1.014557218493085,
            1.017947832498726,
            1.019048526664507,
            1.020032712747005,
            1.018157449129869,
            1.018498829617651,
            1.014870280545071,
            1.014562094755349,
            1.018564873776669,
            1.017080948122566,
            1.013376344366666,
            1.013929913259936,
            1.013065728469486,
            1.018643652588794,
            1.016056236032163,
            1.013523927163479,
            1.01185090278505,
            1.011455288339797,
            1.00874229563671,
            1.012805800711458,
            1.018779238323918,
            1.021467736727202,
            1.037761802244086,
            1.051614580731522,
            1.07128736721726,
            1.079295388423541,
            1.084724413995072,
            1.09223327977425,
            1.094861624281329,
            1.091300508083729,
            1.085904096344181,
            1.070284336042781,
            1.078779498225127,
            1.088793822453065,
            1.095974962270686,
            1.092331571340362,
            1.111117876462722,
            1.143236539257956,
            1.167634033519912,
            1.183305828521169,
            1.176713452769936,
            1.15210345819722,
            1.125736717928005,
            1.101929990027272,
            1.090154013335487,
            1.096782462284,
            1.113231870410792,
            1.119885776564696,
            1.152574290743437,
            1.144568297776994,
            1.175474143265696,
            1.206637794245618,
            1.223368285548087,
            1.186268116204504,
            1.141506770360242,
            1.129871944980037,
            1.125673975764433,
            1.128857952028103,
            1.132680882151429,
            1.118211137656054,
            1.120723683613631,
            1.126517970524139,
            1.131883289507265,
            1.132891353961099,
            1.140732816077296,
            1.116901724691276,
            1.094584845628911,
            1.075427197285505,
            1.064815070488037,
            1.057117704854909,
            1.056119035829482,
            1.055259470434932,
            1.048830818140458,
            1.042888519392499,
            1.038157545380589,
            1.02682206650756,
            1.024510313792351,
            1.017467315383733,
            0.978791467114764,
            0.928515361608108,
            0.898803479574629,
            0.883726709672177,
            0.853898348293814,
            0.81458602847683,
            0.759012275456045,
            0.753089430470701,
            0.68159179160439,
            0.583489672919228,
            0.471627484951014,
            0.403828679704929,
            0.270864932371982,
            0.152290118305549,
            0.128334627035245,
            0.096955901740026,
            0.116814240709015,
            0.029124748378448,
            -0.074970355177952,
            -0.112188250245635,
            -0.174182705922768,
            -0.228954883262758,
            -0.18949446850132,
            -0.117319577796606,
            -0.103772612304803,
            -0.166271137222846,
            -0.364839323444815,
            -0.529182060240115,
            -0.82009381095375,
            -1.133718049566681,
            -1.76816575468236,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(&exemplar, &calc.k(), margin, "k(3)");

        // test_3.s2p
        let exemplar: Array1<f64> = array![
            0.99884491752086,
            0.999824832484492,
            0.99990721490354,
            1.000037345563404,
            1.001266481313192,
            1.000976241408925,
            1.000995998349693,
            1.000931412008862,
            1.001035280047049,
            1.001080683586668,
            1.000968107864624,
            1.000978565201242,
            1.000931213712055,
            1.000963273080367,
            1.000973566330174,
            1.000882598950094,
            1.000947423529706,
            1.001055724489766,
            1.001076983866143,
            1.001190607773704,
            1.001286978748558,
            1.001362863263288,
            1.001419358943348,
            1.001394031786312,
            1.001359877506111,
            1.001403474122406,
            1.001535984744359,
            1.001602997226476,
            1.001659344221649,
            1.001667346638896,
            1.001698570305156,
            1.001782820099967,
            1.001877024994101,
            1.002009952948106,
            1.001978268086541,
            1.002064710691319,
            1.002065074109739,
            1.00205035778464,
            1.002098616531154,
            1.002171232715008,
            1.002136062527624,
            1.002081102984766,
            1.002091260159962,
            1.002162798594731,
            1.002142051527743,
            1.002200717929988,
            1.002304579218881,
            1.002790501482946,
            1.002941391876036,
            1.003189834755135,
            1.003139093376653,
            1.002927861594078,
            1.002853811991363,
            1.00330543951558,
            1.003852418180988,
            1.003868338418934,
            1.003419161118503,
            1.003443721387858,
            1.004028390101594,
            1.004189770849714,
            1.004413891541081,
            1.004230574096484,
            1.004402222571255,
            1.004611559410009,
            1.004955713471349,
            1.004837609104426,
            1.004902762847915,
            1.004691912678147,
            1.004962101154029,
            1.00465836012255,
            1.004205977513277,
            1.004096701413563,
            1.004567793140824,
            1.004487435935498,
            1.00384932220777,
            1.003671286748764,
            1.003720692656756,
            1.004559801703881,
            1.00465940256058,
            1.003831639424202,
            1.003177869257541,
            1.002504542754053,
            1.002671271755515,
            1.003231289890329,
            1.003459575111933,
            1.003361830092294,
            1.003292472246112,
            1.003830248692236,
            1.004044045622575,
            1.004872125352639,
            1.004724540303271,
            1.005692969374125,
            1.005439376271964,
            1.00562807842923,
            1.005213053268364,
            1.00549514368793,
            1.005412723649491,
            1.006935948189589,
            1.006824649212832,
            1.007546178961084,
            1.005474315887343,
            1.006491722530439,
            1.007815194031049,
            1.008525139039717,
            1.009431762408483,
            1.009197182665223,
            1.008824060948217,
            1.009146019211221,
            1.008760256293729,
            1.008475521293789,
            1.007738525742015,
            1.006639236651127,
            1.006372904606021,
            1.006786585371359,
            1.006376274644962,
            1.007675461188843,
            1.009223864400035,
            1.011066345488767,
            1.010166118979825,
            1.008149810169069,
            1.006311293723631,
            1.005174260887901,
            1.00442302257688,
            1.004229480469601,
            1.003940094050078,
            1.004142970163405,
            1.005870088864707,
            1.013189964928452,
            1.016459023195808,
            1.012953840529124,
            1.009113797689327,
            1.012285653640302,
            1.019710479844714,
            1.022630137896108,
            1.012423207061966,
            1.014987468746907,
            1.020202346001612,
            1.025468354036471,
            1.027856206926141,
            1.030035780126153,
            1.0329936721796,
            1.034653627958645,
            1.038912730879084,
            1.035771671813172,
            1.032560060684169,
            1.031247168173466,
            1.028440573410744,
            1.027983108682901,
            1.027214070134816,
            1.027307112110698,
            1.02605292416889,
            1.025361794505033,
            1.024349407951372,
            1.023056572792869,
            1.022656935768375,
            1.024389503732558,
            1.026353174904809,
            1.026131948612361,
            1.028185833726614,
            1.025642633266696,
            1.024461097686035,
            1.025324756576132,
            1.025241639697689,
            1.022298645022899,
            1.018026676604705,
            1.015154631832953,
            1.013188183786219,
            1.011763705486615,
            1.008834942941232,
            1.004761553324948,
            1.003937790308617,
            1.002437541581766,
            0.999441344889291,
            1.001709877986423,
            1.004797202569232,
            1.005908872756473,
            1.003951940213262,
            1.000367721958842,
            0.994451276174376,
            0.98752889640026,
            0.987365963827543,
            0.985775515033658,
            0.985336657246883,
            0.987263481043676,
            0.990841498272702,
            0.989538236949519,
            0.988642174363725,
            0.991761596076345,
            0.996380609305207,
            0.996524599088661,
            0.987461859137092,
            0.985964859941659,
            0.98205403594659,
            0.981773710653043,
            0.984483874409085,
            0.983160637179663,
            0.983785296236299,
            0.982921894121606,
            0.984438909414502,
            0.985240523225107,
            0.98649733921098,
            0.985484449045231,
            0.982722118116409,
            0.983021312677092,
            0.983198266761087,
            0.984135970602136,
            0.985674362933318,
            0.981324658284297,
            0.9747921183938,
            0.957743494095989,
            0.911490006401692,
            0.812943594638269,
            0.649380734986914,
            0.510963794492618,
            0.502321223000366,
            0.605932485745255,
            0.713672627591829,
            0.79979328604525,
            0.85811807687309,
            0.909882228843024,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::Db,
            vec![
                vec![
                    (-0.007, -2.703),
                    (-33.308, 87.205),
                    (-33.310, 87.223),
                    (0.002, -2.522),
                ],
                vec![
                    (-0.014, -5.380),
                    (-27.320, 84.747),
                    (-27.319, 84.763),
                    (-0.006, -5.048),
                ],
                vec![
                    (-0.020, -8.054),
                    (-23.824, 82.282),
                    (-23.822, 82.309),
                    (-0.011, -7.535),
                ],
                vec![
                    (-0.030, -10.721),
                    (-21.356, 79.850),
                    (-21.355, 79.863),
                    (-0.022, -10.008),
                ],
                vec![
                    (-0.070, -12.934),
                    (-19.767, 76.759),
                    (-19.766, 76.774),
                    (-0.073, -12.097),
                ],
                vec![
                    (-0.102, -15.528),
                    (-18.235, 74.354),
                    (-18.233, 74.341),
                    (-0.086, -14.420),
                ],
                vec![
                    (-0.122, -18.012),
                    (-16.968, 71.927),
                    (-16.964, 71.952),
                    (-0.118, -16.765),
                ],
                vec![
                    (-0.156, -20.492),
                    (-15.876, 69.541),
                    (-15.872, 69.531),
                    (-0.143, -19.112),
                ],
                vec![
                    (-0.185, -23.013),
                    (-14.932, 67.149),
                    (-14.928, 67.141),
                    (-0.181, -21.370),
                ],
                vec![
                    (-0.224, -25.457),
                    (-14.103, 64.781),
                    (-14.100, 64.755),
                    (-0.214, -23.659),
                ],
                vec![
                    (-0.265, -27.878),
                    (-13.365, 62.489),
                    (-13.360, 62.451),
                    (-0.244, -25.890),
                ],
                vec![
                    (-0.302, -30.240),
                    (-12.705, 60.203),
                    (-12.703, 60.163),
                    (-0.284, -28.109),
                ],
                vec![
                    (-0.344, -32.603),
                    (-12.109, 57.908),
                    (-12.107, 57.880),
                    (-0.322, -30.298),
                ],
                vec![
                    (-0.387, -34.903),
                    (-11.574, 55.657),
                    (-11.572, 55.620),
                    (-0.364, -32.496),
                ],
                vec![
                    (-0.429, -37.197),
                    (-11.087, 53.446),
                    (-11.086, 53.458),
                    (-0.407, -34.613),
                ],
                vec![
                    (-0.463, -39.356),
                    (-10.675, 51.542),
                    (-10.671, 51.469),
                    (-0.446, -36.438),
                ],
                vec![
                    (-0.515, -41.554),
                    (-10.268, 49.344),
                    (-10.270, 49.333),
                    (-0.488, -38.514),
                ],
                vec![
                    (-0.562, -43.746),
                    (-9.897, 47.250),
                    (-9.897, 47.193),
                    (-0.536, -40.558),
                ],
                vec![
                    (-0.610, -45.867),
                    (-9.555, 45.112),
                    (-9.555, 45.063),
                    (-0.582, -42.571),
                ],
                vec![
                    (-0.660, -47.981),
                    (-9.247, 43.031),
                    (-9.245, 42.983),
                    (-0.630, -44.513),
                ],
                vec![
                    (-0.704, -50.043),
                    (-8.963, 41.024),
                    (-8.964, 40.994),
                    (-0.679, -46.423),
                ],
                vec![
                    (-0.757, -52.064),
                    (-8.703, 39.062),
                    (-8.702, 39.007),
                    (-0.720, -48.330),
                ],
                vec![
                    (-0.799, -54.087),
                    (-8.460, 37.129),
                    (-8.460, 37.075),
                    (-0.767, -50.166),
                ],
                vec![
                    (-0.851, -56.013),
                    (-8.237, 35.258),
                    (-8.238, 35.248),
                    (-0.800, -51.986),
                ],
                vec![
                    (-0.898, -57.881),
                    (-8.025, 33.392),
                    (-8.025, 33.331),
                    (-0.839, -53.849),
                ],
                vec![
                    (-0.939, -59.713),
                    (-7.832, 31.566),
                    (-7.832, 31.496),
                    (-0.882, -55.676),
                ],
                vec![
                    (-0.982, -61.565),
                    (-7.656, 29.737),
                    (-7.658, 29.701),
                    (-0.927, -57.406),
                ],
                vec![
                    (-1.021, -63.357),
                    (-7.495, 27.999),
                    (-7.495, 27.920),
                    (-0.968, -59.135),
                ],
                vec![
                    (-1.056, -65.122),
                    (-7.345, 26.233),
                    (-7.349, 26.172),
                    (-1.010, -60.814),
                ],
                vec![
                    (-1.093, -66.877),
                    (-7.211, 24.513),
                    (-7.210, 24.431),
                    (-1.045, -62.470),
                ],
                vec![
                    (-1.130, -68.611),
                    (-7.084, 22.839),
                    (-7.085, 22.760),
                    (-1.077, -64.114),
                ],
                vec![
                    (-1.166, -70.296),
                    (-6.964, 21.187),
                    (-6.963, 21.105),
                    (-1.113, -65.762),
                ],
                vec![
                    (-1.199, -71.933),
                    (-6.856, 19.554),
                    (-6.855, 19.467),
                    (-1.149, -67.361),
                ],
                vec![
                    (-1.228, -73.544),
                    (-6.761, 17.969),
                    (-6.761, 17.907),
                    (-1.184, -68.913),
                ],
                vec![
                    (-1.258, -75.146),
                    (-6.668, 16.427),
                    (-6.669, 16.307),
                    (-1.209, -70.427),
                ],
                vec![
                    (-1.293, -76.757),
                    (-6.583, 14.877),
                    (-6.587, 14.748),
                    (-1.235, -71.896),
                ],
                vec![
                    (-1.316, -78.293),
                    (-6.503, 13.359),
                    (-6.507, 13.277),
                    (-1.261, -73.391),
                ],
                vec![
                    (-1.339, -79.774),
                    (-6.431, 11.869),
                    (-6.434, 11.757),
                    (-1.286, -74.841),
                ],
                vec![
                    (-1.362, -81.225),
                    (-6.369, 10.428),
                    (-6.372, 10.325),
                    (-1.308, -76.268),
                ],
                vec![
                    (-1.387, -82.666),
                    (-6.312, 8.931),
                    (-6.317, 8.864),
                    (-1.330, -77.697),
                ],
                vec![
                    (-1.405, -84.120),
                    (-6.257, 7.509),
                    (-6.262, 7.430),
                    (-1.347, -79.106),
                ],
                vec![
                    (-1.419, -85.518),
                    (-6.208, 6.101),
                    (-6.213, 5.976),
                    (-1.365, -80.482),
                ],
                vec![
                    (-1.429, -86.867),
                    (-6.165, 4.717),
                    (-6.169, 4.653),
                    (-1.384, -81.874),
                ],
                vec![
                    (-1.447, -88.250),
                    (-6.127, 3.340),
                    (-6.132, 3.226),
                    (-1.400, -83.269),
                ],
                vec![
                    (-1.465, -89.627),
                    (-6.089, 1.959),
                    (-6.092, 1.849),
                    (-1.410, -84.648),
                ],
                vec![
                    (-1.477, -90.985),
                    (-6.063, 0.594),
                    (-6.066, 0.471),
                    (-1.424, -85.964),
                ],
                vec![
                    (-1.481, -92.320),
                    (-6.052, -0.794),
                    (-6.055, -0.900),
                    (-1.441, -87.215),
                ],
                vec![
                    (-1.488, -93.626),
                    (-6.056, -2.034),
                    (-6.060, -2.165),
                    (-1.466, -88.427),
                ],
                vec![
                    (-1.504, -94.858),
                    (-6.038, -3.236),
                    (-6.039, -3.361),
                    (-1.474, -89.665),
                ],
                vec![
                    (-1.520, -96.087),
                    (-6.023, -4.482),
                    (-6.024, -4.590),
                    (-1.486, -90.966),
                ],
                vec![
                    (-1.522, -97.307),
                    (-6.009, -5.661),
                    (-6.008, -5.786),
                    (-1.490, -92.174),
                ],
                vec![
                    (-1.518, -98.492),
                    (-6.002, -6.874),
                    (-6.003, -6.991),
                    (-1.485, -93.349),
                ],
                vec![
                    (-1.517, -99.685),
                    (-5.991, -8.055),
                    (-5.995, -8.196),
                    (-1.487, -94.556),
                ],
                vec![
                    (-1.539, -100.827),
                    (-5.988, -9.359),
                    (-5.989, -9.478),
                    (-1.508, -95.800),
                ],
                vec![
                    (-1.549, -101.915),
                    (-6.011, -10.524),
                    (-6.016, -10.644),
                    (-1.520, -96.906),
                ],
                vec![
                    (-1.535, -103.121),
                    (-6.030, -11.560),
                    (-6.035, -11.671),
                    (-1.513, -97.983),
                ],
                vec![
                    (-1.527, -104.305),
                    (-6.029, -12.498),
                    (-6.032, -12.599),
                    (-1.483, -99.050),
                ],
                vec![
                    (-1.536, -105.323),
                    (-6.025, -13.680),
                    (-6.030, -13.823),
                    (-1.482, -100.338),
                ],
                vec![
                    (-1.543, -106.386),
                    (-6.035, -14.827),
                    (-6.036, -14.953),
                    (-1.508, -101.481),
                ],
                vec![
                    (-1.532, -107.502),
                    (-6.043, -15.925),
                    (-6.049, -16.028),
                    (-1.519, -102.578),
                ],
                vec![
                    (-1.523, -108.630),
                    (-6.073, -17.002),
                    (-6.074, -17.148),
                    (-1.520, -103.680),
                ],
                vec![
                    (-1.517, -109.630),
                    (-6.085, -18.045),
                    (-6.088, -18.163),
                    (-1.507, -104.645),
                ],
                vec![
                    (-1.517, -110.526),
                    (-6.108, -19.017),
                    (-6.115, -19.129),
                    (-1.502, -105.568),
                ],
                vec![
                    (-1.523, -111.551),
                    (-6.133, -19.965),
                    (-6.132, -20.109),
                    (-1.491, -106.576),
                ],
                vec![
                    (-1.520, -112.602),
                    (-6.146, -20.916),
                    (-6.148, -21.090),
                    (-1.497, -107.712),
                ],
                vec![
                    (-1.505, -113.579),
                    (-6.160, -21.929),
                    (-6.164, -22.133),
                    (-1.493, -108.803),
                ],
                vec![
                    (-1.496, -114.428),
                    (-6.187, -22.916),
                    (-6.190, -23.068),
                    (-1.490, -109.691),
                ],
                vec![
                    (-1.492, -115.261),
                    (-6.202, -23.910),
                    (-6.201, -24.079),
                    (-1.478, -110.630),
                ],
                vec![
                    (-1.479, -116.159),
                    (-6.233, -24.888),
                    (-6.235, -25.011),
                    (-1.486, -111.506),
                ],
                vec![
                    (-1.450, -117.227),
                    (-6.269, -25.884),
                    (-6.268, -25.983),
                    (-1.472, -112.436),
                ],
                vec![
                    (-1.415, -118.310),
                    (-6.304, -26.742),
                    (-6.300, -26.933),
                    (-1.449, -113.459),
                ],
                vec![
                    (-1.406, -119.173),
                    (-6.318, -27.627),
                    (-6.319, -27.787),
                    (-1.436, -114.468),
                ],
                vec![
                    (-1.416, -120.155),
                    (-6.333, -28.537),
                    (-6.336, -28.705),
                    (-1.442, -115.341),
                ],
                vec![
                    (-1.412, -121.119),
                    (-6.354, -29.574),
                    (-6.357, -29.733),
                    (-1.429, -116.311),
                ],
                vec![
                    (-1.375, -122.107),
                    (-6.378, -30.598),
                    (-6.386, -30.749),
                    (-1.411, -117.337),
                ],
                vec![
                    (-1.348, -122.983),
                    (-6.417, -31.525),
                    (-6.420, -31.689),
                    (-1.405, -118.305),
                ],
                vec![
                    (-1.340, -123.747),
                    (-6.459, -32.495),
                    (-6.472, -32.682),
                    (-1.397, -119.061),
                ],
                vec![
                    (-1.351, -124.603),
                    (-6.527, -33.267),
                    (-6.525, -33.467),
                    (-1.388, -119.865),
                ],
                vec![
                    (-1.339, -125.547),
                    (-6.566, -34.045),
                    (-6.567, -34.205),
                    (-1.374, -120.594),
                ],
                vec![
                    (-1.313, -126.503),
                    (-6.593, -34.907),
                    (-6.594, -35.053),
                    (-1.332, -121.429),
                ],
                vec![
                    (-1.287, -127.265),
                    (-6.622, -35.784),
                    (-6.631, -36.002),
                    (-1.303, -122.280),
                ],
                vec![
                    (-1.265, -127.923),
                    (-6.640, -36.573),
                    (-6.655, -36.827),
                    (-1.275, -123.026),
                ],
                vec![
                    (-1.276, -128.868),
                    (-6.676, -37.454),
                    (-6.683, -37.685),
                    (-1.248, -124.052),
                ],
                vec![
                    (-1.276, -129.931),
                    (-6.720, -38.438),
                    (-6.734, -38.649),
                    (-1.253, -124.964),
                ],
                vec![
                    (-1.258, -130.950),
                    (-6.775, -39.360),
                    (-6.782, -39.607),
                    (-1.251, -125.870),
                ],
                vec![
                    (-1.250, -131.674),
                    (-6.823, -40.202),
                    (-6.838, -40.386),
                    (-1.227, -126.610),
                ],
                vec![
                    (-1.245, -132.330),
                    (-6.872, -40.937),
                    (-6.868, -41.097),
                    (-1.203, -127.499),
                ],
                vec![
                    (-1.264, -133.188),
                    (-6.919, -41.663),
                    (-6.921, -41.828),
                    (-1.186, -128.275),
                ],
                vec![
                    (-1.241, -133.962),
                    (-6.979, -42.437),
                    (-6.979, -42.740),
                    (-1.181, -129.251),
                ],
                vec![
                    (-1.244, -134.873),
                    (-7.035, -43.207),
                    (-7.046, -43.432),
                    (-1.178, -130.173),
                ],
                vec![
                    (-1.233, -135.444),
                    (-7.079, -43.963),
                    (-7.079, -44.149),
                    (-1.163, -130.927),
                ],
                vec![
                    (-1.204, -137.506),
                    (-7.295, -45.675),
                    (-7.287, -45.895),
                    (-1.120, -132.380),
                ],
                vec![
                    (-1.196, -138.266),
                    (-7.331, -46.540),
                    (-7.340, -46.723),
                    (-1.101, -132.999),
                ],
                vec![
                    (-1.184, -139.237),
                    (-7.406, -47.224),
                    (-7.392, -47.498),
                    (-1.081, -133.941),
                ],
                vec![
                    (-1.156, -139.806),
                    (-7.439, -47.942),
                    (-7.442, -48.261),
                    (-1.066, -134.851),
                ],
                vec![
                    (-1.141, -140.499),
                    (-7.512, -48.823),
                    (-7.507, -49.026),
                    (-1.057, -135.763),
                ],
                vec![
                    (-1.151, -141.070),
                    (-7.544, -49.468),
                    (-7.552, -49.677),
                    (-1.033, -136.450),
                ],
                vec![
                    (-1.139, -141.988),
                    (-7.619, -50.241),
                    (-7.621, -50.511),
                    (-1.060, -137.163),
                ],
                vec![
                    (-1.126, -142.624),
                    (-7.678, -51.043),
                    (-7.661, -51.286),
                    (-1.045, -138.047),
                ],
                vec![
                    (-1.090, -143.317),
                    (-7.754, -51.827),
                    (-7.754, -51.818),
                    (-1.059, -138.695),
                ],
                vec![
                    (-1.090, -143.804),
                    (-7.812, -52.472),
                    (-7.801, -52.722),
                    (-0.973, -139.574),
                ],
                vec![
                    (-1.082, -144.444),
                    (-7.886, -53.018),
                    (-7.910, -53.153),
                    (-0.968, -140.498),
                ],
                vec![
                    (-1.091, -145.084),
                    (-7.937, -53.366),
                    (-7.948, -53.552),
                    (-0.981, -140.970),
                ],
                vec![
                    (-1.073, -145.770),
                    (-7.979, -53.983),
                    (-7.999, -54.149),
                    (-0.994, -141.596),
                ],
                vec![
                    (-1.063, -146.276),
                    (-8.043, -54.559),
                    (-8.026, -54.782),
                    (-1.008, -142.399),
                ],
                vec![
                    (-1.042, -146.836),
                    (-8.078, -55.281),
                    (-8.074, -55.531),
                    (-1.001, -143.284),
                ],
                vec![
                    (-1.033, -147.673),
                    (-8.132, -56.080),
                    (-8.110, -56.310),
                    (-0.981, -143.668),
                ],
                vec![
                    (-1.042, -148.451),
                    (-8.160, -56.728),
                    (-8.174, -56.898),
                    (-0.965, -144.102),
                ],
                vec![
                    (-1.024, -148.900),
                    (-8.249, -57.285),
                    (-8.226, -57.521),
                    (-0.941, -144.796),
                ],
                vec![
                    (-0.995, -149.415),
                    (-8.286, -57.910),
                    (-8.272, -58.101),
                    (-0.938, -145.596),
                ],
                vec![
                    (-0.989, -150.029),
                    (-8.310, -58.664),
                    (-8.327, -58.674),
                    (-0.907, -146.361),
                ],
                vec![
                    (-0.991, -150.499),
                    (-8.371, -59.225),
                    (-8.362, -59.245),
                    (-0.863, -146.713),
                ],
                vec![
                    (-1.005, -150.999),
                    (-8.404, -59.719),
                    (-8.422, -59.990),
                    (-0.836, -147.464),
                ],
                vec![
                    (-0.991, -151.557),
                    (-8.484, -60.340),
                    (-8.449, -60.444),
                    (-0.834, -148.351),
                ],
                vec![
                    (-0.992, -152.156),
                    (-8.521, -61.244),
                    (-8.512, -61.261),
                    (-0.813, -149.016),
                ],
                vec![
                    (-0.994, -152.469),
                    (-8.563, -61.662),
                    (-8.583, -62.009),
                    (-0.825, -149.822),
                ],
                vec![
                    (-0.996, -152.775),
                    (-8.620, -62.219),
                    (-8.625, -62.491),
                    (-0.844, -150.308),
                ],
                vec![
                    (-0.998, -153.136),
                    (-8.656, -62.852),
                    (-8.703, -63.102),
                    (-0.866, -150.828),
                ],
                vec![
                    (-0.956, -153.739),
                    (-8.707, -63.513),
                    (-8.698, -63.648),
                    (-0.869, -151.174),
                ],
                vec![
                    (-0.964, -154.217),
                    (-8.762, -64.131),
                    (-8.749, -64.117),
                    (-0.803, -151.983),
                ],
                vec![
                    (-0.949, -154.618),
                    (-8.809, -64.854),
                    (-8.772, -65.287),
                    (-0.774, -152.579),
                ],
                vec![
                    (-0.920, -155.032),
                    (-8.834, -65.647),
                    (-8.851, -65.891),
                    (-0.753, -153.294),
                ],
                vec![
                    (-0.903, -155.591),
                    (-8.860, -66.464),
                    (-8.894, -66.700),
                    (-0.741, -154.031),
                ],
                vec![
                    (-0.907, -156.265),
                    (-8.938, -67.399),
                    (-8.959, -67.342),
                    (-0.720, -154.847),
                ],
                vec![
                    (-0.896, -156.802),
                    (-9.031, -67.658),
                    (-9.033, -68.226),
                    (-0.692, -155.857),
                ],
                vec![
                    (-0.856, -157.201),
                    (-9.084, -68.597),
                    (-9.143, -68.901),
                    (-0.706, -156.318),
                ],
                vec![
                    (-0.874, -157.546),
                    (-9.202, -69.240),
                    (-9.200, -69.721),
                    (-0.727, -156.732),
                ],
                vec![
                    (-0.922, -158.150),
                    (-9.258, -69.906),
                    (-9.279, -70.164),
                    (-0.807, -157.422),
                ],
                vec![
                    (-0.947, -158.704),
                    (-9.308, -70.926),
                    (-9.333, -70.844),
                    (-0.835, -158.104),
                ],
                vec![
                    (-0.895, -159.033),
                    (-9.413, -71.526),
                    (-9.401, -71.758),
                    (-0.804, -158.610),
                ],
                vec![
                    (-0.854, -159.318),
                    (-9.522, -72.082),
                    (-9.547, -72.352),
                    (-0.752, -158.691),
                ],
                vec![
                    (-0.838, -159.873),
                    (-9.683, -72.618),
                    (-9.692, -73.056),
                    (-0.775, -159.259),
                ],
                vec![
                    (-0.849, -160.766),
                    (-9.746, -73.258),
                    (-9.704, -73.203),
                    (-0.850, -160.019),
                ],
                vec![
                    (-0.846, -161.398),
                    (-9.808, -73.008),
                    (-9.869, -73.463),
                    (-0.825, -161.307),
                ],
                vec![
                    (-0.756, -161.361),
                    (-9.948, -73.728),
                    (-9.931, -73.058),
                    (-0.713, -161.524),
                ],
                vec![
                    (-0.761, -162.046),
                    (-10.012, -74.283),
                    (-9.980, -73.262),
                    (-0.727, -162.222),
                ],
                vec![
                    (-0.760, -162.772),
                    (-10.097, -74.685),
                    (-10.090, -73.887),
                    (-0.778, -163.039),
                ],
                vec![
                    (-0.759, -163.211),
                    (-10.197, -74.987),
                    (-10.190, -74.236),
                    (-0.827, -163.492),
                ],
                vec![
                    (-0.743, -163.534),
                    (-10.288, -75.174),
                    (-10.306, -74.376),
                    (-0.845, -163.829),
                ],
                vec![
                    (-0.718, -164.022),
                    (-10.406, -75.414),
                    (-10.419, -74.530),
                    (-0.869, -164.190),
                ],
                vec![
                    (-0.713, -164.578),
                    (-10.501, -75.436),
                    (-10.514, -74.649),
                    (-0.882, -164.538),
                ],
                vec![
                    (-0.721, -164.938),
                    (-10.563, -75.321),
                    (-10.556, -74.375),
                    (-0.872, -164.822),
                ],
                vec![
                    (-0.740, -165.270),
                    (-10.616, -75.215),
                    (-10.615, -74.492),
                    (-0.882, -164.933),
                ],
                vec![
                    (-0.727, -165.532),
                    (-10.636, -75.164),
                    (-10.632, -74.424),
                    (-0.853, -165.150),
                ],
                vec![
                    (-0.704, -166.050),
                    (-10.623, -75.065),
                    (-10.634, -74.288),
                    (-0.849, -165.342),
                ],
                vec![
                    (-0.701, -166.646),
                    (-10.619, -75.230),
                    (-10.623, -74.467),
                    (-0.840, -165.459),
                ],
                vec![
                    (-0.696, -167.209),
                    (-10.617, -75.602),
                    (-10.618, -74.787),
                    (-0.809, -165.838),
                ],
                vec![
                    (-0.716, -167.810),
                    (-10.631, -75.931),
                    (-10.637, -75.128),
                    (-0.774, -166.238),
                ],
                vec![
                    (-0.741, -168.274),
                    (-10.612, -76.177),
                    (-10.622, -75.507),
                    (-0.745, -166.740),
                ],
                vec![
                    (-0.764, -168.664),
                    (-10.584, -76.727),
                    (-10.596, -76.030),
                    (-0.731, -167.341),
                ],
                vec![
                    (-0.765, -168.885),
                    (-10.590, -77.366),
                    (-10.609, -76.537),
                    (-0.708, -167.767),
                ],
                vec![
                    (-0.772, -169.145),
                    (-10.629, -77.899),
                    (-10.640, -77.188),
                    (-0.683, -168.348),
                ],
                vec![
                    (-0.750, -169.444),
                    (-10.667, -78.526),
                    (-10.673, -77.889),
                    (-0.677, -169.053),
                ],
                vec![
                    (-0.736, -169.760),
                    (-10.715, -79.048),
                    (-10.721, -78.526),
                    (-0.659, -169.607),
                ],
                vec![
                    (-0.729, -169.988),
                    (-10.774, -79.624),
                    (-10.776, -79.055),
                    (-0.645, -170.279),
                ],
                vec![
                    (-0.726, -170.132),
                    (-10.837, -80.185),
                    (-10.837, -79.496),
                    (-0.652, -170.933),
                ],
                vec![
                    (-0.700, -170.491),
                    (-10.903, -80.593),
                    (-10.902, -79.850),
                    (-0.681, -171.309),
                ],
                vec![
                    (-0.666, -170.890),
                    (-10.965, -81.065),
                    (-10.973, -80.269),
                    (-0.694, -171.561),
                ],
                vec![
                    (-0.659, -171.335),
                    (-11.040, -81.499),
                    (-11.050, -80.629),
                    (-0.707, -171.762),
                ],
                vec![
                    (-0.626, -171.867),
                    (-11.113, -81.675),
                    (-11.129, -80.926),
                    (-0.693, -172.083),
                ],
                vec![
                    (-0.607, -172.055),
                    (-11.179, -81.751),
                    (-11.191, -80.954),
                    (-0.685, -172.412),
                ],
                vec![
                    (-0.604, -172.623),
                    (-11.229, -82.010),
                    (-11.237, -81.134),
                    (-0.692, -172.867),
                ],
                vec![
                    (-0.581, -173.185),
                    (-11.233, -82.302),
                    (-11.249, -81.467),
                    (-0.727, -173.284),
                ],
                vec![
                    (-0.555, -173.548),
                    (-11.270, -82.562),
                    (-11.284, -81.615),
                    (-0.719, -173.825),
                ],
                vec![
                    (-0.529, -174.035),
                    (-11.295, -82.750),
                    (-11.294, -81.725),
                    (-0.699, -174.420),
                ],
                vec![
                    (-0.493, -174.425),
                    (-11.329, -82.919),
                    (-11.331, -81.878),
                    (-0.714, -174.634),
                ],
                vec![
                    (-0.474, -174.909),
                    (-11.354, -83.028),
                    (-11.362, -82.069),
                    (-0.722, -174.945),
                ],
                vec![
                    (-0.463, -175.232),
                    (-11.350, -83.348),
                    (-11.367, -82.421),
                    (-0.717, -175.277),
                ],
                vec![
                    (-0.432, -175.548),
                    (-11.368, -83.709),
                    (-11.363, -82.693),
                    (-0.739, -175.601),
                ],
                vec![
                    (-0.393, -176.181),
                    (-11.396, -84.163),
                    (-11.398, -83.231),
                    (-0.752, -175.840),
                ],
                vec![
                    (-0.385, -176.735),
                    (-11.445, -84.475),
                    (-11.452, -83.539),
                    (-0.736, -176.095),
                ],
                vec![
                    (-0.378, -177.418),
                    (-11.479, -84.621),
                    (-11.486, -83.766),
                    (-0.728, -176.458),
                ],
                vec![
                    (-0.357, -178.081),
                    (-11.505, -84.919),
                    (-11.501, -84.040),
                    (-0.745, -176.797),
                ],
                vec![
                    (-0.373, -178.550),
                    (-11.532, -85.329),
                    (-11.527, -84.379),
                    (-0.737, -177.162),
                ],
                vec![
                    (-0.398, -179.219),
                    (-11.555, -85.876),
                    (-11.537, -84.887),
                    (-0.712, -177.517),
                ],
                vec![
                    (-0.406, -179.387),
                    (-11.592, -86.304),
                    (-11.596, -85.116),
                    (-0.697, -178.130),
                ],
                vec![
                    (-0.388, -179.751),
                    (-11.608, -86.455),
                    (-11.617, -85.489),
                    (-0.708, -178.407),
                ],
                vec![
                    (-0.351, 179.972),
                    (-11.662, -86.823),
                    (-11.666, -85.790),
                    (-0.737, -178.606),
                ],
                vec![
                    (-0.309, 179.552),
                    (-11.715, -87.019),
                    (-11.706, -85.940),
                    (-0.744, -178.758),
                ],
                vec![
                    (-0.265, 179.038),
                    (-11.732, -87.209),
                    (-11.729, -86.175),
                    (-0.744, -178.952),
                ],
                vec![
                    (-0.268, 178.546),
                    (-11.731, -87.406),
                    (-11.728, -86.540),
                    (-0.754, -179.276),
                ],
                vec![
                    (-0.257, 178.084),
                    (-11.753, -87.778),
                    (-11.749, -87.009),
                    (-0.785, -179.361),
                ],
                vec![
                    (-0.252, 177.567),
                    (-11.757, -88.069),
                    (-11.774, -87.224),
                    (-0.760, -179.478),
                ],
                vec![
                    (-0.264, 177.133),
                    (-11.802, -88.337),
                    (-11.792, -87.409),
                    (-0.743, -179.786),
                ],
                vec![
                    (-0.284, 176.662),
                    (-11.827, -88.625),
                    (-11.825, -87.942),
                    (-0.737, 179.938),
                ],
                vec![
                    (-0.281, 176.114),
                    (-11.838, -88.888),
                    (-11.841, -88.009),
                    (-0.737, 179.650),
                ],
                vec![
                    (-0.284, 175.439),
                    (-11.849, -89.096),
                    (-11.828, -88.048),
                    (-0.699, 179.298),
                ],
                vec![
                    (-0.315, 174.679),
                    (-11.814, -89.502),
                    (-11.828, -88.608),
                    (-0.682, 178.625),
                ],
                vec![
                    (-0.346, 174.031),
                    (-11.835, -89.972),
                    (-11.824, -88.968),
                    (-0.720, 178.373),
                ],
                vec![
                    (-0.338, 173.956),
                    (-11.827, -90.109),
                    (-11.831, -89.305),
                    (-0.713, 178.598),
                ],
                vec![
                    (-0.273, 173.583),
                    (-11.855, -90.206),
                    (-11.862, -89.369),
                    (-0.648, 178.595),
                ],
                vec![
                    (-0.266, 173.005),
                    (-11.846, -90.457),
                    (-11.845, -89.570),
                    (-0.609, 178.294),
                ],
                vec![
                    (-0.232, 172.456),
                    (-11.843, -90.902),
                    (-11.855, -90.128),
                    (-0.612, 177.855),
                ],
                vec![
                    (-0.223, 172.030),
                    (-11.847, -91.056),
                    (-11.850, -90.397),
                    (-0.575, 177.768),
                ],
                vec![
                    (-0.227, 171.491),
                    (-11.866, -91.933),
                    (-11.857, -91.056),
                    (-0.561, 177.593),
                ],
                vec![
                    (-0.211, 170.829),
                    (-11.908, -92.002),
                    (-11.879, -91.092),
                    (-0.489, 177.305),
                ],
                vec![
                    (-0.233, 170.119),
                    (-11.866, -92.554),
                    (-11.893, -91.468),
                    (-0.476, 176.579),
                ],
                vec![
                    (-0.204, 169.606),
                    (-11.915, -92.825),
                    (-11.903, -92.017),
                    (-0.448, 176.373),
                ],
                vec![
                    (-0.235, 168.777),
                    (-11.933, -93.385),
                    (-11.912, -92.605),
                    (-0.420, 175.689),
                ],
                vec![
                    (-0.271, 167.832),
                    (-11.947, -93.891),
                    (-11.939, -93.206),
                    (-0.472, 175.246),
                ],
                vec![
                    (-0.306, 167.102),
                    (-11.957, -94.149),
                    (-11.954, -93.210),
                    (-0.448, 175.324),
                ],
                vec![
                    (-0.258, 166.964),
                    (-11.928, -94.426),
                    (-11.944, -93.481),
                    (-0.419, 175.523),
                ],
                vec![
                    (-0.229, 165.980),
                    (-11.950, -94.223),
                    (-11.941, -93.532),
                    (-0.339, 175.625),
                ],
                vec![
                    (-0.213, 165.242),
                    (-11.921, -94.718),
                    (-11.931, -93.936),
                    (-0.281, 175.172),
                ],
                vec![
                    (-0.198, 164.306),
                    (-11.915, -95.109),
                    (-11.905, -94.295),
                    (-0.217, 174.767),
                ],
                vec![
                    (-0.188, 163.445),
                    (-11.865, -95.670),
                    (-11.879, -94.820),
                    (-0.205, 174.577),
                ],
                vec![
                    (-0.181, 162.394),
                    (-11.874, -96.178),
                    (-11.869, -95.263),
                    (-0.135, 174.270),
                ],
                vec![
                    (-0.234, 161.079),
                    (-11.820, -96.781),
                    (-11.834, -95.848),
                    (-0.079, 173.670),
                ],
                vec![
                    (-0.311, 159.862),
                    (-11.797, -97.531),
                    (-11.792, -96.706),
                    (-0.026, 173.252),
                ],
                vec![
                    (-0.473, 158.035),
                    (-11.767, -99.069),
                    (-11.777, -98.139),
                    (0.031, 172.884),
                ],
                vec![
                    (-0.865, 156.756),
                    (-11.916, -100.603),
                    (-11.916, -99.695),
                    (0.055, 172.980),
                ],
                vec![
                    (-1.383, 157.170),
                    (-12.183, -101.839),
                    (-12.185, -100.941),
                    (0.148, 172.980),
                ],
                vec![
                    (-1.786, 159.984),
                    (-12.561, -101.601),
                    (-12.577, -100.705),
                    (0.301, 173.255),
                ],
                vec![
                    (-1.716, 163.563),
                    (-12.816, -99.966),
                    (-12.805, -98.952),
                    (0.502, 172.921),
                ],
                vec![
                    (-1.393, 165.231),
                    (-12.859, -97.814),
                    (-12.853, -96.821),
                    (0.657, 171.744),
                ],
                vec![
                    (-1.030, 165.580),
                    (-12.700, -96.201),
                    (-12.669, -95.453),
                    (0.747, 170.716),
                ],
                vec![
                    (-0.759, 164.875),
                    (-12.514, -95.622),
                    (-12.508, -94.678),
                    (0.780, 169.512),
                ],
                vec![
                    (-0.585, 164.044),
                    (-12.361, -95.610),
                    (-12.335, -94.626),
                    (0.755, 168.895),
                ],
                vec![
                    (-0.439, 163.431),
                    (-12.203, -95.395),
                    (-12.209, -94.420),
                    (0.799, 168.418),
                ],
                vec![
                    (-0.318, 162.847),
                    (-12.067, -95.224),
                    (-12.068, -94.311),
                    (0.841, 167.973),
                ],
            ],
            String::from("test_3"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:48 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Vg(V) = -1.500\n\
            Vg(mA) = -0.000\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n",
            ),
        );
        comp_array_f64(&exemplar, &calc.k(), margin, "k(4)");
    }

    #[test]
    fn network_max_gain() {
        let margin = F64Margin {
            epsilon: 1e-12,
            ulps: 4,
        };

        let exemplar_max_gain: Array1<f64> = array![
            2.598182739029478e-02,
            2.269447098311342e-05,
            7.858595949885737e-03,
        ];
        let exemplar_max_stable_gain: Array1<f64> = array![
            0.025981827390294778,
            1.1572727293181975,
            0.007858595949885737,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [c64(0.958, -0.263), c64(-0.846, 0.158)],
                    [c64(0.004, 0.022), c64(0.544, -0.129)],
                ],
                [
                    [c64(2.043, -0.982), c64(-0.238, 3.249)],
                    [c64(-1.421, 3.492), c64(0.123, -394.321)],
                ],
                [
                    [c64(21.329, -0.421), c64(-0.942, 24.282)],
                    [c64(0.138, 0.132), c64(0.329, -0.324)],
                ],
            ]),
            String::from(""),
            String::from(""),
        );
        comp_array_f64(&exemplar_max_gain, &calc.max_gain(), margin, "max_gain(1)");
        comp_array_f64(
            &exemplar_max_stable_gain,
            &calc.max_stable_gain(),
            margin,
            "max_stable_gain(1)",
        );

        let exemplar_max_gain: Array1<f64> = array![
            1.0,
            0.994092797323051,
            0.980871982283681,
            0.975659691008919,
            0.970617076317979,
            0.945318674595296,
            0.947740487883644,
            0.948780732883345,
            0.949343800105823,
            0.948742806692139,
            0.949760919764674,
            0.949660000820972,
            0.949019718161874,
            0.949203992341624,
            0.949130602326096,
            0.948427590861599,
            0.946463914712278,
            0.944474100309854,
            0.943465075570112,
            0.94146714684935,
            0.940190895218058,
            0.938586038947001,
            0.938065184557721,
            0.937911960370146,
            0.935825525057975,
            0.934509110628749,
            0.93414173453021,
            0.933034788120097,
            0.932385053061897,
            0.930167097318895,
            0.929316331888112,
            0.928401668450045,
            0.926635714198457,
            0.926681986311755,
            0.924850625741893,
            0.923144998435958,
            0.923115213702005,
            0.922121837282966,
            0.921449956911788,
            0.919548791526155,
            0.918925454788444,
            0.918143349833795,
            0.918084361180047,
            0.918764456909024,
            0.916961511003236,
            0.91599822656396,
            0.915503885749939,
            0.912363410435533,
            0.912300433490201,
            0.911697389045941,
            0.910130892303694,
            0.908030988140174,
            0.906588862468409,
            0.906723231868661,
            0.90270832177315,
            0.900799290195015,
            0.901664445505674,
            0.899213462968392,
            0.900052363466225,
            0.896900852956737,
            0.894422324246187,
            0.892005395718747,
            0.891503341612457,
            0.889241586639432,
            0.888813004183055,
            0.885651601048039,
            0.882931533020833,
            0.884030027927766,
            0.886658098772947,
            0.885401323604052,
            0.881275968400642,
            0.881805189086383,
            0.88198496135351,
            0.882692597947165,
            0.876196330263242,
            0.869933470187943,
            0.870177752986957,
            0.868995890457553,
            0.869460260014081,
            0.870108091230503,
            0.869676772358321,
            0.870212274673186,
            0.867227697699452,
            0.872275185942727,
            0.871681604512292,
            0.873664060748587,
            0.86951931068901,
            0.864090097890183,
            0.857406544598409,
            0.854779604535115,
            0.836810906839736,
            0.840298639156737,
            0.846802364918049,
            0.849122541652532,
            0.837497637521582,
            0.83124765210588,
            0.827554200177953,
            0.848472307612517,
            0.853790188460385,
            0.843626972065088,
            0.844138746194853,
            0.829192158887129,
            0.83435570041604,
            0.838836909545909,
            0.837287082668848,
            0.836895192224119,
            0.841475727001511,
            0.840985978765968,
            0.843624722884158,
            0.840844316827825,
            0.842218980128504,
            0.847873311789357,
            0.82957486272439,
            0.824555885967823,
            0.821476085256111,
            0.829385838501851,
            0.825263569034793,
            0.84074213780203,
            0.847811995558976,
            0.830155192187746,
            0.833911529372934,
            0.852867914824187,
            0.844265741630912,
            0.85681322996008,
            0.824283869576165,
            0.843614362138789,
            0.850034653199883,
            0.861353231871547,
            0.874324586088362,
            0.875411853412295,
            0.857070770973191,
            0.831230336620208,
            0.817909763042788,
            0.760363968444156,
            0.728023154736361,
            0.687722655945199,
            0.676688468851562,
            0.666175582094376,
            0.653631253211627,
            0.650459478045049,
            0.657153221064535,
            0.666566062338588,
            0.691469128247877,
            0.675585161470064,
            0.658858059211845,
            0.648187693449005,
            0.653906770390068,
            0.627857193057362,
            0.590195188308254,
            0.565166503793858,
            0.551350886898908,
            0.556161901987619,
            0.579965032298388,
            0.609896746473801,
            0.639865221886674,
            0.656475516607114,
            0.646302973767871,
            0.624064008324258,
            0.615758657127631,
            0.579861250947202,
            0.587759005221522,
            0.55798626422155,
            0.531726291175208,
            0.519330493550792,
            0.548120763688345,
            0.58944421321507,
            0.603934104496039,
            0.608816258676543,
            0.605913149169632,
            0.601542854350711,
            0.617815146087329,
            0.615577170434142,
            0.605743080676773,
            0.601223060370802,
            0.59925116338547,
            0.592675779413556,
            0.619438698308061,
            0.649939045709335,
            0.68072863881235,
            0.699973657404935,
            0.714845105079887,
            0.717963728804057,
            0.717743717213561,
            0.732530193605536,
            0.747424678827708,
            0.765365973057491,
            0.794237663950127,
            0.801751738049684,
            0.832185238992792,
            1.001465201465201,
            0.998543335761107,
            1.0,
            1.0,
            0.997041420118343,
            1.001501501501502,
            0.993993993993994,
            1.001532567049809,
            1.004597701149425,
            0.998469778117827,
            1.000772200772201,
            0.99921875,
            1.002379064234735,
            0.998397435897436,
            0.999192897497982,
            1.003276003276003,
            1.000816993464052,
            1.004950495049505,
            1.00498753117207,
            1.002481389578164,
            0.999174917491749,
            1.003265306122449,
            0.996755879967559,
            1.000801282051282,
            0.998408910103421,
            0.997626582278481,
            0.996022275258552,
            0.995207667731629,
            0.997574777687955,
            1.000809061488673,
            1.000806451612903,
        ];
        let exemplar_max_stable_gain: Array1<f64> = array![
            1.0,
            0.998697916666667,
            1.0,
            1.0,
            0.9994617868676,
            1.0,
            1.000830220008302,
            1.00074294205052,
            1.001014198782961,
            1.000936329588015,
            1.000873871249636,
            1.00082349711776,
            1.000521648408972,
            1.00099825305715,
            1.000960384153661,
            1.000930448941614,
            1.000903546419697,
            1.000660501981506,
            1.000861326442722,
            1.000844951415294,
            1.001038853106171,
            1.00102396067991,
            1.000808897876643,
            1.001200960768615,
            1.000991866693117,
            1.000984445757038,
            1.001370131141124,
            1.001168907071888,
            1.001358695652174,
            1.001161665053243,
            1.001159644375725,
            1.001158077591199,
            1.001351090523065,
            1.001351090523065,
            1.001159420289855,
            1.00135527589545,
            1.001552192471867,
            1.00194666147557,
            1.001758155889822,
            1.001765744555621,
            1.001970443349754,
            1.001582591493571,
            1.001788908765653,
            1.001797842588893,
            1.00160610319213,
            1.001817080557238,
            1.001829268292683,
            1.002048340843916,
            1.001649484536083,
            1.001661129568106,
            1.001882845188284,
            1.001896733403583,
            1.00148588410104,
            1.001497005988024,
            1.001511879049676,
            1.001744059298016,
            1.001978891820581,
            1.001995123032587,
            1.00201072386059,
            1.001351655778328,
            1.001590547602818,
            1.001834862385321,
            1.001854857407837,
            1.001404165691552,
            1.001888574126534,
            1.001904761904762,
            1.001441614608362,
            1.001939393939394,
            1.00122279285889,
            1.002226620484908,
            1.00225056264066,
            1.002018672722685,
            1.001273236567354,
            1.002055498458376,
            1.002857142857143,
            1.002887897085849,
            1.002122578933404,
            1.001878690284487,
            1.002712232167073,
            1.00328587075575,
            1.003319502074689,
            1.001673173452315,
            1.002533070644526,
            1.001986379114643,
            1.001432664756447,
            1.001742160278746,
            1.001468428781204,
            1.002676181980375,
            1.00119940029985,
            1.001816530426885,
            1.001876759461996,
            1.002846299810247,
            1.0,
            1.002260251856635,
            0.999018645731109,
            1.003316749585406,
            0.997997997997998,
            1.004716981132076,
            1.001354554690146,
            0.997940974605353,
            1.002434782608696,
            1.004231311706629,
            1.004621400639886,
            1.003576537911302,
            1.000361141206212,
            1.00036469730124,
            1.00258207303578,
            0.998511904761905,
            1.005289006422365,
            1.001139817629179,
            1.004221028396009,
            1.005413766434648,
            1.002340093603744,
            1.001967729240457,
            1.003183446080382,
            1.003212851405622,
            1.0,
            0.998772001637331,
            1.005369681949608,
            1.006270903010033,
            1.002943650126156,
            1.004239084357779,
            0.997434801197093,
            1.006965607313888,
            0.999560246262093,
            1.008940545373268,
            1.0018009905448,
            1.00456204379562,
            1.017058552328262,
            0.999072786277237,
            1.005647058823529,
            1.008687258687259,
            1.005853658536585,
            1.0,
            1.002507522567703,
            1.001013684744045,
            1.005117707267144,
            1.002590673575129,
            1.001049868766404,
            1.002126528442318,
            1.004303388918774,
            1.005988023952096,
            1.003833515881708,
            1.002202643171806,
            1.001107419712071,
            1.001116071428571,
            1.001695873374788,
            1.001718213058419,
            1.001740139211137,
            1.000590667454223,
            1.001214329083181,
            0.999377722464219,
            1.0,
            1.001896333754741,
            1.001261829652997,
            1.000630119722747,
            1.0,
            1.0,
            1.0,
            1.000643500643501,
            1.0,
            1.000650618087183,
            1.000654022236756,
            1.001314924391847,
            1.0,
            0.997336884154461,
            1.0,
            1.0,
            1.001340482573727,
            1.001344086021505,
            1.0,
            1.001365187713311,
            0.996575342465753,
            0.999309868875086,
            0.997923875432526,
            1.001393728222997,
            1.0,
            1.000702740688686,
            1.001408450704225,
            1.001413427561838,
            1.000709723207949,
            1.002138275124733,
            0.999289267945984,
            1.0,
            1.000718907260963,
            1.008011653313911,
            1.000725163161711,
            1.0,
            1.002941176470588,
            1.001465201465201,
            0.998543335761107,
            1.0,
            1.0,
            0.997041420118343,
            1.001501501501502,
            0.993993993993994,
            1.001532567049809,
            1.004597701149425,
            0.998469778117827,
            1.000772200772201,
            0.99921875,
            1.002379064234735,
            0.998397435897436,
            0.999192897497982,
            1.003276003276003,
            1.000816993464052,
            1.004950495049505,
            1.00498753117207,
            1.002481389578164,
            0.999174917491749,
            1.003265306122449,
            0.996755879967559,
            1.000801282051282,
            0.998408910103421,
            0.997626582278481,
            0.996022275258552,
            0.995207667731629,
            0.997574777687955,
            1.000809061488673,
            1.000806451612903,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(&exemplar_max_gain, &calc.max_gain(), margin, "max_gain(2)");
        comp_array_f64(
            &exemplar_max_stable_gain,
            &calc.max_stable_gain(),
            margin,
            "max_stable_gain(2)",
        );

        let exemplar_max_gain: Array1<f64> = array![
            658.6428778530811,
            326.3623111913169,
            217.17008433479918,
            165.04409856522784,
        ];
        let exemplar_max_stable_gain: Array1<f64> = array![
            658.6428778530811,
            326.3623111913169,
            217.17008433479918,
            165.04409856522784,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![0.5, 1.0, 1.5, 2.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(0.9881388526914863, -0.13442709904013195),
                        c64(0.0010346705444205045, 0.011178864909012504)
                    ],
                    [
                        c64(-7.363542304219899, 0.6742816969789206),
                        c64(0.5574653418486702, -0.06665134724424635)
                    ],
                ],
                [
                    [
                        c64(0.9578079036840927, -0.2633207328693372),
                        c64(0.0037206104781559784, 0.021909191616475577)
                    ],
                    [
                        c64(-7.130124628011368, 1.3277987152036197),
                        c64(0.5435045929943587, -0.12869941397967788)
                    ],
                ],
                [
                    [
                        c64(0.9133108288727866, -0.38508398385543624),
                        c64(0.008042664765986755, 0.03190603796445517)
                    ],
                    [
                        c64(-6.9151682810378095, 1.800750901131042),
                        c64(0.5235871604669029, -0.18886435408156288)
                    ],
                ],
                [
                    [
                        c64(0.849070850314753, -0.49577931076259807),
                        c64(0.01381064392153511, 0.04080882571424955)
                    ],
                    [
                        c64(-6.688405272002992, 2.4133819411904995),
                        c64(0.4942211124266797, -0.24774648346309974)
                    ],
                ],
            ]),
            String::from("test"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:43 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 2.500\n\
            set_Vd(mA) = 11.250\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Vg(V) = -0.520\n\
            Vg(mA) = -0.001\n\
            Vd(V) = 2.500\n\
            Vd(mA) = 11.076",
            ),
        );
        comp_array_f64(&exemplar_max_gain, &calc.max_gain(), margin, "max_gain(3)");
        comp_array_f64(
            &exemplar_max_stable_gain,
            &calc.max_stable_gain(),
            margin,
            "max_stable_gain(3)",
        );

        let exemplar_max_gain: Array1<f64> = array![
            0.999769767998156,
            1.000115135882277,
            1.000230285020825,
            0.991509007159453,
            0.951031487204107,
            0.956998882455617,
            0.956793614504666,
            0.958202084857227,
            0.955960231481596,
            0.954907375002993,
            0.957506034139694,
            0.956948597344396,
            0.95798587665326,
            0.95728066235778,
            0.956946667715644,
            0.959300743853081,
            0.957186847636256,
            0.955093092561973,
            0.954653631697928,
            0.952597703454007,
            0.950427044963536,
            0.949245866769095,
            0.948120836763966,
            0.948464313636284,
            0.949190880208343,
            0.9484042478599,
            0.945871535063469,
            0.94495880905982,
            0.943592751843832,
            0.94400515384036,
            0.943280225139791,
            0.942151692793181,
            0.940686271878515,
            0.938575373994145,
            0.93893805859428,
            0.937339215212454,
            0.937333917178859,
            0.937656800682132,
            0.936954910437876,
            0.935699300294005,
            0.936200597050334,
            0.936992853546766,
            0.936953467184306,
            0.935819118532254,
            0.936330497059433,
            0.93549774382757,
            0.934052026540573,
            0.927605015865508,
            0.926079079609324,
            0.923147044129482,
            0.923948293944404,
            0.92624252915204,
            0.926824195525833,
            0.921824830752,
            0.915463588046133,
            0.915297895529712,
            0.920336428559824,
            0.919852022552048,
            0.914073227440744,
            0.911923949228758,
            0.910249185303243,
            0.911833848288283,
            0.909733552969048,
            0.908568429751667,
            0.905067873573164,
            0.905938737533909,
            0.905445827411739,
            0.907812715579725,
            0.905009895355364,
            0.908127402493316,
            0.912813147041322,
            0.913381462891693,
            0.908564585444315,
            0.90933128876457,
            0.915179706738006,
            0.917586879925492,
            0.916005032288692,
            0.909163706214779,
            0.90790852910066,
            0.916102192102914,
            0.922435382370343,
            0.930077773622852,
            0.928780959802298,
            0.921290061447211,
            0.919464933678558,
            0.919705368204672,
            0.922502765119077,
            0.91601126777567,
            0.914019351509532,
            0.904892290409419,
            0.907403412650437,
            0.899664536164083,
            0.900063284664436,
            0.900834673003954,
            0.902659936441928,
            0.901035179031777,
            0.900396938474388,
            0.888748217238333,
            0.891538485993479,
            0.884463675171631,
            0.901836884671557,
            0.889899742410515,
            0.881432642236487,
            0.875651839462107,
            0.873471629672916,
            0.873662111713589,
            0.877905873274339,
            0.872182139002867,
            0.878428590019707,
            0.879420089196214,
            0.881364584820884,
            0.892139797239225,
            0.89144655352311,
            0.893678990386667,
            0.894195270880045,
            0.881507020828142,
            0.87258616081288,
            0.857233600326794,
            0.868112062000056,
            0.881538380940771,
            0.897599320950864,
            0.901548833240845,
            0.906709489050314,
            0.909956995284551,
            0.914871591329131,
            0.906840799077453,
            0.897565743896254,
            0.848183129609345,
            0.831882887464413,
            0.852651961090752,
            0.871285548769643,
            0.854167048489007,
            0.824162953369795,
            0.803025880225043,
            0.855980636353501,
            0.844312095688127,
            0.818840140045152,
            0.798987928526602,
            0.788547537685319,
            0.781936046557058,
            0.77284589698191,
            0.769740045348144,
            0.757326945718559,
            0.766267181197991,
            0.774323539596971,
            0.778954400500438,
            0.788162373685402,
            0.789216692314151,
            0.791422107161761,
            0.790926799471769,
            0.794563947832488,
            0.797708448240125,
            0.801778004772158,
            0.806525292998122,
            0.808397869735603,
            0.802187218961593,
            0.795358521776261,
            0.795297450752768,
            0.788183530352134,
            0.796264851697481,
            0.800821845257851,
            0.798114480074445,
            0.797672665707624,
            0.808639518789818,
            0.827390928237794,
            0.840207241924895,
            0.849462824258241,
            0.856249302783762,
            0.876117729158887,
            0.906850216572628,
            0.914368809689005,
            0.931821947946032,
            1.000460623072841,
            0.943749315273574,
            0.908609752258492,
            0.896625942442433,
            0.914012820006469,
            0.972798066506588,
            1.001036700294489,
            1.000345447417171,
            1.000345447417171,
            1.00046062307284,
            0.998044716732711,
            1.001151955538169,
            1.000230285020825,
            0.999654671875538,
            1.002420639375797,
            0.998389488702328,
            1.001267224051863,
            0.999539589003088,
            0.999194419871492,
            1.000115135882277,
            0.998619402846525,
            0.999654671875538,
            1.001036700294489,
            1.003344328213383,
            0.996896336476927,
            1.001382505837099,
            1.002420639375796,
            1.000921458319296,
            1.00034544741717,
            0.998159627491724,
            1.001036700294489,
            0.998849369936505,
            1.001151955538169,
            0.998389488702328,
            1.000575811989361,
            0.998389488702328,
            1.000575811989361,
            0.998849369936505,
            1.0,
            0.999769767998156,
            0.998159627491724,
            1.001267224051863,
            1.000691014168259,
            1.0035753833829,
            1.000691014168259,
            1.002997845198331,
            0.99930946300259,
            0.999884877372469,
        ];
        let exemplar_max_stable_gain: Array1<f64> = array![
            0.999769767998156,
            1.000115135882277,
            1.000230285020825,
            1.000115135882277,
            1.000115135882277,
            1.000230285020824,
            1.000460623072841,
            1.00046062307284,
            1.00046062307284,
            1.00034544741717,
            1.000575811989361,
            1.000230285020825,
            1.000230285020825,
            1.000230285020825,
            1.000115135882276,
            1.000460623072841,
            0.999769767998157,
            1.0,
            1.0,
            1.000230285020825,
            0.999884877372468,
            1.000115135882277,
            1.,
            0.999884877372469,
            1.0,
            1.0,
            0.999769767998157,
            1.0,
            0.999539589003088,
            1.000115135882277,
            0.999884877372468,
            1.000115135882277,
            1.000115135882277,
            1.0,
            0.999884877372469,
            0.999539589003088,
            0.999539589003088,
            0.999654671875538,
            0.999654671875538,
            0.99942451937928,
            0.99942451937928,
            0.99942451937928,
            0.999539589003088,
            0.99942451937928,
            0.999654671875538,
            0.999654671875538,
            0.999654671875538,
            0.999539589003088,
            0.999884877372469,
            0.999884877372469,
            1.000115135882277,
            0.999884877372468,
            0.999539589003088,
            0.999884877372469,
            0.99942451937928,
            0.99942451937928,
            0.999654671875538,
            0.99942451937928,
            0.999884877372469,
            0.99930946300259,
            0.999884877372469,
            0.999654671875538,
            0.999194419871492,
            1.000115135882277,
            0.999769767998157,
            0.999539589003088,
            0.999654671875538,
            1.000115135882277,
            0.999769767998156,
            1.000115135882277,
            1.00046062307284,
            0.999884877372469,
            0.999654671875538,
            0.999654671875538,
            0.999079389984462,
            0.999654671875538,
            0.998504439156965,
            1.000230285020825,
            0.999884877372469,
            0.999884877372469,
            0.998964373339974,
            0.998274551481088,
            0.999194419871492,
            0.998389488702327,
            0.999194419871492,
            0.998274551481089,
            1.00046062307284,
            0.999769767998157,
            1.0,
            0.99873437977253,
            1.0,
            1.000921458319296,
            0.998964373339975,
            1.001613109228309,
            0.999654671875538,
            1.000575811989361,
            0.999079389984462,
            0.999769767998157,
            1.001959113889896,
            1.0,
            1.001267224051863,
            0.997240711741549,
            0.99873437977253,
            0.997700063822553,
            1.001959113889896,
            1.00046062307284,
            1.002536053960523,
            0.998389488702328,
            1.00265148183361,
            1.001613109228309,
            0.998044716732712,
            1.001036700294489,
            0.997929819202528,
            1.004037653359821,
            1.001036700294489,
            0.997700063822553,
            0.99942451937928,
            0.9946035385275,
            1.001036700294489,
            1.001497800895405,
            1.004268868191733,
            0.998044716732711,
            0.996093256598079,
            0.997585205969718,
            0.999769767998157,
            0.993230391713372,
            1.000230285020825,
            0.997585205969718,
            0.997125906770532,
            1.001382505837099,
            0.997125906770532,
            0.998964373339974,
            1.00484713824658,
            0.993001718291996,
            1.001959113889896,
            1.003690930920097,
            1.000806229611062,
            1.000806229611062,
            0.997929819202528,
            0.998504439156965,
            0.998504439156966,
            1.000806229611062,
            1.000115135882276,
            1.00046062307284,
            0.99873437977253,
            0.999539589003088,
            0.999884877372468,
            0.99930946300259,
            0.998849369936505,
            0.998619402846525,
            0.997814934899649,
            0.99873437977253,
            0.99930946300259,
            0.99930946300259,
            0.999769767998156,
            1.0,
            1.000115135882277,
            0.999079389984462,
            0.998849369936505,
            0.998159627491724,
            0.998619402846525,
            0.999079389984462,
            0.998159627491724,
            0.998389488702328,
            1.000115135882277,
            0.999769767998157,
            0.999079389984461,
            0.998044716732711,
            1.000575811989361,
            0.999769767998157,
            0.999194419871492,
            0.999194419871492,
            1.000460623072841,
            1.000575811989361,
            1.002074475336479,
            0.999539589003088,
            0.998964373339974,
            0.999539589003088,
            1.001036700294489,
            1.000345447417171,
            1.000345447417171,
            1.00046062307284,
            0.998044716732711,
            1.001151955538169,
            1.000230285020825,
            0.999654671875538,
            1.002420639375797,
            0.998389488702328,
            1.001267224051863,
            0.999539589003088,
            0.999194419871492,
            1.000115135882277,
            0.998619402846525,
            0.999654671875538,
            1.001036700294489,
            1.003344328213383,
            0.996896336476927,
            1.001382505837099,
            1.002420639375796,
            1.000921458319296,
            1.00034544741717,
            0.998159627491724,
            1.001036700294489,
            0.998849369936505,
            1.001151955538169,
            0.998389488702328,
            1.000575811989361,
            0.998389488702328,
            1.000575811989361,
            0.998849369936505,
            1.0,
            0.999769767998156,
            0.998159627491724,
            1.001267224051863,
            1.000691014168259,
            1.0035753833829,
            1.000691014168259,
            1.002997845198331,
            0.99930946300259,
            0.999884877372469,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::Db,
            vec![
                vec![
                    (-0.007, -2.703),
                    (-33.308, 87.205),
                    (-33.310, 87.223),
                    (0.002, -2.522),
                ],
                vec![
                    (-0.014, -5.380),
                    (-27.320, 84.747),
                    (-27.319, 84.763),
                    (-0.006, -5.048),
                ],
                vec![
                    (-0.020, -8.054),
                    (-23.824, 82.282),
                    (-23.822, 82.309),
                    (-0.011, -7.535),
                ],
                vec![
                    (-0.030, -10.721),
                    (-21.356, 79.850),
                    (-21.355, 79.863),
                    (-0.022, -10.008),
                ],
                vec![
                    (-0.070, -12.934),
                    (-19.767, 76.759),
                    (-19.766, 76.774),
                    (-0.073, -12.097),
                ],
                vec![
                    (-0.102, -15.528),
                    (-18.235, 74.354),
                    (-18.233, 74.341),
                    (-0.086, -14.420),
                ],
                vec![
                    (-0.122, -18.012),
                    (-16.968, 71.927),
                    (-16.964, 71.952),
                    (-0.118, -16.765),
                ],
                vec![
                    (-0.156, -20.492),
                    (-15.876, 69.541),
                    (-15.872, 69.531),
                    (-0.143, -19.112),
                ],
                vec![
                    (-0.185, -23.013),
                    (-14.932, 67.149),
                    (-14.928, 67.141),
                    (-0.181, -21.370),
                ],
                vec![
                    (-0.224, -25.457),
                    (-14.103, 64.781),
                    (-14.100, 64.755),
                    (-0.214, -23.659),
                ],
                vec![
                    (-0.265, -27.878),
                    (-13.365, 62.489),
                    (-13.360, 62.451),
                    (-0.244, -25.890),
                ],
                vec![
                    (-0.302, -30.240),
                    (-12.705, 60.203),
                    (-12.703, 60.163),
                    (-0.284, -28.109),
                ],
                vec![
                    (-0.344, -32.603),
                    (-12.109, 57.908),
                    (-12.107, 57.880),
                    (-0.322, -30.298),
                ],
                vec![
                    (-0.387, -34.903),
                    (-11.574, 55.657),
                    (-11.572, 55.620),
                    (-0.364, -32.496),
                ],
                vec![
                    (-0.429, -37.197),
                    (-11.087, 53.446),
                    (-11.086, 53.458),
                    (-0.407, -34.613),
                ],
                vec![
                    (-0.463, -39.356),
                    (-10.675, 51.542),
                    (-10.671, 51.469),
                    (-0.446, -36.438),
                ],
                vec![
                    (-0.515, -41.554),
                    (-10.268, 49.344),
                    (-10.270, 49.333),
                    (-0.488, -38.514),
                ],
                vec![
                    (-0.562, -43.746),
                    (-9.897, 47.250),
                    (-9.897, 47.193),
                    (-0.536, -40.558),
                ],
                vec![
                    (-0.610, -45.867),
                    (-9.555, 45.112),
                    (-9.555, 45.063),
                    (-0.582, -42.571),
                ],
                vec![
                    (-0.660, -47.981),
                    (-9.247, 43.031),
                    (-9.245, 42.983),
                    (-0.630, -44.513),
                ],
                vec![
                    (-0.704, -50.043),
                    (-8.963, 41.024),
                    (-8.964, 40.994),
                    (-0.679, -46.423),
                ],
                vec![
                    (-0.757, -52.064),
                    (-8.703, 39.062),
                    (-8.702, 39.007),
                    (-0.720, -48.330),
                ],
                vec![
                    (-0.799, -54.087),
                    (-8.460, 37.129),
                    (-8.460, 37.075),
                    (-0.767, -50.166),
                ],
                vec![
                    (-0.851, -56.013),
                    (-8.237, 35.258),
                    (-8.238, 35.248),
                    (-0.800, -51.986),
                ],
                vec![
                    (-0.898, -57.881),
                    (-8.025, 33.392),
                    (-8.025, 33.331),
                    (-0.839, -53.849),
                ],
                vec![
                    (-0.939, -59.713),
                    (-7.832, 31.566),
                    (-7.832, 31.496),
                    (-0.882, -55.676),
                ],
                vec![
                    (-0.982, -61.565),
                    (-7.656, 29.737),
                    (-7.658, 29.701),
                    (-0.927, -57.406),
                ],
                vec![
                    (-1.021, -63.357),
                    (-7.495, 27.999),
                    (-7.495, 27.920),
                    (-0.968, -59.135),
                ],
                vec![
                    (-1.056, -65.122),
                    (-7.345, 26.233),
                    (-7.349, 26.172),
                    (-1.010, -60.814),
                ],
                vec![
                    (-1.093, -66.877),
                    (-7.211, 24.513),
                    (-7.210, 24.431),
                    (-1.045, -62.470),
                ],
                vec![
                    (-1.130, -68.611),
                    (-7.084, 22.839),
                    (-7.085, 22.760),
                    (-1.077, -64.114),
                ],
                vec![
                    (-1.166, -70.296),
                    (-6.964, 21.187),
                    (-6.963, 21.105),
                    (-1.113, -65.762),
                ],
                vec![
                    (-1.199, -71.933),
                    (-6.856, 19.554),
                    (-6.855, 19.467),
                    (-1.149, -67.361),
                ],
                vec![
                    (-1.228, -73.544),
                    (-6.761, 17.969),
                    (-6.761, 17.907),
                    (-1.184, -68.913),
                ],
                vec![
                    (-1.258, -75.146),
                    (-6.668, 16.427),
                    (-6.669, 16.307),
                    (-1.209, -70.427),
                ],
                vec![
                    (-1.293, -76.757),
                    (-6.583, 14.877),
                    (-6.587, 14.748),
                    (-1.235, -71.896),
                ],
                vec![
                    (-1.316, -78.293),
                    (-6.503, 13.359),
                    (-6.507, 13.277),
                    (-1.261, -73.391),
                ],
                vec![
                    (-1.339, -79.774),
                    (-6.431, 11.869),
                    (-6.434, 11.757),
                    (-1.286, -74.841),
                ],
                vec![
                    (-1.362, -81.225),
                    (-6.369, 10.428),
                    (-6.372, 10.325),
                    (-1.308, -76.268),
                ],
                vec![
                    (-1.387, -82.666),
                    (-6.312, 8.931),
                    (-6.317, 8.864),
                    (-1.330, -77.697),
                ],
                vec![
                    (-1.405, -84.120),
                    (-6.257, 7.509),
                    (-6.262, 7.430),
                    (-1.347, -79.106),
                ],
                vec![
                    (-1.419, -85.518),
                    (-6.208, 6.101),
                    (-6.213, 5.976),
                    (-1.365, -80.482),
                ],
                vec![
                    (-1.429, -86.867),
                    (-6.165, 4.717),
                    (-6.169, 4.653),
                    (-1.384, -81.874),
                ],
                vec![
                    (-1.447, -88.250),
                    (-6.127, 3.340),
                    (-6.132, 3.226),
                    (-1.400, -83.269),
                ],
                vec![
                    (-1.465, -89.627),
                    (-6.089, 1.959),
                    (-6.092, 1.849),
                    (-1.410, -84.648),
                ],
                vec![
                    (-1.477, -90.985),
                    (-6.063, 0.594),
                    (-6.066, 0.471),
                    (-1.424, -85.964),
                ],
                vec![
                    (-1.481, -92.320),
                    (-6.052, -0.794),
                    (-6.055, -0.900),
                    (-1.441, -87.215),
                ],
                vec![
                    (-1.488, -93.626),
                    (-6.056, -2.034),
                    (-6.060, -2.165),
                    (-1.466, -88.427),
                ],
                vec![
                    (-1.504, -94.858),
                    (-6.038, -3.236),
                    (-6.039, -3.361),
                    (-1.474, -89.665),
                ],
                vec![
                    (-1.520, -96.087),
                    (-6.023, -4.482),
                    (-6.024, -4.590),
                    (-1.486, -90.966),
                ],
                vec![
                    (-1.522, -97.307),
                    (-6.009, -5.661),
                    (-6.008, -5.786),
                    (-1.490, -92.174),
                ],
                vec![
                    (-1.518, -98.492),
                    (-6.002, -6.874),
                    (-6.003, -6.991),
                    (-1.485, -93.349),
                ],
                vec![
                    (-1.517, -99.685),
                    (-5.991, -8.055),
                    (-5.995, -8.196),
                    (-1.487, -94.556),
                ],
                vec![
                    (-1.539, -100.827),
                    (-5.988, -9.359),
                    (-5.989, -9.478),
                    (-1.508, -95.800),
                ],
                vec![
                    (-1.549, -101.915),
                    (-6.011, -10.524),
                    (-6.016, -10.644),
                    (-1.520, -96.906),
                ],
                vec![
                    (-1.535, -103.121),
                    (-6.030, -11.560),
                    (-6.035, -11.671),
                    (-1.513, -97.983),
                ],
                vec![
                    (-1.527, -104.305),
                    (-6.029, -12.498),
                    (-6.032, -12.599),
                    (-1.483, -99.050),
                ],
                vec![
                    (-1.536, -105.323),
                    (-6.025, -13.680),
                    (-6.030, -13.823),
                    (-1.482, -100.338),
                ],
                vec![
                    (-1.543, -106.386),
                    (-6.035, -14.827),
                    (-6.036, -14.953),
                    (-1.508, -101.481),
                ],
                vec![
                    (-1.532, -107.502),
                    (-6.043, -15.925),
                    (-6.049, -16.028),
                    (-1.519, -102.578),
                ],
                vec![
                    (-1.523, -108.630),
                    (-6.073, -17.002),
                    (-6.074, -17.148),
                    (-1.520, -103.680),
                ],
                vec![
                    (-1.517, -109.630),
                    (-6.085, -18.045),
                    (-6.088, -18.163),
                    (-1.507, -104.645),
                ],
                vec![
                    (-1.517, -110.526),
                    (-6.108, -19.017),
                    (-6.115, -19.129),
                    (-1.502, -105.568),
                ],
                vec![
                    (-1.523, -111.551),
                    (-6.133, -19.965),
                    (-6.132, -20.109),
                    (-1.491, -106.576),
                ],
                vec![
                    (-1.520, -112.602),
                    (-6.146, -20.916),
                    (-6.148, -21.090),
                    (-1.497, -107.712),
                ],
                vec![
                    (-1.505, -113.579),
                    (-6.160, -21.929),
                    (-6.164, -22.133),
                    (-1.493, -108.803),
                ],
                vec![
                    (-1.496, -114.428),
                    (-6.187, -22.916),
                    (-6.190, -23.068),
                    (-1.490, -109.691),
                ],
                vec![
                    (-1.492, -115.261),
                    (-6.202, -23.910),
                    (-6.201, -24.079),
                    (-1.478, -110.630),
                ],
                vec![
                    (-1.479, -116.159),
                    (-6.233, -24.888),
                    (-6.235, -25.011),
                    (-1.486, -111.506),
                ],
                vec![
                    (-1.450, -117.227),
                    (-6.269, -25.884),
                    (-6.268, -25.983),
                    (-1.472, -112.436),
                ],
                vec![
                    (-1.415, -118.310),
                    (-6.304, -26.742),
                    (-6.300, -26.933),
                    (-1.449, -113.459),
                ],
                vec![
                    (-1.406, -119.173),
                    (-6.318, -27.627),
                    (-6.319, -27.787),
                    (-1.436, -114.468),
                ],
                vec![
                    (-1.416, -120.155),
                    (-6.333, -28.537),
                    (-6.336, -28.705),
                    (-1.442, -115.341),
                ],
                vec![
                    (-1.412, -121.119),
                    (-6.354, -29.574),
                    (-6.357, -29.733),
                    (-1.429, -116.311),
                ],
                vec![
                    (-1.375, -122.107),
                    (-6.378, -30.598),
                    (-6.386, -30.749),
                    (-1.411, -117.337),
                ],
                vec![
                    (-1.348, -122.983),
                    (-6.417, -31.525),
                    (-6.420, -31.689),
                    (-1.405, -118.305),
                ],
                vec![
                    (-1.340, -123.747),
                    (-6.459, -32.495),
                    (-6.472, -32.682),
                    (-1.397, -119.061),
                ],
                vec![
                    (-1.351, -124.603),
                    (-6.527, -33.267),
                    (-6.525, -33.467),
                    (-1.388, -119.865),
                ],
                vec![
                    (-1.339, -125.547),
                    (-6.566, -34.045),
                    (-6.567, -34.205),
                    (-1.374, -120.594),
                ],
                vec![
                    (-1.313, -126.503),
                    (-6.593, -34.907),
                    (-6.594, -35.053),
                    (-1.332, -121.429),
                ],
                vec![
                    (-1.287, -127.265),
                    (-6.622, -35.784),
                    (-6.631, -36.002),
                    (-1.303, -122.280),
                ],
                vec![
                    (-1.265, -127.923),
                    (-6.640, -36.573),
                    (-6.655, -36.827),
                    (-1.275, -123.026),
                ],
                vec![
                    (-1.276, -128.868),
                    (-6.676, -37.454),
                    (-6.683, -37.685),
                    (-1.248, -124.052),
                ],
                vec![
                    (-1.276, -129.931),
                    (-6.720, -38.438),
                    (-6.734, -38.649),
                    (-1.253, -124.964),
                ],
                vec![
                    (-1.258, -130.950),
                    (-6.775, -39.360),
                    (-6.782, -39.607),
                    (-1.251, -125.870),
                ],
                vec![
                    (-1.250, -131.674),
                    (-6.823, -40.202),
                    (-6.838, -40.386),
                    (-1.227, -126.610),
                ],
                vec![
                    (-1.245, -132.330),
                    (-6.872, -40.937),
                    (-6.868, -41.097),
                    (-1.203, -127.499),
                ],
                vec![
                    (-1.264, -133.188),
                    (-6.919, -41.663),
                    (-6.921, -41.828),
                    (-1.186, -128.275),
                ],
                vec![
                    (-1.241, -133.962),
                    (-6.979, -42.437),
                    (-6.979, -42.740),
                    (-1.181, -129.251),
                ],
                vec![
                    (-1.244, -134.873),
                    (-7.035, -43.207),
                    (-7.046, -43.432),
                    (-1.178, -130.173),
                ],
                vec![
                    (-1.233, -135.444),
                    (-7.079, -43.963),
                    (-7.079, -44.149),
                    (-1.163, -130.927),
                ],
                vec![
                    (-1.204, -137.506),
                    (-7.295, -45.675),
                    (-7.287, -45.895),
                    (-1.120, -132.380),
                ],
                vec![
                    (-1.196, -138.266),
                    (-7.331, -46.540),
                    (-7.340, -46.723),
                    (-1.101, -132.999),
                ],
                vec![
                    (-1.184, -139.237),
                    (-7.406, -47.224),
                    (-7.392, -47.498),
                    (-1.081, -133.941),
                ],
                vec![
                    (-1.156, -139.806),
                    (-7.439, -47.942),
                    (-7.442, -48.261),
                    (-1.066, -134.851),
                ],
                vec![
                    (-1.141, -140.499),
                    (-7.512, -48.823),
                    (-7.507, -49.026),
                    (-1.057, -135.763),
                ],
                vec![
                    (-1.151, -141.070),
                    (-7.544, -49.468),
                    (-7.552, -49.677),
                    (-1.033, -136.450),
                ],
                vec![
                    (-1.139, -141.988),
                    (-7.619, -50.241),
                    (-7.621, -50.511),
                    (-1.060, -137.163),
                ],
                vec![
                    (-1.126, -142.624),
                    (-7.678, -51.043),
                    (-7.661, -51.286),
                    (-1.045, -138.047),
                ],
                vec![
                    (-1.090, -143.317),
                    (-7.754, -51.827),
                    (-7.754, -51.818),
                    (-1.059, -138.695),
                ],
                vec![
                    (-1.090, -143.804),
                    (-7.812, -52.472),
                    (-7.801, -52.722),
                    (-0.973, -139.574),
                ],
                vec![
                    (-1.082, -144.444),
                    (-7.886, -53.018),
                    (-7.910, -53.153),
                    (-0.968, -140.498),
                ],
                vec![
                    (-1.091, -145.084),
                    (-7.937, -53.366),
                    (-7.948, -53.552),
                    (-0.981, -140.970),
                ],
                vec![
                    (-1.073, -145.770),
                    (-7.979, -53.983),
                    (-7.999, -54.149),
                    (-0.994, -141.596),
                ],
                vec![
                    (-1.063, -146.276),
                    (-8.043, -54.559),
                    (-8.026, -54.782),
                    (-1.008, -142.399),
                ],
                vec![
                    (-1.042, -146.836),
                    (-8.078, -55.281),
                    (-8.074, -55.531),
                    (-1.001, -143.284),
                ],
                vec![
                    (-1.033, -147.673),
                    (-8.132, -56.080),
                    (-8.110, -56.310),
                    (-0.981, -143.668),
                ],
                vec![
                    (-1.042, -148.451),
                    (-8.160, -56.728),
                    (-8.174, -56.898),
                    (-0.965, -144.102),
                ],
                vec![
                    (-1.024, -148.900),
                    (-8.249, -57.285),
                    (-8.226, -57.521),
                    (-0.941, -144.796),
                ],
                vec![
                    (-0.995, -149.415),
                    (-8.286, -57.910),
                    (-8.272, -58.101),
                    (-0.938, -145.596),
                ],
                vec![
                    (-0.989, -150.029),
                    (-8.310, -58.664),
                    (-8.327, -58.674),
                    (-0.907, -146.361),
                ],
                vec![
                    (-0.991, -150.499),
                    (-8.371, -59.225),
                    (-8.362, -59.245),
                    (-0.863, -146.713),
                ],
                vec![
                    (-1.005, -150.999),
                    (-8.404, -59.719),
                    (-8.422, -59.990),
                    (-0.836, -147.464),
                ],
                vec![
                    (-0.991, -151.557),
                    (-8.484, -60.340),
                    (-8.449, -60.444),
                    (-0.834, -148.351),
                ],
                vec![
                    (-0.992, -152.156),
                    (-8.521, -61.244),
                    (-8.512, -61.261),
                    (-0.813, -149.016),
                ],
                vec![
                    (-0.994, -152.469),
                    (-8.563, -61.662),
                    (-8.583, -62.009),
                    (-0.825, -149.822),
                ],
                vec![
                    (-0.996, -152.775),
                    (-8.620, -62.219),
                    (-8.625, -62.491),
                    (-0.844, -150.308),
                ],
                vec![
                    (-0.998, -153.136),
                    (-8.656, -62.852),
                    (-8.703, -63.102),
                    (-0.866, -150.828),
                ],
                vec![
                    (-0.956, -153.739),
                    (-8.707, -63.513),
                    (-8.698, -63.648),
                    (-0.869, -151.174),
                ],
                vec![
                    (-0.964, -154.217),
                    (-8.762, -64.131),
                    (-8.749, -64.117),
                    (-0.803, -151.983),
                ],
                vec![
                    (-0.949, -154.618),
                    (-8.809, -64.854),
                    (-8.772, -65.287),
                    (-0.774, -152.579),
                ],
                vec![
                    (-0.920, -155.032),
                    (-8.834, -65.647),
                    (-8.851, -65.891),
                    (-0.753, -153.294),
                ],
                vec![
                    (-0.903, -155.591),
                    (-8.860, -66.464),
                    (-8.894, -66.700),
                    (-0.741, -154.031),
                ],
                vec![
                    (-0.907, -156.265),
                    (-8.938, -67.399),
                    (-8.959, -67.342),
                    (-0.720, -154.847),
                ],
                vec![
                    (-0.896, -156.802),
                    (-9.031, -67.658),
                    (-9.033, -68.226),
                    (-0.692, -155.857),
                ],
                vec![
                    (-0.856, -157.201),
                    (-9.084, -68.597),
                    (-9.143, -68.901),
                    (-0.706, -156.318),
                ],
                vec![
                    (-0.874, -157.546),
                    (-9.202, -69.240),
                    (-9.200, -69.721),
                    (-0.727, -156.732),
                ],
                vec![
                    (-0.922, -158.150),
                    (-9.258, -69.906),
                    (-9.279, -70.164),
                    (-0.807, -157.422),
                ],
                vec![
                    (-0.947, -158.704),
                    (-9.308, -70.926),
                    (-9.333, -70.844),
                    (-0.835, -158.104),
                ],
                vec![
                    (-0.895, -159.033),
                    (-9.413, -71.526),
                    (-9.401, -71.758),
                    (-0.804, -158.610),
                ],
                vec![
                    (-0.854, -159.318),
                    (-9.522, -72.082),
                    (-9.547, -72.352),
                    (-0.752, -158.691),
                ],
                vec![
                    (-0.838, -159.873),
                    (-9.683, -72.618),
                    (-9.692, -73.056),
                    (-0.775, -159.259),
                ],
                vec![
                    (-0.849, -160.766),
                    (-9.746, -73.258),
                    (-9.704, -73.203),
                    (-0.850, -160.019),
                ],
                vec![
                    (-0.846, -161.398),
                    (-9.808, -73.008),
                    (-9.869, -73.463),
                    (-0.825, -161.307),
                ],
                vec![
                    (-0.756, -161.361),
                    (-9.948, -73.728),
                    (-9.931, -73.058),
                    (-0.713, -161.524),
                ],
                vec![
                    (-0.761, -162.046),
                    (-10.012, -74.283),
                    (-9.980, -73.262),
                    (-0.727, -162.222),
                ],
                vec![
                    (-0.760, -162.772),
                    (-10.097, -74.685),
                    (-10.090, -73.887),
                    (-0.778, -163.039),
                ],
                vec![
                    (-0.759, -163.211),
                    (-10.197, -74.987),
                    (-10.190, -74.236),
                    (-0.827, -163.492),
                ],
                vec![
                    (-0.743, -163.534),
                    (-10.288, -75.174),
                    (-10.306, -74.376),
                    (-0.845, -163.829),
                ],
                vec![
                    (-0.718, -164.022),
                    (-10.406, -75.414),
                    (-10.419, -74.530),
                    (-0.869, -164.190),
                ],
                vec![
                    (-0.713, -164.578),
                    (-10.501, -75.436),
                    (-10.514, -74.649),
                    (-0.882, -164.538),
                ],
                vec![
                    (-0.721, -164.938),
                    (-10.563, -75.321),
                    (-10.556, -74.375),
                    (-0.872, -164.822),
                ],
                vec![
                    (-0.740, -165.270),
                    (-10.616, -75.215),
                    (-10.615, -74.492),
                    (-0.882, -164.933),
                ],
                vec![
                    (-0.727, -165.532),
                    (-10.636, -75.164),
                    (-10.632, -74.424),
                    (-0.853, -165.150),
                ],
                vec![
                    (-0.704, -166.050),
                    (-10.623, -75.065),
                    (-10.634, -74.288),
                    (-0.849, -165.342),
                ],
                vec![
                    (-0.701, -166.646),
                    (-10.619, -75.230),
                    (-10.623, -74.467),
                    (-0.840, -165.459),
                ],
                vec![
                    (-0.696, -167.209),
                    (-10.617, -75.602),
                    (-10.618, -74.787),
                    (-0.809, -165.838),
                ],
                vec![
                    (-0.716, -167.810),
                    (-10.631, -75.931),
                    (-10.637, -75.128),
                    (-0.774, -166.238),
                ],
                vec![
                    (-0.741, -168.274),
                    (-10.612, -76.177),
                    (-10.622, -75.507),
                    (-0.745, -166.740),
                ],
                vec![
                    (-0.764, -168.664),
                    (-10.584, -76.727),
                    (-10.596, -76.030),
                    (-0.731, -167.341),
                ],
                vec![
                    (-0.765, -168.885),
                    (-10.590, -77.366),
                    (-10.609, -76.537),
                    (-0.708, -167.767),
                ],
                vec![
                    (-0.772, -169.145),
                    (-10.629, -77.899),
                    (-10.640, -77.188),
                    (-0.683, -168.348),
                ],
                vec![
                    (-0.750, -169.444),
                    (-10.667, -78.526),
                    (-10.673, -77.889),
                    (-0.677, -169.053),
                ],
                vec![
                    (-0.736, -169.760),
                    (-10.715, -79.048),
                    (-10.721, -78.526),
                    (-0.659, -169.607),
                ],
                vec![
                    (-0.729, -169.988),
                    (-10.774, -79.624),
                    (-10.776, -79.055),
                    (-0.645, -170.279),
                ],
                vec![
                    (-0.726, -170.132),
                    (-10.837, -80.185),
                    (-10.837, -79.496),
                    (-0.652, -170.933),
                ],
                vec![
                    (-0.700, -170.491),
                    (-10.903, -80.593),
                    (-10.902, -79.850),
                    (-0.681, -171.309),
                ],
                vec![
                    (-0.666, -170.890),
                    (-10.965, -81.065),
                    (-10.973, -80.269),
                    (-0.694, -171.561),
                ],
                vec![
                    (-0.659, -171.335),
                    (-11.040, -81.499),
                    (-11.050, -80.629),
                    (-0.707, -171.762),
                ],
                vec![
                    (-0.626, -171.867),
                    (-11.113, -81.675),
                    (-11.129, -80.926),
                    (-0.693, -172.083),
                ],
                vec![
                    (-0.607, -172.055),
                    (-11.179, -81.751),
                    (-11.191, -80.954),
                    (-0.685, -172.412),
                ],
                vec![
                    (-0.604, -172.623),
                    (-11.229, -82.010),
                    (-11.237, -81.134),
                    (-0.692, -172.867),
                ],
                vec![
                    (-0.581, -173.185),
                    (-11.233, -82.302),
                    (-11.249, -81.467),
                    (-0.727, -173.284),
                ],
                vec![
                    (-0.555, -173.548),
                    (-11.270, -82.562),
                    (-11.284, -81.615),
                    (-0.719, -173.825),
                ],
                vec![
                    (-0.529, -174.035),
                    (-11.295, -82.750),
                    (-11.294, -81.725),
                    (-0.699, -174.420),
                ],
                vec![
                    (-0.493, -174.425),
                    (-11.329, -82.919),
                    (-11.331, -81.878),
                    (-0.714, -174.634),
                ],
                vec![
                    (-0.474, -174.909),
                    (-11.354, -83.028),
                    (-11.362, -82.069),
                    (-0.722, -174.945),
                ],
                vec![
                    (-0.463, -175.232),
                    (-11.350, -83.348),
                    (-11.367, -82.421),
                    (-0.717, -175.277),
                ],
                vec![
                    (-0.432, -175.548),
                    (-11.368, -83.709),
                    (-11.363, -82.693),
                    (-0.739, -175.601),
                ],
                vec![
                    (-0.393, -176.181),
                    (-11.396, -84.163),
                    (-11.398, -83.231),
                    (-0.752, -175.840),
                ],
                vec![
                    (-0.385, -176.735),
                    (-11.445, -84.475),
                    (-11.452, -83.539),
                    (-0.736, -176.095),
                ],
                vec![
                    (-0.378, -177.418),
                    (-11.479, -84.621),
                    (-11.486, -83.766),
                    (-0.728, -176.458),
                ],
                vec![
                    (-0.357, -178.081),
                    (-11.505, -84.919),
                    (-11.501, -84.040),
                    (-0.745, -176.797),
                ],
                vec![
                    (-0.373, -178.550),
                    (-11.532, -85.329),
                    (-11.527, -84.379),
                    (-0.737, -177.162),
                ],
                vec![
                    (-0.398, -179.219),
                    (-11.555, -85.876),
                    (-11.537, -84.887),
                    (-0.712, -177.517),
                ],
                vec![
                    (-0.406, -179.387),
                    (-11.592, -86.304),
                    (-11.596, -85.116),
                    (-0.697, -178.130),
                ],
                vec![
                    (-0.388, -179.751),
                    (-11.608, -86.455),
                    (-11.617, -85.489),
                    (-0.708, -178.407),
                ],
                vec![
                    (-0.351, 179.972),
                    (-11.662, -86.823),
                    (-11.666, -85.790),
                    (-0.737, -178.606),
                ],
                vec![
                    (-0.309, 179.552),
                    (-11.715, -87.019),
                    (-11.706, -85.940),
                    (-0.744, -178.758),
                ],
                vec![
                    (-0.265, 179.038),
                    (-11.732, -87.209),
                    (-11.729, -86.175),
                    (-0.744, -178.952),
                ],
                vec![
                    (-0.268, 178.546),
                    (-11.731, -87.406),
                    (-11.728, -86.540),
                    (-0.754, -179.276),
                ],
                vec![
                    (-0.257, 178.084),
                    (-11.753, -87.778),
                    (-11.749, -87.009),
                    (-0.785, -179.361),
                ],
                vec![
                    (-0.252, 177.567),
                    (-11.757, -88.069),
                    (-11.774, -87.224),
                    (-0.760, -179.478),
                ],
                vec![
                    (-0.264, 177.133),
                    (-11.802, -88.337),
                    (-11.792, -87.409),
                    (-0.743, -179.786),
                ],
                vec![
                    (-0.284, 176.662),
                    (-11.827, -88.625),
                    (-11.825, -87.942),
                    (-0.737, 179.938),
                ],
                vec![
                    (-0.281, 176.114),
                    (-11.838, -88.888),
                    (-11.841, -88.009),
                    (-0.737, 179.650),
                ],
                vec![
                    (-0.284, 175.439),
                    (-11.849, -89.096),
                    (-11.828, -88.048),
                    (-0.699, 179.298),
                ],
                vec![
                    (-0.315, 174.679),
                    (-11.814, -89.502),
                    (-11.828, -88.608),
                    (-0.682, 178.625),
                ],
                vec![
                    (-0.346, 174.031),
                    (-11.835, -89.972),
                    (-11.824, -88.968),
                    (-0.720, 178.373),
                ],
                vec![
                    (-0.338, 173.956),
                    (-11.827, -90.109),
                    (-11.831, -89.305),
                    (-0.713, 178.598),
                ],
                vec![
                    (-0.273, 173.583),
                    (-11.855, -90.206),
                    (-11.862, -89.369),
                    (-0.648, 178.595),
                ],
                vec![
                    (-0.266, 173.005),
                    (-11.846, -90.457),
                    (-11.845, -89.570),
                    (-0.609, 178.294),
                ],
                vec![
                    (-0.232, 172.456),
                    (-11.843, -90.902),
                    (-11.855, -90.128),
                    (-0.612, 177.855),
                ],
                vec![
                    (-0.223, 172.030),
                    (-11.847, -91.056),
                    (-11.850, -90.397),
                    (-0.575, 177.768),
                ],
                vec![
                    (-0.227, 171.491),
                    (-11.866, -91.933),
                    (-11.857, -91.056),
                    (-0.561, 177.593),
                ],
                vec![
                    (-0.211, 170.829),
                    (-11.908, -92.002),
                    (-11.879, -91.092),
                    (-0.489, 177.305),
                ],
                vec![
                    (-0.233, 170.119),
                    (-11.866, -92.554),
                    (-11.893, -91.468),
                    (-0.476, 176.579),
                ],
                vec![
                    (-0.204, 169.606),
                    (-11.915, -92.825),
                    (-11.903, -92.017),
                    (-0.448, 176.373),
                ],
                vec![
                    (-0.235, 168.777),
                    (-11.933, -93.385),
                    (-11.912, -92.605),
                    (-0.420, 175.689),
                ],
                vec![
                    (-0.271, 167.832),
                    (-11.947, -93.891),
                    (-11.939, -93.206),
                    (-0.472, 175.246),
                ],
                vec![
                    (-0.306, 167.102),
                    (-11.957, -94.149),
                    (-11.954, -93.210),
                    (-0.448, 175.324),
                ],
                vec![
                    (-0.258, 166.964),
                    (-11.928, -94.426),
                    (-11.944, -93.481),
                    (-0.419, 175.523),
                ],
                vec![
                    (-0.229, 165.980),
                    (-11.950, -94.223),
                    (-11.941, -93.532),
                    (-0.339, 175.625),
                ],
                vec![
                    (-0.213, 165.242),
                    (-11.921, -94.718),
                    (-11.931, -93.936),
                    (-0.281, 175.172),
                ],
                vec![
                    (-0.198, 164.306),
                    (-11.915, -95.109),
                    (-11.905, -94.295),
                    (-0.217, 174.767),
                ],
                vec![
                    (-0.188, 163.445),
                    (-11.865, -95.670),
                    (-11.879, -94.820),
                    (-0.205, 174.577),
                ],
                vec![
                    (-0.181, 162.394),
                    (-11.874, -96.178),
                    (-11.869, -95.263),
                    (-0.135, 174.270),
                ],
                vec![
                    (-0.234, 161.079),
                    (-11.820, -96.781),
                    (-11.834, -95.848),
                    (-0.079, 173.670),
                ],
                vec![
                    (-0.311, 159.862),
                    (-11.797, -97.531),
                    (-11.792, -96.706),
                    (-0.026, 173.252),
                ],
                vec![
                    (-0.473, 158.035),
                    (-11.767, -99.069),
                    (-11.777, -98.139),
                    (0.031, 172.884),
                ],
                vec![
                    (-0.865, 156.756),
                    (-11.916, -100.603),
                    (-11.916, -99.695),
                    (0.055, 172.980),
                ],
                vec![
                    (-1.383, 157.170),
                    (-12.183, -101.839),
                    (-12.185, -100.941),
                    (0.148, 172.980),
                ],
                vec![
                    (-1.786, 159.984),
                    (-12.561, -101.601),
                    (-12.577, -100.705),
                    (0.301, 173.255),
                ],
                vec![
                    (-1.716, 163.563),
                    (-12.816, -99.966),
                    (-12.805, -98.952),
                    (0.502, 172.921),
                ],
                vec![
                    (-1.393, 165.231),
                    (-12.859, -97.814),
                    (-12.853, -96.821),
                    (0.657, 171.744),
                ],
                vec![
                    (-1.030, 165.580),
                    (-12.700, -96.201),
                    (-12.669, -95.453),
                    (0.747, 170.716),
                ],
                vec![
                    (-0.759, 164.875),
                    (-12.514, -95.622),
                    (-12.508, -94.678),
                    (0.780, 169.512),
                ],
                vec![
                    (-0.585, 164.044),
                    (-12.361, -95.610),
                    (-12.335, -94.626),
                    (0.755, 168.895),
                ],
                vec![
                    (-0.439, 163.431),
                    (-12.203, -95.395),
                    (-12.209, -94.420),
                    (0.799, 168.418),
                ],
                vec![
                    (-0.318, 162.847),
                    (-12.067, -95.224),
                    (-12.068, -94.311),
                    (0.841, 167.973),
                ],
            ],
            String::from("test_3"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:48 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Vg(V) = -1.500\n\
            Vg(mA) = -0.000\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n",
            ),
        );
        comp_array_f64(&exemplar_max_gain, &calc.max_gain(), margin, "max_gain(5)");
        comp_array_f64(
            &exemplar_max_stable_gain,
            &calc.max_stable_gain(),
            margin,
            "max_stable_gain(5)",
        );
    }

    #[test]
    fn network_mu() {
        let margin = F64Margin {
            epsilon: 1e-12,
            ulps: 4,
        };

        let exemplar_mu: Array1<f64> =
            array![0.321707950574174, -0.002550951441449, -1.449447552621264];
        let exemplar_mu_prime: Array1<f64> =
            array![0.976861302098624, -0.445955657662617, 0.0408012371672];
        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [c64(0.958, -0.263), c64(-0.846, 0.158)],
                    [c64(0.004, 0.022), c64(0.544, -0.129)],
                ],
                [
                    [c64(2.043, -0.982), c64(-0.238, 3.249)],
                    [c64(-1.421, 3.492), c64(0.123, -394.321)],
                ],
                [
                    [c64(21.329, -0.421), c64(-0.942, 24.282)],
                    [c64(0.138, 0.132), c64(0.329, -0.324)],
                ],
            ]),
            String::from(""),
            String::from(""),
        );
        comp_array_f64(&exemplar_mu, &calc.mu(), margin, "mu(1)");
        comp_array_f64(&exemplar_mu_prime, &calc.mu_prime(), margin, "mu_prime(1)");

        // test.s2p
        let exemplar_mu: Array1<f64> = array![
            0.033183769882311,
            0.041130227236983,
            0.037333083239591,
            0.054219705626116,
        ];
        let exemplar_mu_prime: Array1<f64> = array![
            0.888443470698538,
            0.796693849590944,
            0.717478768636478,
            0.655211604332342,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![0.5, 1.0, 1.5, 2.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(0.9881388526914863, -0.13442709904013195),
                        c64(0.0010346705444205045, 0.011178864909012504)
                    ],
                    [
                        c64(-7.363542304219899, 0.6742816969789206),
                        c64(0.5574653418486702, -0.06665134724424635)
                    ],
                ],
                [
                    [
                        c64(0.9578079036840927, -0.2633207328693372),
                        c64(0.0037206104781559784, 0.021909191616475577)
                    ],
                    [
                        c64(-7.130124628011368, 1.3277987152036197),
                        c64(0.5435045929943587, -0.12869941397967788)
                    ],
                ],
                [
                    [
                        c64(0.9133108288727866, -0.38508398385543624),
                        c64(0.008042664765986755, 0.03190603796445517)
                    ],
                    [
                        c64(-6.9151682810378095, 1.800750901131042),
                        c64(0.5235871604669029, -0.18886435408156288)
                    ],
                ],
                [
                    [
                        c64(0.849070850314753, -0.49577931076259807),
                        c64(0.01381064392153511, 0.04080882571424955)
                    ],
                    [
                        c64(-6.688405272002992, 2.4133819411904995),
                        c64(0.4942211124266797, -0.24774648346309974)
                    ],
                ],
            ]),
            String::from("test"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:43 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 2.500\n\
            set_Vd(mA) = 11.250\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Vg(V) = -0.520\n\
            Vg(mA) = -0.001\n\
            Vd(V) = 2.500\n\
            Vd(mA) = 11.076",
            ),
        );
        comp_array_f64(&exemplar_mu, &calc.mu(), margin, "mu(2)");
        comp_array_f64(&exemplar_mu_prime, &calc.mu_prime(), margin, "mu_prime(2)");

        // test_2.s2p
        let exemplar_mu: Array1<f64> = array![
            0.252051215740252,
            0.639380809343797,
            0.660662373486303,
            0.69037752619849,
            0.708765571701647,
            1.004336211402834,
            1.004917164578735,
            1.005237942144416,
            1.005597331926434,
            1.006145453887719,
            1.006374369921609,
            1.006766666575417,
            1.00715245828949,
            1.007529143828652,
            1.00794734571573,
            1.008629842881335,
            1.009200276311899,
            1.009761647988067,
            1.010317021471788,
            1.01103697473101,
            1.011596902415992,
            1.012112621589616,
            1.012208258413788,
            1.012345004990124,
            1.01314175903448,
            1.013582883971879,
            1.013780002102202,
            1.014000088663175,
            1.014241874547965,
            1.015093931051118,
            1.01537104180066,
            1.015545417236198,
            1.016001022423394,
            1.015817775984552,
            1.016282153201014,
            1.016626756288028,
            1.016408552823028,
            1.016716521316849,
            1.016734518650359,
            1.017377549729379,
            1.017153565762558,
            1.016689717529402,
            1.016611916271118,
            1.016324729119221,
            1.017250110364518,
            1.017275571927999,
            1.016961863871139,
            1.017675037329914,
            1.018008378546716,
            1.018596630892662,
            1.018917447657991,
            1.018664006399435,
            1.018578863561278,
            1.018418016860917,
            1.019423114127993,
            1.019332232223617,
            1.019679985452757,
            1.020820366141235,
            1.020389610620984,
            1.020832618312405,
            1.020989648930052,
            1.022024673643647,
            1.022455718919499,
            1.022603207118212,
            1.021579103606366,
            1.021737797504203,
            1.022354957597552,
            1.022369026552042,
            1.021305178354288,
            1.021365538793394,
            1.022617091007939,
            1.022383568844012,
            1.022827649715578,
            1.022487610878999,
            1.02370446760665,
            1.02404921015651,
            1.02299866895785,
            1.023264594562241,
            1.023028124402027,
            1.022045247091938,
            1.021605540100736,
            1.021653963202479,
            1.023414311056928,
            1.021619389966857,
            1.019901450286508,
            1.01852328886081,
            1.020148458838615,
            1.020391178345735,
            1.020395570758893,
            1.019977075358412,
            1.023593615489884,
            1.023653348911812,
            1.020745234321158,
            1.020330768822056,
            1.021327171035666,
            1.024048726284998,
            1.023346340878528,
            1.017981725037367,
            1.015055227508229,
            1.015743703873584,
            1.016515868693986,
            1.020075088790992,
            1.019038732804847,
            1.018520029085547,
            1.01717021172956,
            1.0165223452372,
            1.015598691663956,
            1.014590056533678,
            1.01437509734385,
            1.013387280246436,
            1.013067404854255,
            1.012368317979973,
            1.014938208673959,
            1.016292133436504,
            1.016299876905317,
            1.015722805077512,
            1.015781036395422,
            1.012964449250795,
            1.012586033733653,
            1.015759597940788,
            1.016663850758965,
            1.013529528380627,
            1.013484704516014,
            1.012682807983253,
            1.01575473995535,
            1.013548404141982,
            1.011642927298109,
            1.010163752457205,
            1.011376986373333,
            1.008478684014505,
            1.01076719680859,
            1.015337428415854,
            1.015222093430748,
            1.023956268877837,
            1.028750969689179,
            1.035269519245858,
            1.035513915058335,
            1.034682983634216,
            1.037199590276675,
            1.039170512167414,
            1.038969949195272,
            1.037666290030598,
            1.033529751695997,
            1.036136934479846,
            1.036538339548753,
            1.035889109013614,
            1.031715740053231,
            1.034826945935777,
            1.03923308853556,
            1.040612673470696,
            1.040481808314478,
            1.038824638064001,
            1.036241011095529,
            1.032552613690202,
            1.028778610050099,
            1.027717730725616,
            1.030937712182736,
            1.032079168722811,
            1.032303113095378,
            1.037043383963788,
            1.036329573728378,
            1.041070768860438,
            1.044691677277668,
            1.046925332437581,
            1.043303410276278,
            1.035368352410736,
            1.031318587238934,
            1.030894315873749,
            1.032324088887647,
            1.031245980085998,
            1.026437235240361,
            1.025165781233839,
            1.026090252860105,
            1.025619000910384,
            1.023136034669391,
            1.024981846935682,
            1.022880602180046,
            1.019237955591011,
            1.015352307410109,
            1.012840365887581,
            1.012029243045865,
            1.012048307703205,
            1.010717415706258,
            1.008955507148084,
            1.008526161719548,
            1.00736171581792,
            1.004414193550127,
            1.003714686103693,
            1.002636343417854,
            0.996542535146892,
            0.989320622076918,
            0.986208874181558,
            0.984555270670563,
            0.980400058449194,
            0.974958856502573,
            0.968990582348322,
            0.970791932672292,
            0.964051672980846,
            0.957615340129608,
            0.953689284418134,
            0.949679218256916,
            0.943811871136564,
            0.938302185671799,
            0.937563023618026,
            0.935489449476821,
            0.935744294069269,
            0.932183689402094,
            0.930796766845063,
            0.931397940566161,
            0.92987773638335,
            0.927925840245067,
            0.929943488921957,
            0.932096814886774,
            0.928973491882601,
            0.928450238737925,
            0.921456076991707,
            0.912615508442264,
            0.898123536472606,
            0.883985980701241,
            0.863580401992552,
        ];
        let exemplar_mu_prime: Array1<f64> = array![
            0.999513861245046,
            0.967558564020585,
            0.74956916752738,
            0.728865728310883,
            0.72574195312705,
            1.005454419001063,
            1.006228340417498,
            1.007127711181864,
            1.008063868579998,
            1.009124145319178,
            1.009705676547208,
            1.010620931293127,
            1.011512694879991,
            1.012341331325565,
            1.013042495162817,
            1.013535654534111,
            1.014857354520209,
            1.015975974770221,
            1.016773817341982,
            1.017875610862462,
            1.018633539909264,
            1.019972904617374,
            1.020600776726502,
            1.02114764060472,
            1.022417485808281,
            1.023245545877594,
            1.023855478286165,
            1.024656482008552,
            1.025277905406186,
            1.025964147436222,
            1.026238473790198,
            1.027044786706042,
            1.028174160266387,
            1.028512213512235,
            1.029019857482313,
            1.029924052662622,
            1.030461598120221,
            1.030950614892938,
            1.031063938818807,
            1.031256615013425,
            1.032280752680531,
            1.033501884617801,
            1.033533972557256,
            1.032875628413542,
            1.032063531784998,
            1.032655211979587,
            1.032927840063433,
            1.03380671850628,
            1.032544591839065,
            1.03148868860227,
            1.032233101050519,
            1.034175025708812,
            1.034631453062808,
            1.033828967258967,
            1.033729079725963,
            1.034974066229826,
            1.034003726202058,
            1.033577861713763,
            1.032975070988221,
            1.03324653356771,
            1.034272185260723,
            1.034045221752118,
            1.032825395052188,
            1.033383532957392,
            1.034966395806474,
            1.036595365344987,
            1.036298796812527,
            1.035078406724978,
            1.033500955272553,
            1.033831506188649,
            1.033868732043101,
            1.033419331372768,
            1.031578597115649,
            1.031295760834975,
            1.033080183262813,
            1.035489851447274,
            1.035399188185708,
            1.034873929806235,
            1.034640488375393,
            1.035540272757286,
            1.035922289828786,
            1.034000929499806,
            1.032697988255067,
            1.031134574421041,
            1.032881425032288,
            1.032995098912292,
            1.03231886646807,
            1.034924622530812,
            1.037482751329606,
            1.039482001151578,
            1.040843422567221,
            1.038198223131203,
            1.037255895750936,
            1.037015944391323,
            1.039136913615756,
            1.038763778799785,
            1.038803071823406,
            1.040160264844581,
            1.04188533281607,
            1.043297270615782,
            1.042252086300856,
            1.042343012555476,
            1.04114142410376,
            1.038715814495035,
            1.040153015640629,
            1.041122099369821,
            1.041009790168563,
            1.04116367936102,
            1.04260451844155,
            1.044655670519184,
            1.045737428547007,
            1.044501942630465,
            1.044666458670755,
            1.042449709053656,
            1.043806680794975,
            1.040066332331092,
            1.03967350514247,
            1.037820837246265,
            1.037239001165811,
            1.037517650429221,
            1.031421512425222,
            1.029236275996933,
            1.029775240923104,
            1.027513258019199,
            1.031704567743896,
            1.030816032868093,
            1.029258622780576,
            1.026099951162651,
            1.021437732795694,
            1.020670072694144,
            1.025413325606894,
            1.026229825472253,
            1.028306603814805,
            1.034481094349484,
            1.039673279243756,
            1.044656934257898,
            1.04897544023222,
            1.052310935385496,
            1.051819360485423,
            1.049439664962193,
            1.046724327200109,
            1.044316480498577,
            1.039699705753889,
            1.041030565009095,
            1.045497247491123,
            1.049463321921996,
            1.052187395168293,
            1.05633917639916,
            1.063850774577815,
            1.069970191972416,
            1.072708191139316,
            1.069302176705327,
            1.061597903828138,
            1.055619029410076,
            1.049840043897954,
            1.044769684226801,
            1.04274729941471,
            1.048093126913018,
            1.049853162635761,
            1.0560616697891,
            1.052928492327918,
            1.057578097037095,
            1.062733358498047,
            1.064251858083066,
            1.055439075543997,
            1.049065259382641,
            1.049279124819313,
            1.047307644370756,
            1.046195695462444,
            1.04977711878309,
            1.050763902368039,
            1.053622590223363,
            1.053504566957641,
            1.05646928307185,
            1.063109767300985,
            1.061495515033982,
            1.053295146902848,
            1.047776202023021,
            1.044757398129665,
            1.042958530910127,
            1.03863948792039,
            1.03600383244007,
            1.039955412719496,
            1.039919505125238,
            1.035573613326898,
            1.034952455763765,
            1.037645446998059,
            1.036874186512053,
            1.030409836230587,
            0.805747757691069,
            0.11749497550635,
            -0.088006786919662,
            -0.177190051934908,
            -0.329632779178878,
            -0.4644494952375,
            -0.561285156244805,
            -0.550780406563594,
            -0.638247980342613,
            -0.706375346833058,
            -0.74990678415303,
            -0.784286874159083,
            -0.82698082726274,
            -0.858871359861229,
            -0.864493212212537,
            -0.87490698407293,
            -0.870502980965857,
            -0.888011690517074,
            -0.900774158096832,
            -0.901808853377789,
            -0.909466262189392,
            -0.914601099583661,
            -0.906789253841854,
            -0.892580872077899,
            -0.895949257029328,
            -0.90269261481202,
            -0.933968475691396,
            -0.957036304098969,
            -0.98646613110374,
            -1.008738574532424,
            -1.042603601102657,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(&exemplar_mu, &calc.mu(), margin, "mu(3)");
        comp_array_f64(&exemplar_mu_prime, &calc.mu_prime(), margin, "mu_prime(3)");

        // test_3.s2p
        let exemplar_mu: Array1<f64> = array![
            0.999528984364959,
            0.999762181064495,
            0.999145489050567,
            0.88829279395353,
            1.002473311191691,
            1.001796741627201,
            1.00265831925158,
            1.002574486306286,
            1.00351683027847,
            1.003738024002067,
            1.003448673071998,
            1.003936565396111,
            1.003980361352763,
            1.004357646908905,
            1.004769791404231,
            1.004988644652032,
            1.005101750782874,
            1.005836668918461,
            1.006045542146288,
            1.006611603461428,
            1.007435243087128,
            1.007543051860377,
            1.008226742368407,
            1.007747858249009,
            1.007572243178434,
            1.008059637219768,
            1.008868458812013,
            1.009448379220154,
            1.010101367769093,
            1.010233454090505,
            1.010418952006637,
            1.011000192021531,
            1.011664657201274,
            1.012640168866786,
            1.012472060278003,
            1.012625127625917,
            1.012946283305764,
            1.013035324317147,
            1.013336750632542,
            1.013558506977115,
            1.013502801700956,
            1.013431828699128,
            1.013948937181776,
            1.014237623835311,
            1.013932254501188,
            1.014242638082677,
            1.014995720073716,
            1.017724786772058,
            1.018128614983891,
            1.01900954703803,
            1.019013905110993,
            1.018113277022355,
            1.017993877057357,
            1.019426677604203,
            1.021197312591404,
            1.021778862608526,
            1.019698473045427,
            1.01926384680949,
            1.021977855638653,
            1.023532559528251,
            1.02468118244718,
            1.023597706784121,
            1.023681662427586,
            1.023594319937831,
            1.025261778642814,
            1.025405716647787,
            1.025597819002918,
            1.024258640452326,
            1.025786207307144,
            1.025524583076597,
            1.025035063091029,
            1.024608057958407,
            1.025852352144526,
            1.024902815996521,
            1.023731703868449,
            1.024101509164294,
            1.023348021071617,
            1.025037124526025,
            1.025338692173392,
            1.022003987831836,
            1.019242833622976,
            1.01605538063258,
            1.0153959660974,
            1.017498574986551,
            1.018885926182374,
            1.017530371241366,
            1.016530025302907,
            1.016768535218064,
            1.017998361283112,
            1.020072287121264,
            1.019297094170437,
            1.020412029057682,
            1.019034802386793,
            1.019205419001391,
            1.018671048865379,
            1.019330432489407,
            1.017637820140676,
            1.022216677999239,
            1.021705797335569,
            1.025190387243317,
            1.016957624361492,
            1.019097942240908,
            1.021666250005322,
            1.024152153112026,
            1.026686499521207,
            1.026845593856781,
            1.025389434131405,
            1.02453279430916,
            1.023355522298842,
            1.024008452547939,
            1.021465955932115,
            1.017377769644077,
            1.015307352484093,
            1.016354717716911,
            1.014681581371351,
            1.016948439078886,
            1.019677756285704,
            1.022728729046904,
            1.023168579571673,
            1.017199315659763,
            1.013440113743299,
            1.011526782108022,
            1.010100833560144,
            1.009024936629574,
            1.008178695275263,
            1.009252007298022,
            1.01169515754927,
            1.022536888525096,
            1.025801973320925,
            1.021794643688982,
            1.015653321137733,
            1.020325746189997,
            1.031719672009427,
            1.034589599249255,
            1.022769756810092,
            1.026279730291789,
            1.034061159078448,
            1.041219485139242,
            1.044950901973057,
            1.049559267623192,
            1.052692530372447,
            1.05234086155338,
            1.054045758764346,
            1.050755095743111,
            1.049632483444648,
            1.048149413297286,
            1.044272502129536,
            1.040154616736789,
            1.036326725924378,
            1.034580397763072,
            1.032615563984588,
            1.030501289796419,
            1.030478695466627,
            1.029185563804762,
            1.028355548325157,
            1.029921206896223,
            1.033959812883475,
            1.036301617254152,
            1.038643361364163,
            1.037683707101625,
            1.037149930349603,
            1.03792119893674,
            1.041523426560752,
            1.040074967542835,
            1.036043401653541,
            1.036350148326241,
            1.035034588612304,
            1.033777864063204,
            1.033080819435642,
            1.027574162784245,
            1.023640359569429,
            1.015086757478202,
            0.99529206397026,
            1.010511573472717,
            1.020615916758437,
            1.022009542897345,
            1.017624044583531,
            1.002683664794222,
            0.919136326723154,
            0.741130163170881,
            0.751828567379112,
            0.710706557672304,
            0.693869330473409,
            0.75251867274655,
            0.847195135491049,
            0.83366802679309,
            0.839169663809495,
            0.927621568473872,
            0.980156204598675,
            0.97789927979733,
            0.801273997491488,
            0.76749330492844,
            0.629582902115378,
            0.595011692303855,
            0.616508001447077,
            0.560071167184073,
            0.642609705075093,
            0.536530862495005,
            0.660985370343645,
            0.804934411212802,
            0.90247156936459,
            0.757088065450019,
            0.639353899519848,
            0.573669752370492,
            0.514853460210996,
            0.473585450724909,
            0.450131866561873,
            0.631471674316544,
            0.846693239436669,
            0.934808201015689,
            0.954892163188615,
            0.950939971177055,
            0.937880496826707,
            0.918496543635446,
            0.902061593086212,
            0.889176345065333,
            0.877840159774429,
            0.868922150457399,
            0.840755523293482,
            0.771720977009805,
        ];
        let exemplar_mu_prime: Array1<f64> = array![
            -0.33045894975032,
            0.593080915967793,
            0.438844767029363,
            0.527195248768602,
            1.002196502621164,
            1.003226631134393,
            1.003018729747956,
            1.003735154810106,
            1.0038815302831,
            1.004650976299392,
            1.005352751794814,
            1.00557397252608,
            1.005940989934005,
            1.006410799695239,
            1.006759243515396,
            1.006538395917264,
            1.007545483666778,
            1.008237311336017,
            1.00853547369791,
            1.009249947622244,
            1.009653084254639,
            1.010839387679318,
            1.011092990899932,
            1.012343585718866,
            1.012778122010956,
            1.013112561334996,
            1.013775787747125,
            1.01421303364344,
            1.014192406807842,
            1.014449792205505,
            1.015121025756389,
            1.015768445599155,
            1.01616728607233,
            1.016663089477774,
            1.016909828253521,
            1.01787983095621,
            1.01796627653123,
            1.017786227574923,
            1.018198553734736,
            1.018628446513179,
            1.018648277606169,
            1.018108621167277,
            1.017877259124016,
            1.018344005682595,
            1.018721542248935,
            1.018816924891986,
            1.018362016258228,
            1.019659996311078,
            1.020826294547401,
            1.022136487540635,
            1.021972560131891,
            1.021101097380913,
            1.020703865042928,
            1.022209797277023,
            1.023820525313387,
            1.023833356494463,
            1.023948826839559,
            1.024415273271301,
            1.025355850496352,
            1.024799566912542,
            1.02497652593545,
            1.024567393372368,
            1.025123894419323,
            1.026726614695802,
            1.027588671588654,
            1.026618827852014,
            1.026194748105923,
            1.02561071020031,
            1.025115170530792,
            1.023425796832106,
            1.021719080851613,
            1.021645504979026,
            1.023247435365595,
            1.023225304279114,
            1.020253933119237,
            1.018644519198912,
            1.018210049484029,
            1.021532082284959,
            1.021948053828476,
            1.020193352463714,
            1.017810770074989,
            1.015240150599011,
            1.017824716272332,
            1.019582832711856,
            1.019533027433083,
            1.019595257020359,
            1.020346790248718,
            1.024225834063017,
            1.023832382411923,
            1.026884976359778,
            1.026406624359488,
            1.029149159874339,
            1.028604782343948,
            1.030040505640839,
            1.028089574354705,
            1.028132835587193,
            1.029953363468186,
            1.030629785331973,
            1.030294353726911,
            1.028506769853607,
            1.029009209538814,
            1.03150115901446,
            1.033907072312294,
            1.032989058820716,
            1.032868330816892,
            1.03145025086753,
            1.031167043083174,
            1.033094794192268,
            1.032565246164226,
            1.030352758638647,
            1.030589383859312,
            1.031352960554975,
            1.033747888462782,
            1.033837799116049,
            1.034129596314754,
            1.035437399395063,
            1.036253442054141,
            1.037015418053927,
            1.032391856205759,
            1.03418446347414,
            1.029856511173064,
            1.026029956986333,
            1.02289242757197,
            1.023326969104214,
            1.02453562218691,
            1.019993998088054,
            1.022366124908934,
            1.033253782233876,
            1.036323958837755,
            1.029420605233849,
            1.022590744800133,
            1.025073331023639,
            1.031626902241026,
            1.036813387953016,
            1.026942304160723,
            1.029815636509692,
            1.032089646010751,
            1.033600372611352,
            1.033367768101646,
            1.032191783450497,
            1.033001218499755,
            1.034655901059811,
            1.037390951338054,
            1.036065578010946,
            1.032994171201195,
            1.032349815281675,
            1.031571266790825,
            1.033705783874042,
            1.035887841508363,
            1.038212312290906,
            1.038971735563817,
            1.040513202119944,
            1.0387457276339,
            1.03794871109255,
            1.037954242313629,
            1.038429134879143,
            1.036149691117994,
            1.033063779050629,
            1.033072580909652,
            1.029962560113974,
            1.028241362252937,
            1.027997440578273,
            1.025223448185417,
            1.02223692743489,
            1.018650575111342,
            1.014725754518213,
            1.012349344449222,
            1.01112323380688,
            1.007848868710173,
            1.004021464627897,
            1.00337015859438,
            1.002089572096603,
            0.999545537657934,
            1.001405591541721,
            1.004173784401128,
            1.005234797455673,
            1.003369071063844,
            1.00028530413717,
            0.995873800136236,
            0.990886361318866,
            0.990955838332254,
            0.990521978484067,
            0.989801033188096,
            0.990915683235312,
            0.993428212277952,
            0.992545431497394,
            0.991206488219399,
            0.99328020005537,
            0.997288543205053,
            0.997354112091465,
            0.989103368724425,
            0.986361239066992,
            0.98286739600647,
            0.980508951702593,
            0.982610432763415,
            0.975397530085581,
            0.974559570625121,
            0.969617888547517,
            0.967237922998867,
            0.977100984352601,
            0.976300300590189,
            0.969346437992307,
            0.927533291860737,
            0.824776761119865,
            0.586075882160252,
            0.534695043765738,
            0.305476794124825,
            0.158462370699285,
            0.047183561077737,
            -0.051210960003111,
            -0.090740047456459,
            -0.228956090398148,
            -0.419715369713377,
            -0.588330741882935,
            -0.662881309493186,
            -0.674225344452813,
            -0.664435870614598,
            -0.638745955770156,
            -0.639868274075586,
            -0.640604539606232,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::Db,
            vec![
                vec![
                    (-0.007, -2.703),
                    (-33.308, 87.205),
                    (-33.310, 87.223),
                    (0.002, -2.522),
                ],
                vec![
                    (-0.014, -5.380),
                    (-27.320, 84.747),
                    (-27.319, 84.763),
                    (-0.006, -5.048),
                ],
                vec![
                    (-0.020, -8.054),
                    (-23.824, 82.282),
                    (-23.822, 82.309),
                    (-0.011, -7.535),
                ],
                vec![
                    (-0.030, -10.721),
                    (-21.356, 79.850),
                    (-21.355, 79.863),
                    (-0.022, -10.008),
                ],
                vec![
                    (-0.070, -12.934),
                    (-19.767, 76.759),
                    (-19.766, 76.774),
                    (-0.073, -12.097),
                ],
                vec![
                    (-0.102, -15.528),
                    (-18.235, 74.354),
                    (-18.233, 74.341),
                    (-0.086, -14.420),
                ],
                vec![
                    (-0.122, -18.012),
                    (-16.968, 71.927),
                    (-16.964, 71.952),
                    (-0.118, -16.765),
                ],
                vec![
                    (-0.156, -20.492),
                    (-15.876, 69.541),
                    (-15.872, 69.531),
                    (-0.143, -19.112),
                ],
                vec![
                    (-0.185, -23.013),
                    (-14.932, 67.149),
                    (-14.928, 67.141),
                    (-0.181, -21.370),
                ],
                vec![
                    (-0.224, -25.457),
                    (-14.103, 64.781),
                    (-14.100, 64.755),
                    (-0.214, -23.659),
                ],
                vec![
                    (-0.265, -27.878),
                    (-13.365, 62.489),
                    (-13.360, 62.451),
                    (-0.244, -25.890),
                ],
                vec![
                    (-0.302, -30.240),
                    (-12.705, 60.203),
                    (-12.703, 60.163),
                    (-0.284, -28.109),
                ],
                vec![
                    (-0.344, -32.603),
                    (-12.109, 57.908),
                    (-12.107, 57.880),
                    (-0.322, -30.298),
                ],
                vec![
                    (-0.387, -34.903),
                    (-11.574, 55.657),
                    (-11.572, 55.620),
                    (-0.364, -32.496),
                ],
                vec![
                    (-0.429, -37.197),
                    (-11.087, 53.446),
                    (-11.086, 53.458),
                    (-0.407, -34.613),
                ],
                vec![
                    (-0.463, -39.356),
                    (-10.675, 51.542),
                    (-10.671, 51.469),
                    (-0.446, -36.438),
                ],
                vec![
                    (-0.515, -41.554),
                    (-10.268, 49.344),
                    (-10.270, 49.333),
                    (-0.488, -38.514),
                ],
                vec![
                    (-0.562, -43.746),
                    (-9.897, 47.250),
                    (-9.897, 47.193),
                    (-0.536, -40.558),
                ],
                vec![
                    (-0.610, -45.867),
                    (-9.555, 45.112),
                    (-9.555, 45.063),
                    (-0.582, -42.571),
                ],
                vec![
                    (-0.660, -47.981),
                    (-9.247, 43.031),
                    (-9.245, 42.983),
                    (-0.630, -44.513),
                ],
                vec![
                    (-0.704, -50.043),
                    (-8.963, 41.024),
                    (-8.964, 40.994),
                    (-0.679, -46.423),
                ],
                vec![
                    (-0.757, -52.064),
                    (-8.703, 39.062),
                    (-8.702, 39.007),
                    (-0.720, -48.330),
                ],
                vec![
                    (-0.799, -54.087),
                    (-8.460, 37.129),
                    (-8.460, 37.075),
                    (-0.767, -50.166),
                ],
                vec![
                    (-0.851, -56.013),
                    (-8.237, 35.258),
                    (-8.238, 35.248),
                    (-0.800, -51.986),
                ],
                vec![
                    (-0.898, -57.881),
                    (-8.025, 33.392),
                    (-8.025, 33.331),
                    (-0.839, -53.849),
                ],
                vec![
                    (-0.939, -59.713),
                    (-7.832, 31.566),
                    (-7.832, 31.496),
                    (-0.882, -55.676),
                ],
                vec![
                    (-0.982, -61.565),
                    (-7.656, 29.737),
                    (-7.658, 29.701),
                    (-0.927, -57.406),
                ],
                vec![
                    (-1.021, -63.357),
                    (-7.495, 27.999),
                    (-7.495, 27.920),
                    (-0.968, -59.135),
                ],
                vec![
                    (-1.056, -65.122),
                    (-7.345, 26.233),
                    (-7.349, 26.172),
                    (-1.010, -60.814),
                ],
                vec![
                    (-1.093, -66.877),
                    (-7.211, 24.513),
                    (-7.210, 24.431),
                    (-1.045, -62.470),
                ],
                vec![
                    (-1.130, -68.611),
                    (-7.084, 22.839),
                    (-7.085, 22.760),
                    (-1.077, -64.114),
                ],
                vec![
                    (-1.166, -70.296),
                    (-6.964, 21.187),
                    (-6.963, 21.105),
                    (-1.113, -65.762),
                ],
                vec![
                    (-1.199, -71.933),
                    (-6.856, 19.554),
                    (-6.855, 19.467),
                    (-1.149, -67.361),
                ],
                vec![
                    (-1.228, -73.544),
                    (-6.761, 17.969),
                    (-6.761, 17.907),
                    (-1.184, -68.913),
                ],
                vec![
                    (-1.258, -75.146),
                    (-6.668, 16.427),
                    (-6.669, 16.307),
                    (-1.209, -70.427),
                ],
                vec![
                    (-1.293, -76.757),
                    (-6.583, 14.877),
                    (-6.587, 14.748),
                    (-1.235, -71.896),
                ],
                vec![
                    (-1.316, -78.293),
                    (-6.503, 13.359),
                    (-6.507, 13.277),
                    (-1.261, -73.391),
                ],
                vec![
                    (-1.339, -79.774),
                    (-6.431, 11.869),
                    (-6.434, 11.757),
                    (-1.286, -74.841),
                ],
                vec![
                    (-1.362, -81.225),
                    (-6.369, 10.428),
                    (-6.372, 10.325),
                    (-1.308, -76.268),
                ],
                vec![
                    (-1.387, -82.666),
                    (-6.312, 8.931),
                    (-6.317, 8.864),
                    (-1.330, -77.697),
                ],
                vec![
                    (-1.405, -84.120),
                    (-6.257, 7.509),
                    (-6.262, 7.430),
                    (-1.347, -79.106),
                ],
                vec![
                    (-1.419, -85.518),
                    (-6.208, 6.101),
                    (-6.213, 5.976),
                    (-1.365, -80.482),
                ],
                vec![
                    (-1.429, -86.867),
                    (-6.165, 4.717),
                    (-6.169, 4.653),
                    (-1.384, -81.874),
                ],
                vec![
                    (-1.447, -88.250),
                    (-6.127, 3.340),
                    (-6.132, 3.226),
                    (-1.400, -83.269),
                ],
                vec![
                    (-1.465, -89.627),
                    (-6.089, 1.959),
                    (-6.092, 1.849),
                    (-1.410, -84.648),
                ],
                vec![
                    (-1.477, -90.985),
                    (-6.063, 0.594),
                    (-6.066, 0.471),
                    (-1.424, -85.964),
                ],
                vec![
                    (-1.481, -92.320),
                    (-6.052, -0.794),
                    (-6.055, -0.900),
                    (-1.441, -87.215),
                ],
                vec![
                    (-1.488, -93.626),
                    (-6.056, -2.034),
                    (-6.060, -2.165),
                    (-1.466, -88.427),
                ],
                vec![
                    (-1.504, -94.858),
                    (-6.038, -3.236),
                    (-6.039, -3.361),
                    (-1.474, -89.665),
                ],
                vec![
                    (-1.520, -96.087),
                    (-6.023, -4.482),
                    (-6.024, -4.590),
                    (-1.486, -90.966),
                ],
                vec![
                    (-1.522, -97.307),
                    (-6.009, -5.661),
                    (-6.008, -5.786),
                    (-1.490, -92.174),
                ],
                vec![
                    (-1.518, -98.492),
                    (-6.002, -6.874),
                    (-6.003, -6.991),
                    (-1.485, -93.349),
                ],
                vec![
                    (-1.517, -99.685),
                    (-5.991, -8.055),
                    (-5.995, -8.196),
                    (-1.487, -94.556),
                ],
                vec![
                    (-1.539, -100.827),
                    (-5.988, -9.359),
                    (-5.989, -9.478),
                    (-1.508, -95.800),
                ],
                vec![
                    (-1.549, -101.915),
                    (-6.011, -10.524),
                    (-6.016, -10.644),
                    (-1.520, -96.906),
                ],
                vec![
                    (-1.535, -103.121),
                    (-6.030, -11.560),
                    (-6.035, -11.671),
                    (-1.513, -97.983),
                ],
                vec![
                    (-1.527, -104.305),
                    (-6.029, -12.498),
                    (-6.032, -12.599),
                    (-1.483, -99.050),
                ],
                vec![
                    (-1.536, -105.323),
                    (-6.025, -13.680),
                    (-6.030, -13.823),
                    (-1.482, -100.338),
                ],
                vec![
                    (-1.543, -106.386),
                    (-6.035, -14.827),
                    (-6.036, -14.953),
                    (-1.508, -101.481),
                ],
                vec![
                    (-1.532, -107.502),
                    (-6.043, -15.925),
                    (-6.049, -16.028),
                    (-1.519, -102.578),
                ],
                vec![
                    (-1.523, -108.630),
                    (-6.073, -17.002),
                    (-6.074, -17.148),
                    (-1.520, -103.680),
                ],
                vec![
                    (-1.517, -109.630),
                    (-6.085, -18.045),
                    (-6.088, -18.163),
                    (-1.507, -104.645),
                ],
                vec![
                    (-1.517, -110.526),
                    (-6.108, -19.017),
                    (-6.115, -19.129),
                    (-1.502, -105.568),
                ],
                vec![
                    (-1.523, -111.551),
                    (-6.133, -19.965),
                    (-6.132, -20.109),
                    (-1.491, -106.576),
                ],
                vec![
                    (-1.520, -112.602),
                    (-6.146, -20.916),
                    (-6.148, -21.090),
                    (-1.497, -107.712),
                ],
                vec![
                    (-1.505, -113.579),
                    (-6.160, -21.929),
                    (-6.164, -22.133),
                    (-1.493, -108.803),
                ],
                vec![
                    (-1.496, -114.428),
                    (-6.187, -22.916),
                    (-6.190, -23.068),
                    (-1.490, -109.691),
                ],
                vec![
                    (-1.492, -115.261),
                    (-6.202, -23.910),
                    (-6.201, -24.079),
                    (-1.478, -110.630),
                ],
                vec![
                    (-1.479, -116.159),
                    (-6.233, -24.888),
                    (-6.235, -25.011),
                    (-1.486, -111.506),
                ],
                vec![
                    (-1.450, -117.227),
                    (-6.269, -25.884),
                    (-6.268, -25.983),
                    (-1.472, -112.436),
                ],
                vec![
                    (-1.415, -118.310),
                    (-6.304, -26.742),
                    (-6.300, -26.933),
                    (-1.449, -113.459),
                ],
                vec![
                    (-1.406, -119.173),
                    (-6.318, -27.627),
                    (-6.319, -27.787),
                    (-1.436, -114.468),
                ],
                vec![
                    (-1.416, -120.155),
                    (-6.333, -28.537),
                    (-6.336, -28.705),
                    (-1.442, -115.341),
                ],
                vec![
                    (-1.412, -121.119),
                    (-6.354, -29.574),
                    (-6.357, -29.733),
                    (-1.429, -116.311),
                ],
                vec![
                    (-1.375, -122.107),
                    (-6.378, -30.598),
                    (-6.386, -30.749),
                    (-1.411, -117.337),
                ],
                vec![
                    (-1.348, -122.983),
                    (-6.417, -31.525),
                    (-6.420, -31.689),
                    (-1.405, -118.305),
                ],
                vec![
                    (-1.340, -123.747),
                    (-6.459, -32.495),
                    (-6.472, -32.682),
                    (-1.397, -119.061),
                ],
                vec![
                    (-1.351, -124.603),
                    (-6.527, -33.267),
                    (-6.525, -33.467),
                    (-1.388, -119.865),
                ],
                vec![
                    (-1.339, -125.547),
                    (-6.566, -34.045),
                    (-6.567, -34.205),
                    (-1.374, -120.594),
                ],
                vec![
                    (-1.313, -126.503),
                    (-6.593, -34.907),
                    (-6.594, -35.053),
                    (-1.332, -121.429),
                ],
                vec![
                    (-1.287, -127.265),
                    (-6.622, -35.784),
                    (-6.631, -36.002),
                    (-1.303, -122.280),
                ],
                vec![
                    (-1.265, -127.923),
                    (-6.640, -36.573),
                    (-6.655, -36.827),
                    (-1.275, -123.026),
                ],
                vec![
                    (-1.276, -128.868),
                    (-6.676, -37.454),
                    (-6.683, -37.685),
                    (-1.248, -124.052),
                ],
                vec![
                    (-1.276, -129.931),
                    (-6.720, -38.438),
                    (-6.734, -38.649),
                    (-1.253, -124.964),
                ],
                vec![
                    (-1.258, -130.950),
                    (-6.775, -39.360),
                    (-6.782, -39.607),
                    (-1.251, -125.870),
                ],
                vec![
                    (-1.250, -131.674),
                    (-6.823, -40.202),
                    (-6.838, -40.386),
                    (-1.227, -126.610),
                ],
                vec![
                    (-1.245, -132.330),
                    (-6.872, -40.937),
                    (-6.868, -41.097),
                    (-1.203, -127.499),
                ],
                vec![
                    (-1.264, -133.188),
                    (-6.919, -41.663),
                    (-6.921, -41.828),
                    (-1.186, -128.275),
                ],
                vec![
                    (-1.241, -133.962),
                    (-6.979, -42.437),
                    (-6.979, -42.740),
                    (-1.181, -129.251),
                ],
                vec![
                    (-1.244, -134.873),
                    (-7.035, -43.207),
                    (-7.046, -43.432),
                    (-1.178, -130.173),
                ],
                vec![
                    (-1.233, -135.444),
                    (-7.079, -43.963),
                    (-7.079, -44.149),
                    (-1.163, -130.927),
                ],
                vec![
                    (-1.204, -137.506),
                    (-7.295, -45.675),
                    (-7.287, -45.895),
                    (-1.120, -132.380),
                ],
                vec![
                    (-1.196, -138.266),
                    (-7.331, -46.540),
                    (-7.340, -46.723),
                    (-1.101, -132.999),
                ],
                vec![
                    (-1.184, -139.237),
                    (-7.406, -47.224),
                    (-7.392, -47.498),
                    (-1.081, -133.941),
                ],
                vec![
                    (-1.156, -139.806),
                    (-7.439, -47.942),
                    (-7.442, -48.261),
                    (-1.066, -134.851),
                ],
                vec![
                    (-1.141, -140.499),
                    (-7.512, -48.823),
                    (-7.507, -49.026),
                    (-1.057, -135.763),
                ],
                vec![
                    (-1.151, -141.070),
                    (-7.544, -49.468),
                    (-7.552, -49.677),
                    (-1.033, -136.450),
                ],
                vec![
                    (-1.139, -141.988),
                    (-7.619, -50.241),
                    (-7.621, -50.511),
                    (-1.060, -137.163),
                ],
                vec![
                    (-1.126, -142.624),
                    (-7.678, -51.043),
                    (-7.661, -51.286),
                    (-1.045, -138.047),
                ],
                vec![
                    (-1.090, -143.317),
                    (-7.754, -51.827),
                    (-7.754, -51.818),
                    (-1.059, -138.695),
                ],
                vec![
                    (-1.090, -143.804),
                    (-7.812, -52.472),
                    (-7.801, -52.722),
                    (-0.973, -139.574),
                ],
                vec![
                    (-1.082, -144.444),
                    (-7.886, -53.018),
                    (-7.910, -53.153),
                    (-0.968, -140.498),
                ],
                vec![
                    (-1.091, -145.084),
                    (-7.937, -53.366),
                    (-7.948, -53.552),
                    (-0.981, -140.970),
                ],
                vec![
                    (-1.073, -145.770),
                    (-7.979, -53.983),
                    (-7.999, -54.149),
                    (-0.994, -141.596),
                ],
                vec![
                    (-1.063, -146.276),
                    (-8.043, -54.559),
                    (-8.026, -54.782),
                    (-1.008, -142.399),
                ],
                vec![
                    (-1.042, -146.836),
                    (-8.078, -55.281),
                    (-8.074, -55.531),
                    (-1.001, -143.284),
                ],
                vec![
                    (-1.033, -147.673),
                    (-8.132, -56.080),
                    (-8.110, -56.310),
                    (-0.981, -143.668),
                ],
                vec![
                    (-1.042, -148.451),
                    (-8.160, -56.728),
                    (-8.174, -56.898),
                    (-0.965, -144.102),
                ],
                vec![
                    (-1.024, -148.900),
                    (-8.249, -57.285),
                    (-8.226, -57.521),
                    (-0.941, -144.796),
                ],
                vec![
                    (-0.995, -149.415),
                    (-8.286, -57.910),
                    (-8.272, -58.101),
                    (-0.938, -145.596),
                ],
                vec![
                    (-0.989, -150.029),
                    (-8.310, -58.664),
                    (-8.327, -58.674),
                    (-0.907, -146.361),
                ],
                vec![
                    (-0.991, -150.499),
                    (-8.371, -59.225),
                    (-8.362, -59.245),
                    (-0.863, -146.713),
                ],
                vec![
                    (-1.005, -150.999),
                    (-8.404, -59.719),
                    (-8.422, -59.990),
                    (-0.836, -147.464),
                ],
                vec![
                    (-0.991, -151.557),
                    (-8.484, -60.340),
                    (-8.449, -60.444),
                    (-0.834, -148.351),
                ],
                vec![
                    (-0.992, -152.156),
                    (-8.521, -61.244),
                    (-8.512, -61.261),
                    (-0.813, -149.016),
                ],
                vec![
                    (-0.994, -152.469),
                    (-8.563, -61.662),
                    (-8.583, -62.009),
                    (-0.825, -149.822),
                ],
                vec![
                    (-0.996, -152.775),
                    (-8.620, -62.219),
                    (-8.625, -62.491),
                    (-0.844, -150.308),
                ],
                vec![
                    (-0.998, -153.136),
                    (-8.656, -62.852),
                    (-8.703, -63.102),
                    (-0.866, -150.828),
                ],
                vec![
                    (-0.956, -153.739),
                    (-8.707, -63.513),
                    (-8.698, -63.648),
                    (-0.869, -151.174),
                ],
                vec![
                    (-0.964, -154.217),
                    (-8.762, -64.131),
                    (-8.749, -64.117),
                    (-0.803, -151.983),
                ],
                vec![
                    (-0.949, -154.618),
                    (-8.809, -64.854),
                    (-8.772, -65.287),
                    (-0.774, -152.579),
                ],
                vec![
                    (-0.920, -155.032),
                    (-8.834, -65.647),
                    (-8.851, -65.891),
                    (-0.753, -153.294),
                ],
                vec![
                    (-0.903, -155.591),
                    (-8.860, -66.464),
                    (-8.894, -66.700),
                    (-0.741, -154.031),
                ],
                vec![
                    (-0.907, -156.265),
                    (-8.938, -67.399),
                    (-8.959, -67.342),
                    (-0.720, -154.847),
                ],
                vec![
                    (-0.896, -156.802),
                    (-9.031, -67.658),
                    (-9.033, -68.226),
                    (-0.692, -155.857),
                ],
                vec![
                    (-0.856, -157.201),
                    (-9.084, -68.597),
                    (-9.143, -68.901),
                    (-0.706, -156.318),
                ],
                vec![
                    (-0.874, -157.546),
                    (-9.202, -69.240),
                    (-9.200, -69.721),
                    (-0.727, -156.732),
                ],
                vec![
                    (-0.922, -158.150),
                    (-9.258, -69.906),
                    (-9.279, -70.164),
                    (-0.807, -157.422),
                ],
                vec![
                    (-0.947, -158.704),
                    (-9.308, -70.926),
                    (-9.333, -70.844),
                    (-0.835, -158.104),
                ],
                vec![
                    (-0.895, -159.033),
                    (-9.413, -71.526),
                    (-9.401, -71.758),
                    (-0.804, -158.610),
                ],
                vec![
                    (-0.854, -159.318),
                    (-9.522, -72.082),
                    (-9.547, -72.352),
                    (-0.752, -158.691),
                ],
                vec![
                    (-0.838, -159.873),
                    (-9.683, -72.618),
                    (-9.692, -73.056),
                    (-0.775, -159.259),
                ],
                vec![
                    (-0.849, -160.766),
                    (-9.746, -73.258),
                    (-9.704, -73.203),
                    (-0.850, -160.019),
                ],
                vec![
                    (-0.846, -161.398),
                    (-9.808, -73.008),
                    (-9.869, -73.463),
                    (-0.825, -161.307),
                ],
                vec![
                    (-0.756, -161.361),
                    (-9.948, -73.728),
                    (-9.931, -73.058),
                    (-0.713, -161.524),
                ],
                vec![
                    (-0.761, -162.046),
                    (-10.012, -74.283),
                    (-9.980, -73.262),
                    (-0.727, -162.222),
                ],
                vec![
                    (-0.760, -162.772),
                    (-10.097, -74.685),
                    (-10.090, -73.887),
                    (-0.778, -163.039),
                ],
                vec![
                    (-0.759, -163.211),
                    (-10.197, -74.987),
                    (-10.190, -74.236),
                    (-0.827, -163.492),
                ],
                vec![
                    (-0.743, -163.534),
                    (-10.288, -75.174),
                    (-10.306, -74.376),
                    (-0.845, -163.829),
                ],
                vec![
                    (-0.718, -164.022),
                    (-10.406, -75.414),
                    (-10.419, -74.530),
                    (-0.869, -164.190),
                ],
                vec![
                    (-0.713, -164.578),
                    (-10.501, -75.436),
                    (-10.514, -74.649),
                    (-0.882, -164.538),
                ],
                vec![
                    (-0.721, -164.938),
                    (-10.563, -75.321),
                    (-10.556, -74.375),
                    (-0.872, -164.822),
                ],
                vec![
                    (-0.740, -165.270),
                    (-10.616, -75.215),
                    (-10.615, -74.492),
                    (-0.882, -164.933),
                ],
                vec![
                    (-0.727, -165.532),
                    (-10.636, -75.164),
                    (-10.632, -74.424),
                    (-0.853, -165.150),
                ],
                vec![
                    (-0.704, -166.050),
                    (-10.623, -75.065),
                    (-10.634, -74.288),
                    (-0.849, -165.342),
                ],
                vec![
                    (-0.701, -166.646),
                    (-10.619, -75.230),
                    (-10.623, -74.467),
                    (-0.840, -165.459),
                ],
                vec![
                    (-0.696, -167.209),
                    (-10.617, -75.602),
                    (-10.618, -74.787),
                    (-0.809, -165.838),
                ],
                vec![
                    (-0.716, -167.810),
                    (-10.631, -75.931),
                    (-10.637, -75.128),
                    (-0.774, -166.238),
                ],
                vec![
                    (-0.741, -168.274),
                    (-10.612, -76.177),
                    (-10.622, -75.507),
                    (-0.745, -166.740),
                ],
                vec![
                    (-0.764, -168.664),
                    (-10.584, -76.727),
                    (-10.596, -76.030),
                    (-0.731, -167.341),
                ],
                vec![
                    (-0.765, -168.885),
                    (-10.590, -77.366),
                    (-10.609, -76.537),
                    (-0.708, -167.767),
                ],
                vec![
                    (-0.772, -169.145),
                    (-10.629, -77.899),
                    (-10.640, -77.188),
                    (-0.683, -168.348),
                ],
                vec![
                    (-0.750, -169.444),
                    (-10.667, -78.526),
                    (-10.673, -77.889),
                    (-0.677, -169.053),
                ],
                vec![
                    (-0.736, -169.760),
                    (-10.715, -79.048),
                    (-10.721, -78.526),
                    (-0.659, -169.607),
                ],
                vec![
                    (-0.729, -169.988),
                    (-10.774, -79.624),
                    (-10.776, -79.055),
                    (-0.645, -170.279),
                ],
                vec![
                    (-0.726, -170.132),
                    (-10.837, -80.185),
                    (-10.837, -79.496),
                    (-0.652, -170.933),
                ],
                vec![
                    (-0.700, -170.491),
                    (-10.903, -80.593),
                    (-10.902, -79.850),
                    (-0.681, -171.309),
                ],
                vec![
                    (-0.666, -170.890),
                    (-10.965, -81.065),
                    (-10.973, -80.269),
                    (-0.694, -171.561),
                ],
                vec![
                    (-0.659, -171.335),
                    (-11.040, -81.499),
                    (-11.050, -80.629),
                    (-0.707, -171.762),
                ],
                vec![
                    (-0.626, -171.867),
                    (-11.113, -81.675),
                    (-11.129, -80.926),
                    (-0.693, -172.083),
                ],
                vec![
                    (-0.607, -172.055),
                    (-11.179, -81.751),
                    (-11.191, -80.954),
                    (-0.685, -172.412),
                ],
                vec![
                    (-0.604, -172.623),
                    (-11.229, -82.010),
                    (-11.237, -81.134),
                    (-0.692, -172.867),
                ],
                vec![
                    (-0.581, -173.185),
                    (-11.233, -82.302),
                    (-11.249, -81.467),
                    (-0.727, -173.284),
                ],
                vec![
                    (-0.555, -173.548),
                    (-11.270, -82.562),
                    (-11.284, -81.615),
                    (-0.719, -173.825),
                ],
                vec![
                    (-0.529, -174.035),
                    (-11.295, -82.750),
                    (-11.294, -81.725),
                    (-0.699, -174.420),
                ],
                vec![
                    (-0.493, -174.425),
                    (-11.329, -82.919),
                    (-11.331, -81.878),
                    (-0.714, -174.634),
                ],
                vec![
                    (-0.474, -174.909),
                    (-11.354, -83.028),
                    (-11.362, -82.069),
                    (-0.722, -174.945),
                ],
                vec![
                    (-0.463, -175.232),
                    (-11.350, -83.348),
                    (-11.367, -82.421),
                    (-0.717, -175.277),
                ],
                vec![
                    (-0.432, -175.548),
                    (-11.368, -83.709),
                    (-11.363, -82.693),
                    (-0.739, -175.601),
                ],
                vec![
                    (-0.393, -176.181),
                    (-11.396, -84.163),
                    (-11.398, -83.231),
                    (-0.752, -175.840),
                ],
                vec![
                    (-0.385, -176.735),
                    (-11.445, -84.475),
                    (-11.452, -83.539),
                    (-0.736, -176.095),
                ],
                vec![
                    (-0.378, -177.418),
                    (-11.479, -84.621),
                    (-11.486, -83.766),
                    (-0.728, -176.458),
                ],
                vec![
                    (-0.357, -178.081),
                    (-11.505, -84.919),
                    (-11.501, -84.040),
                    (-0.745, -176.797),
                ],
                vec![
                    (-0.373, -178.550),
                    (-11.532, -85.329),
                    (-11.527, -84.379),
                    (-0.737, -177.162),
                ],
                vec![
                    (-0.398, -179.219),
                    (-11.555, -85.876),
                    (-11.537, -84.887),
                    (-0.712, -177.517),
                ],
                vec![
                    (-0.406, -179.387),
                    (-11.592, -86.304),
                    (-11.596, -85.116),
                    (-0.697, -178.130),
                ],
                vec![
                    (-0.388, -179.751),
                    (-11.608, -86.455),
                    (-11.617, -85.489),
                    (-0.708, -178.407),
                ],
                vec![
                    (-0.351, 179.972),
                    (-11.662, -86.823),
                    (-11.666, -85.790),
                    (-0.737, -178.606),
                ],
                vec![
                    (-0.309, 179.552),
                    (-11.715, -87.019),
                    (-11.706, -85.940),
                    (-0.744, -178.758),
                ],
                vec![
                    (-0.265, 179.038),
                    (-11.732, -87.209),
                    (-11.729, -86.175),
                    (-0.744, -178.952),
                ],
                vec![
                    (-0.268, 178.546),
                    (-11.731, -87.406),
                    (-11.728, -86.540),
                    (-0.754, -179.276),
                ],
                vec![
                    (-0.257, 178.084),
                    (-11.753, -87.778),
                    (-11.749, -87.009),
                    (-0.785, -179.361),
                ],
                vec![
                    (-0.252, 177.567),
                    (-11.757, -88.069),
                    (-11.774, -87.224),
                    (-0.760, -179.478),
                ],
                vec![
                    (-0.264, 177.133),
                    (-11.802, -88.337),
                    (-11.792, -87.409),
                    (-0.743, -179.786),
                ],
                vec![
                    (-0.284, 176.662),
                    (-11.827, -88.625),
                    (-11.825, -87.942),
                    (-0.737, 179.938),
                ],
                vec![
                    (-0.281, 176.114),
                    (-11.838, -88.888),
                    (-11.841, -88.009),
                    (-0.737, 179.650),
                ],
                vec![
                    (-0.284, 175.439),
                    (-11.849, -89.096),
                    (-11.828, -88.048),
                    (-0.699, 179.298),
                ],
                vec![
                    (-0.315, 174.679),
                    (-11.814, -89.502),
                    (-11.828, -88.608),
                    (-0.682, 178.625),
                ],
                vec![
                    (-0.346, 174.031),
                    (-11.835, -89.972),
                    (-11.824, -88.968),
                    (-0.720, 178.373),
                ],
                vec![
                    (-0.338, 173.956),
                    (-11.827, -90.109),
                    (-11.831, -89.305),
                    (-0.713, 178.598),
                ],
                vec![
                    (-0.273, 173.583),
                    (-11.855, -90.206),
                    (-11.862, -89.369),
                    (-0.648, 178.595),
                ],
                vec![
                    (-0.266, 173.005),
                    (-11.846, -90.457),
                    (-11.845, -89.570),
                    (-0.609, 178.294),
                ],
                vec![
                    (-0.232, 172.456),
                    (-11.843, -90.902),
                    (-11.855, -90.128),
                    (-0.612, 177.855),
                ],
                vec![
                    (-0.223, 172.030),
                    (-11.847, -91.056),
                    (-11.850, -90.397),
                    (-0.575, 177.768),
                ],
                vec![
                    (-0.227, 171.491),
                    (-11.866, -91.933),
                    (-11.857, -91.056),
                    (-0.561, 177.593),
                ],
                vec![
                    (-0.211, 170.829),
                    (-11.908, -92.002),
                    (-11.879, -91.092),
                    (-0.489, 177.305),
                ],
                vec![
                    (-0.233, 170.119),
                    (-11.866, -92.554),
                    (-11.893, -91.468),
                    (-0.476, 176.579),
                ],
                vec![
                    (-0.204, 169.606),
                    (-11.915, -92.825),
                    (-11.903, -92.017),
                    (-0.448, 176.373),
                ],
                vec![
                    (-0.235, 168.777),
                    (-11.933, -93.385),
                    (-11.912, -92.605),
                    (-0.420, 175.689),
                ],
                vec![
                    (-0.271, 167.832),
                    (-11.947, -93.891),
                    (-11.939, -93.206),
                    (-0.472, 175.246),
                ],
                vec![
                    (-0.306, 167.102),
                    (-11.957, -94.149),
                    (-11.954, -93.210),
                    (-0.448, 175.324),
                ],
                vec![
                    (-0.258, 166.964),
                    (-11.928, -94.426),
                    (-11.944, -93.481),
                    (-0.419, 175.523),
                ],
                vec![
                    (-0.229, 165.980),
                    (-11.950, -94.223),
                    (-11.941, -93.532),
                    (-0.339, 175.625),
                ],
                vec![
                    (-0.213, 165.242),
                    (-11.921, -94.718),
                    (-11.931, -93.936),
                    (-0.281, 175.172),
                ],
                vec![
                    (-0.198, 164.306),
                    (-11.915, -95.109),
                    (-11.905, -94.295),
                    (-0.217, 174.767),
                ],
                vec![
                    (-0.188, 163.445),
                    (-11.865, -95.670),
                    (-11.879, -94.820),
                    (-0.205, 174.577),
                ],
                vec![
                    (-0.181, 162.394),
                    (-11.874, -96.178),
                    (-11.869, -95.263),
                    (-0.135, 174.270),
                ],
                vec![
                    (-0.234, 161.079),
                    (-11.820, -96.781),
                    (-11.834, -95.848),
                    (-0.079, 173.670),
                ],
                vec![
                    (-0.311, 159.862),
                    (-11.797, -97.531),
                    (-11.792, -96.706),
                    (-0.026, 173.252),
                ],
                vec![
                    (-0.473, 158.035),
                    (-11.767, -99.069),
                    (-11.777, -98.139),
                    (0.031, 172.884),
                ],
                vec![
                    (-0.865, 156.756),
                    (-11.916, -100.603),
                    (-11.916, -99.695),
                    (0.055, 172.980),
                ],
                vec![
                    (-1.383, 157.170),
                    (-12.183, -101.839),
                    (-12.185, -100.941),
                    (0.148, 172.980),
                ],
                vec![
                    (-1.786, 159.984),
                    (-12.561, -101.601),
                    (-12.577, -100.705),
                    (0.301, 173.255),
                ],
                vec![
                    (-1.716, 163.563),
                    (-12.816, -99.966),
                    (-12.805, -98.952),
                    (0.502, 172.921),
                ],
                vec![
                    (-1.393, 165.231),
                    (-12.859, -97.814),
                    (-12.853, -96.821),
                    (0.657, 171.744),
                ],
                vec![
                    (-1.030, 165.580),
                    (-12.700, -96.201),
                    (-12.669, -95.453),
                    (0.747, 170.716),
                ],
                vec![
                    (-0.759, 164.875),
                    (-12.514, -95.622),
                    (-12.508, -94.678),
                    (0.780, 169.512),
                ],
                vec![
                    (-0.585, 164.044),
                    (-12.361, -95.610),
                    (-12.335, -94.626),
                    (0.755, 168.895),
                ],
                vec![
                    (-0.439, 163.431),
                    (-12.203, -95.395),
                    (-12.209, -94.420),
                    (0.799, 168.418),
                ],
                vec![
                    (-0.318, 162.847),
                    (-12.067, -95.224),
                    (-12.068, -94.311),
                    (0.841, 167.973),
                ],
            ],
            String::from("test_3"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:48 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Vg(V) = -1.500\n\
            Vg(mA) = -0.000\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n",
            ),
        );
        comp_array_f64(&exemplar_mu, &calc.mu(), margin, "mu(4)");
        comp_array_f64(&exemplar_mu_prime, &calc.mu_prime(), margin, "mu_prime(4)");
    }

    #[test]
    fn network_passivity() {
        let exemplar: Array1<f64> = array![
            -0.877751072712341,
            -1.555130032052954e+05,
            -1.044655264772562e+03,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [c64(0.958, -0.263), c64(-0.846, 0.158)],
                    [c64(0.004, 0.022), c64(0.544, -0.129)],
                ],
                [
                    [c64(2.043, -0.982), c64(-0.238, 3.249)],
                    [c64(-1.421, 3.492), c64(0.123, -394.321)],
                ],
                [
                    [c64(21.329, -0.421), c64(-0.942, 24.282)],
                    [c64(0.138, 0.132), c64(0.329, -0.324)],
                ],
            ]),
            String::from(""),
            String::from(""),
        );
        comp_array_f64(
            &exemplar,
            &calc.passivity(),
            F64Margin::default(),
            "passivity1()",
        );
        assert_eq!(false, calc.is_passive());

        // test.s2p
        let exemplar: Array1<f64> = array![
            -54.98063052120305,
            -52.895212459878266,
            -51.34990324807767,
            -50.82789810196128,
        ];
        let calc = Network::new(
            Frequency::new_scaled(array![0.5, 1.0, 1.5, 2.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(0.9881388526914863, -0.13442709904013195),
                        c64(0.0010346705444205045, 0.011178864909012504)
                    ],
                    [
                        c64(-7.363542304219899, 0.6742816969789206),
                        c64(0.5574653418486702, -0.06665134724424635)
                    ],
                ],
                [
                    [
                        c64(0.9578079036840927, -0.2633207328693372),
                        c64(0.0037206104781559784, 0.021909191616475577)
                    ],
                    [
                        c64(-7.130124628011368, 1.3277987152036197),
                        c64(0.5435045929943587, -0.12869941397967788)
                    ],
                ],
                [
                    [
                        c64(0.9133108288727866, -0.38508398385543624),
                        c64(0.008042664765986755, 0.03190603796445517)
                    ],
                    [
                        c64(-6.9151682810378095, 1.800750901131042),
                        c64(0.5235871604669029, -0.18886435408156288)
                    ],
                ],
                [
                    [
                        c64(0.849070850314753, -0.49577931076259807),
                        c64(0.01381064392153511, 0.04080882571424955)
                    ],
                    [
                        c64(-6.688405272002992, 2.4133819411904995),
                        c64(0.4942211124266797, -0.24774648346309974)
                    ],
                ],
            ]),
            String::from("test"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:43 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 2.500\n\
            set_Vd(mA) = 11.250\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Vg(V) = -0.520\n\
            Vg(mA) = -0.001\n\
            Vd(V) = 2.500\n\
            Vd(mA) = 11.076",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.passivity(),
            F64Margin::default(),
            "passivity2()",
        );
        assert_eq!(false, calc.is_passive());

        // test_2.s2p
        let exemplar: Array1<f64> = array![
            -0.0009113505709374644,
            -0.0013061511353752017,
            -0.0028111846539421717,
            -0.00496615580318217,
            -0.007462604097576172,
            0.005967961262325987,
            0.006869923575824449,
            0.00753106261158415,
            0.00819752844677656,
            0.00914827007826932,
            0.009521049810236742,
            0.010241661184905743,
            0.010917077238731603,
            0.011564599930562214,
            0.012274394710785357,
            0.013234711909150286,
            0.014266591620574963,
            0.015191923848350702,
            0.01604027371714972,
            0.01720529034235054,
            0.018004198231139357,
            0.019087257479334787,
            0.019301974359642343,
            0.019510476240204837,
            0.021057257326300788,
            0.021838109645750465,
            0.022201822577831747,
            0.02270888068459751,
            0.0231898738484958,
            0.024619064343828826,
            0.025006438365973875,
            0.02551584865280376,
            0.02647961108880548,
            0.0262989026103647,
            0.02705173901007702,
            0.027701787734328673,
            0.02751957248649402,
            0.028093581411617226,
            0.02815041478756059,
            0.029163369500518895,
            0.029103220590142923,
            0.02867904494780491,
            0.02859377059405514,
            0.02800751596802347,
            0.029344428946609588,
            0.029518300981194442,
            0.028984986445898055,
            0.030200380783176165,
            0.030681294569593797,
            0.031452623293653115,
            0.03240338396929959,
            0.03253810844373716,
            0.032508519366455456,
            0.03195942884419677,
            0.03306597275717813,
            0.03325208864647101,
            0.03432376242064617,
            0.03637083680544545,
            0.03557387302979074,
            0.03614760572850935,
            0.03657622730474128,
            0.038508878481687506,
            0.038923980384335505,
            0.03968551433457323,
            0.03838132984244879,
            0.0392351621211167,
            0.040182704361672275,
            0.04010493374550444,
            0.03779204639658498,
            0.03779822113975446,
            0.03999075769478012,
            0.040202932453216024,
            0.040787098268263725,
            0.03995076031597712,
            0.04213880690487434,
            0.042994250514755986,
            0.04099379369430408,
            0.04174724092901425,
            0.04144702300681605,
            0.040190196380211436,
            0.03976164713428357,
            0.03988649562931532,
            0.04252301139295391,
            0.03848661135404618,
            0.035694958632223904,
            0.03323211781125624,
            0.036593518975938454,
            0.037508511612305055,
            0.03780666639791438,
            0.03720278871928409,
            0.044139432078531465,
            0.044056623774601754,
            0.03887175866512051,
            0.038075509325837836,
            0.04020936495467023,
            0.045254051417611095,
            0.04400153112091241,
            0.03409109084063034,
            0.028688481927859583,
            0.030008089888061588,
            0.03148019546274176,
            0.03809635765028074,
            0.036190078145246496,
            0.035254266743281006,
            0.03277880951970087,
            0.031586928735969744,
            0.02985510452449489,
            0.027993006839669397,
            0.027578428758575666,
            0.025760861805766203,
            0.025142365746019985,
            0.023805451452598728,
            0.028679673113880817,
            0.03115824387123067,
            0.031203762583634626,
            0.030094617992653328,
            0.030309040965643245,
            0.02496708436270264,
            0.024095313340673427,
            0.030103189603230704,
            0.031255614854633344,
            0.025377755952780124,
            0.025316008107437032,
            0.023090061705665667,
            0.028835209539988867,
            0.02500577122694489,
            0.021719511529547104,
            0.018291033239153348,
            0.018909259624275417,
            0.014885588026512698,
            0.019316090985082326,
            0.026805015173749676,
            0.0262635225810875,
            0.04235073743766532,
            0.05231672578244275,
            0.0638013521909085,
            0.06504086826981742,
            0.06402076857105221,
            0.06824594952455373,
            0.07144181455827142,
            0.07136599361793175,
            0.06952631004712198,
            0.06275933347361051,
            0.06766496017682025,
            0.0685303319129975,
            0.06724000412984288,
            0.05956340723166936,
            0.06521765483986641,
            0.07292107994518404,
            0.07503332196148016,
            0.07467043144165426,
            0.07233302557215136,
            0.06825204957636108,
            0.06132748352992964,
            0.05363040954640209,
            0.05051157221903351,
            0.054849228109390406,
            0.057604228213037914,
            0.05786524453451938,
            0.0662723138101016,
            0.06414789855962837,
            0.0728647747180782,
            0.07957353100152792,
            0.08293774288700174,
            0.07485654921038042,
            0.06089952550401751,
            0.05435494796547718,
            0.05251516747357109,
            0.05381015572102627,
            0.05394547411276976,
            0.046837752519932115,
            0.04510734406857079,
            0.04643632763390148,
            0.046061084380162785,
            0.04250143684122031,
            0.04553742993401123,
            0.04119544400040008,
            0.03424043284490034,
            0.027448187423823378,
            0.023019633074324173,
            0.02135887882051983,
            0.02099905690808135,
            0.019185055518465128,
            0.016243854498552937,
            0.01533114911580749,
            0.013359796840335088,
            0.008312413006756787,
            0.007022132578955594,
            0.004989544833378738,
            -0.006601862267667072,
            -0.02073514669233851,
            -0.02722435500844955,
            -0.030663791064866868,
            -0.0393041161596658,
            -0.050593634222853213,
            -0.06311070603168209,
            -0.0593895886756688,
            -0.07404092241808669,
            -0.08862641768741994,
            -0.0979043867388246,
            -0.10723365377196141,
            -0.12100152410633919,
            -0.13393382185206967,
            -0.13549795427120048,
            -0.14005487588952903,
            -0.1387109790237376,
            -0.1471695284193781,
            -0.15013492116704322,
            -0.1484866627654558,
            -0.1518836619155665,
            -0.1562716849596431,
            -0.15135737950068326,
            -0.14622277833953148,
            -0.15363795257912005,
            -0.1554792896405395,
            -0.17372768824036347,
            -0.1965605181028082,
            -0.23485793234935984,
            -0.27371713925331753,
            -0.33326270086189624,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.passivity(),
            F64Margin::default(),
            "passivity3()",
        );
        assert_eq!(false, calc.is_passive());

        // test_2.s2p subset
        let exemplar: Array1<f64> = array![
            0.005967961262325987,
            0.006869923575824449,
            0.00753106261158415,
            0.00819752844677656,
            0.00914827007826932,
            0.009521049810236742,
            0.010241661184905743,
            0.010917077238731603,
            0.011564599930562214,
            0.012274394710785357,
            0.013234711909150286,
            0.014266591620574963,
            0.015191923848350702,
            0.01604027371714972,
            0.01720529034235054,
            0.018004198231139357,
            0.019087257479334787,
            0.019301974359642343,
            0.019510476240204837,
            0.021057257326300788,
            0.021838109645750465,
            0.022201822577831747,
            0.02270888068459751,
            0.0231898738484958,
            0.024619064343828826,
            0.025006438365973875,
            0.02551584865280376,
            0.02647961108880548,
            0.0262989026103647,
            0.02705173901007702,
            0.027701787734328673,
            0.02751957248649402,
            0.028093581411617226,
            0.02815041478756059,
            0.029163369500518895,
            0.029103220590142923,
            0.02867904494780491,
            0.02859377059405514,
            0.02800751596802347,
            0.029344428946609588,
            0.029518300981194442,
            0.028984986445898055,
            0.030200380783176165,
            0.030681294569593797,
            0.031452623293653115,
            0.03240338396929959,
            0.03253810844373716,
            0.032508519366455456,
            0.03195942884419677,
            0.03306597275717813,
            0.03325208864647101,
            0.03432376242064617,
            0.03637083680544545,
            0.03557387302979074,
            0.03614760572850935,
            0.03657622730474128,
            0.038508878481687506,
            0.038923980384335505,
            0.03968551433457323,
            0.03838132984244879,
            0.0392351621211167,
            0.040182704361672275,
            0.04010493374550444,
            0.03779204639658498,
            0.03779822113975446,
            0.03999075769478012,
            0.040202932453216024,
            0.040787098268263725,
            0.03995076031597712,
            0.04213880690487434,
            0.042994250514755986,
            0.04099379369430408,
            0.04174724092901425,
            0.04144702300681605,
            0.040190196380211436,
            0.03976164713428357,
            0.03988649562931532,
            0.04252301139295391,
            0.03848661135404618,
            0.035694958632223904,
            0.03323211781125624,
            0.036593518975938454,
            0.037508511612305055,
            0.03780666639791438,
            0.03720278871928409,
            0.044139432078531465,
            0.044056623774601754,
            0.03887175866512051,
            0.038075509325837836,
            0.04020936495467023,
            0.045254051417611095,
            0.04400153112091241,
            0.03409109084063034,
            0.028688481927859583,
            0.030008089888061588,
            0.03148019546274176,
            0.03809635765028074,
            0.036190078145246496,
            0.035254266743281006,
            0.03277880951970087,
            0.031586928735969744,
            0.02985510452449489,
            0.027993006839669397,
            0.027578428758575666,
            0.025760861805766203,
            0.025142365746019985,
            0.023805451452598728,
            0.028679673113880817,
            0.03115824387123067,
            0.031203762583634626,
            0.030094617992653328,
            0.030309040965643245,
            0.02496708436270264,
            0.024095313340673427,
            0.030103189603230704,
            0.031255614854633344,
            0.025377755952780124,
            0.025316008107437032,
            0.023090061705665667,
            0.028835209539988867,
            0.02500577122694489,
            0.021719511529547104,
            0.018291033239153348,
            0.018909259624275417,
            0.014885588026512698,
            0.019316090985082326,
            0.026805015173749676,
            0.0262635225810875,
            0.04235073743766532,
            0.05231672578244275,
            0.0638013521909085,
            0.06504086826981742,
            0.06402076857105221,
            0.06824594952455373,
            0.07144181455827142,
            0.07136599361793175,
            0.06952631004712198,
            0.06275933347361051,
            0.06766496017682025,
            0.0685303319129975,
            0.06724000412984288,
            0.05956340723166936,
            0.06521765483986641,
            0.07292107994518404,
            0.07503332196148016,
            0.07467043144165426,
            0.07233302557215136,
            0.06825204957636108,
            0.06132748352992964,
            0.05363040954640209,
            0.05051157221903351,
            0.054849228109390406,
            0.057604228213037914,
            0.05786524453451938,
            0.0662723138101016,
            0.06414789855962837,
            0.0728647747180782,
            0.07957353100152792,
            0.08293774288700174,
            0.07485654921038042,
            0.06089952550401751,
            0.05435494796547718,
            0.05251516747357109,
            0.05381015572102627,
            0.05394547411276976,
            0.046837752519932115,
            0.04510734406857079,
            0.04643632763390148,
            0.046061084380162785,
            0.04250143684122031,
            0.04553742993401123,
            0.04119544400040008,
            0.03424043284490034,
            0.027448187423823378,
            0.023019633074324173,
            0.02135887882051983,
            0.02099905690808135,
            0.019185055518465128,
            0.016243854498552937,
            0.01533114911580749,
            0.013359796840335088,
            0.008312413006756787,
            0.007022132578955594,
            0.004989544833378738,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(3.0, 94.5, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Array1 = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.passivity(),
            F64Margin::default(),
            "passivity4()",
        );
        assert_eq!(true, calc.is_passive());

        // test_3.s2p
        let exemplar: Array1<f64> = array![
            -0.0009358703851608015,
            -0.0004751910352422173,
            -0.0016388105307448282,
            -0.0024807926237956157,
            0.003200092101430906,
            0.0029949588071418753,
            0.00383082026379238,
            0.0040424437424541534,
            0.005034367373001606,
            0.005624175617901847,
            0.005557485709442892,
            0.006181145766643306,
            0.006284983097060534,
            0.006852469931130024,
            0.007459840851656765,
            0.0076356667316768035,
            0.008082818389970825,
            0.009208243030462183,
            0.009346430581839493,
            0.010126397539540108,
            0.01114104769233018,
            0.011707778269526998,
            0.012537948695204651,
            0.012438269161183948,
            0.012211385534083167,
            0.012898514545170133,
            0.014049175359776657,
            0.014882022262722167,
            0.01549935075447642,
            0.015627767777468087,
            0.01611588662135346,
            0.017060680591684647,
            0.017903244562487795,
            0.019236111744253943,
            0.01905243532861832,
            0.019534865768967305,
            0.019989463848929735,
            0.019825313536182707,
            0.020323977943514742,
            0.020561011562141115,
            0.02048477566544946,
            0.019984679313648276,
            0.020474148417859335,
            0.020934779912172904,
            0.02075075004377474,
            0.0209928925257917,
            0.02123054865573266,
            0.024566050388645566,
            0.025802798595331595,
            0.027544443054322573,
            0.027569639636774966,
            0.026035052466148467,
            0.02570798971949613,
            0.02751140962249576,
            0.029847259027179975,
            0.03094481955319904,
            0.029997047137258405,
            0.02946596701193553,
            0.032684553422944726,
            0.033925276630326295,
            0.03517933926213475,
            0.03367363174340867,
            0.033813081260648775,
            0.035128066781452146,
            0.03818429981821651,
            0.03770699020376847,
            0.03693779886932706,
            0.03463303848834578,
            0.03513244390097195,
            0.03360170266946791,
            0.032526033447043926,
            0.03261456490561606,
            0.03510310069553454,
            0.03404702493118406,
            0.030343234235483005,
            0.028748235700935473,
            0.026954258995783402,
            0.031590640356446736,
            0.03263378900601126,
            0.029069265248180542,
            0.02459386719164757,
            0.01991686926069227,
            0.02159837338279605,
            0.024705364215590008,
            0.02603498669921643,
            0.02458900975822646,
            0.02429984465437427,
            0.026690698897847455,
            0.02824244691477921,
            0.03309089107789362,
            0.031541216746356536,
            0.03428245784568834,
            0.03150927894380914,
            0.033139324888041236,
            0.03194029387945722,
            0.03286917590362975,
            0.030659603101776543,
            0.037772496850547344,
            0.036816482848160005,
            0.04068928790873966,
            0.029257239739822197,
            0.03413502808587646,
            0.039364004064280936,
            0.043634788268460206,
            0.04801938423743364,
            0.04795921186432038,
            0.04450529011622111,
            0.04367782124222273,
            0.04165263444884464,
            0.04254949996780433,
            0.03873068294480828,
            0.03148476150747967,
            0.02808617572306865,
            0.030246278780146615,
            0.026975950030315922,
            0.03097796874532648,
            0.03545082396137062,
            0.03986405733595653,
            0.03898763786643733,
            0.03062214938145293,
            0.02304749741226174,
            0.019401326831886178,
            0.016576208479219468,
            0.015087838670989201,
            0.014141940698872237,
            0.014649218556528346,
            0.018110827634297198,
            0.035010768166816454,
            0.03947915117195888,
            0.03139948484587253,
            0.02178562558609046,
            0.027308118008855062,
            0.041774835903177876,
            0.05174524124611374,
            0.033769701010609075,
            0.04033507556884541,
            0.050485591327169455,
            0.057028294145788555,
            0.058803323291099316,
            0.05854631745655725,
            0.06097516420553098,
            0.06430555329805104,
            0.06899937778602178,
            0.06611325546448668,
            0.05956313715800718,
            0.0577168687206608,
            0.0552719371176061,
            0.055935664866732196,
            0.05476961885094263,
            0.055083050609508064,
            0.054597304995674965,
            0.053230914452553096,
            0.053439409608030966,
            0.05202694851354697,
            0.051225213233986884,
            0.05425099104158546,
            0.059226607548691135,
            0.05906432253639421,
            0.06028911909406964,
            0.054397539064063236,
            0.05077275536148523,
            0.04944043031144994,
            0.045268261751885544,
            0.039628172979378845,
            0.032463326409157514,
            0.026065989121250653,
            0.02181159257055469,
            0.019700783440572675,
            0.014190813774107493,
            0.007414600365762775,
            0.006181684116958937,
            0.0037762249881671556,
            -0.0008245442323683452,
            0.002524852888044526,
            0.007355778670751651,
            0.009111591774107072,
            0.005925771759814499,
            0.000517442800342054,
            -0.007575925989490236,
            -0.016873483808923527,
            -0.016697349977655224,
            -0.017773567816869465,
            -0.019013184338647928,
            -0.016722403817951138,
            -0.012022925104389893,
            -0.013521129348525007,
            -0.015431066279963851,
            -0.011384770428554571,
            -0.004624950300987505,
            -0.004580124674270889,
            -0.018620947804218232,
            -0.022409111309694037,
            -0.02875244289630876,
            -0.031712626956895695,
            -0.02867424946936037,
            -0.03597230815473083,
            -0.034569277320124506,
            -0.039993759079498384,
            -0.03726518845481035,
            -0.029141171918869065,
            -0.02647244378968814,
            -0.0336567125284827,
            -0.04765981774015077,
            -0.05645202798567673,
            -0.06756644241618362,
            -0.07067154673827776,
            -0.08142325058822568,
            -0.08853990385904707,
            -0.09306711230088108,
            -0.09569409919938685,
            -0.0888999697029387,
            -0.10286185721876094,
            -0.1342950768489003,
            -0.18207558593154793,
            -0.22338798459441622,
            -0.25253596575339493,
            -0.268299468798073,
            -0.2674989719098801,
            -0.2859116231773699,
            -0.3041241279188261,
        ];
        let calc = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            ComplexNumberType::Db,
            vec![
                vec![
                    (-0.007, -2.703),
                    (-33.308, 87.205),
                    (-33.310, 87.223),
                    (0.002, -2.522),
                ],
                vec![
                    (-0.014, -5.380),
                    (-27.320, 84.747),
                    (-27.319, 84.763),
                    (-0.006, -5.048),
                ],
                vec![
                    (-0.020, -8.054),
                    (-23.824, 82.282),
                    (-23.822, 82.309),
                    (-0.011, -7.535),
                ],
                vec![
                    (-0.030, -10.721),
                    (-21.356, 79.850),
                    (-21.355, 79.863),
                    (-0.022, -10.008),
                ],
                vec![
                    (-0.070, -12.934),
                    (-19.767, 76.759),
                    (-19.766, 76.774),
                    (-0.073, -12.097),
                ],
                vec![
                    (-0.102, -15.528),
                    (-18.235, 74.354),
                    (-18.233, 74.341),
                    (-0.086, -14.420),
                ],
                vec![
                    (-0.122, -18.012),
                    (-16.968, 71.927),
                    (-16.964, 71.952),
                    (-0.118, -16.765),
                ],
                vec![
                    (-0.156, -20.492),
                    (-15.876, 69.541),
                    (-15.872, 69.531),
                    (-0.143, -19.112),
                ],
                vec![
                    (-0.185, -23.013),
                    (-14.932, 67.149),
                    (-14.928, 67.141),
                    (-0.181, -21.370),
                ],
                vec![
                    (-0.224, -25.457),
                    (-14.103, 64.781),
                    (-14.100, 64.755),
                    (-0.214, -23.659),
                ],
                vec![
                    (-0.265, -27.878),
                    (-13.365, 62.489),
                    (-13.360, 62.451),
                    (-0.244, -25.890),
                ],
                vec![
                    (-0.302, -30.240),
                    (-12.705, 60.203),
                    (-12.703, 60.163),
                    (-0.284, -28.109),
                ],
                vec![
                    (-0.344, -32.603),
                    (-12.109, 57.908),
                    (-12.107, 57.880),
                    (-0.322, -30.298),
                ],
                vec![
                    (-0.387, -34.903),
                    (-11.574, 55.657),
                    (-11.572, 55.620),
                    (-0.364, -32.496),
                ],
                vec![
                    (-0.429, -37.197),
                    (-11.087, 53.446),
                    (-11.086, 53.458),
                    (-0.407, -34.613),
                ],
                vec![
                    (-0.463, -39.356),
                    (-10.675, 51.542),
                    (-10.671, 51.469),
                    (-0.446, -36.438),
                ],
                vec![
                    (-0.515, -41.554),
                    (-10.268, 49.344),
                    (-10.270, 49.333),
                    (-0.488, -38.514),
                ],
                vec![
                    (-0.562, -43.746),
                    (-9.897, 47.250),
                    (-9.897, 47.193),
                    (-0.536, -40.558),
                ],
                vec![
                    (-0.610, -45.867),
                    (-9.555, 45.112),
                    (-9.555, 45.063),
                    (-0.582, -42.571),
                ],
                vec![
                    (-0.660, -47.981),
                    (-9.247, 43.031),
                    (-9.245, 42.983),
                    (-0.630, -44.513),
                ],
                vec![
                    (-0.704, -50.043),
                    (-8.963, 41.024),
                    (-8.964, 40.994),
                    (-0.679, -46.423),
                ],
                vec![
                    (-0.757, -52.064),
                    (-8.703, 39.062),
                    (-8.702, 39.007),
                    (-0.720, -48.330),
                ],
                vec![
                    (-0.799, -54.087),
                    (-8.460, 37.129),
                    (-8.460, 37.075),
                    (-0.767, -50.166),
                ],
                vec![
                    (-0.851, -56.013),
                    (-8.237, 35.258),
                    (-8.238, 35.248),
                    (-0.800, -51.986),
                ],
                vec![
                    (-0.898, -57.881),
                    (-8.025, 33.392),
                    (-8.025, 33.331),
                    (-0.839, -53.849),
                ],
                vec![
                    (-0.939, -59.713),
                    (-7.832, 31.566),
                    (-7.832, 31.496),
                    (-0.882, -55.676),
                ],
                vec![
                    (-0.982, -61.565),
                    (-7.656, 29.737),
                    (-7.658, 29.701),
                    (-0.927, -57.406),
                ],
                vec![
                    (-1.021, -63.357),
                    (-7.495, 27.999),
                    (-7.495, 27.920),
                    (-0.968, -59.135),
                ],
                vec![
                    (-1.056, -65.122),
                    (-7.345, 26.233),
                    (-7.349, 26.172),
                    (-1.010, -60.814),
                ],
                vec![
                    (-1.093, -66.877),
                    (-7.211, 24.513),
                    (-7.210, 24.431),
                    (-1.045, -62.470),
                ],
                vec![
                    (-1.130, -68.611),
                    (-7.084, 22.839),
                    (-7.085, 22.760),
                    (-1.077, -64.114),
                ],
                vec![
                    (-1.166, -70.296),
                    (-6.964, 21.187),
                    (-6.963, 21.105),
                    (-1.113, -65.762),
                ],
                vec![
                    (-1.199, -71.933),
                    (-6.856, 19.554),
                    (-6.855, 19.467),
                    (-1.149, -67.361),
                ],
                vec![
                    (-1.228, -73.544),
                    (-6.761, 17.969),
                    (-6.761, 17.907),
                    (-1.184, -68.913),
                ],
                vec![
                    (-1.258, -75.146),
                    (-6.668, 16.427),
                    (-6.669, 16.307),
                    (-1.209, -70.427),
                ],
                vec![
                    (-1.293, -76.757),
                    (-6.583, 14.877),
                    (-6.587, 14.748),
                    (-1.235, -71.896),
                ],
                vec![
                    (-1.316, -78.293),
                    (-6.503, 13.359),
                    (-6.507, 13.277),
                    (-1.261, -73.391),
                ],
                vec![
                    (-1.339, -79.774),
                    (-6.431, 11.869),
                    (-6.434, 11.757),
                    (-1.286, -74.841),
                ],
                vec![
                    (-1.362, -81.225),
                    (-6.369, 10.428),
                    (-6.372, 10.325),
                    (-1.308, -76.268),
                ],
                vec![
                    (-1.387, -82.666),
                    (-6.312, 8.931),
                    (-6.317, 8.864),
                    (-1.330, -77.697),
                ],
                vec![
                    (-1.405, -84.120),
                    (-6.257, 7.509),
                    (-6.262, 7.430),
                    (-1.347, -79.106),
                ],
                vec![
                    (-1.419, -85.518),
                    (-6.208, 6.101),
                    (-6.213, 5.976),
                    (-1.365, -80.482),
                ],
                vec![
                    (-1.429, -86.867),
                    (-6.165, 4.717),
                    (-6.169, 4.653),
                    (-1.384, -81.874),
                ],
                vec![
                    (-1.447, -88.250),
                    (-6.127, 3.340),
                    (-6.132, 3.226),
                    (-1.400, -83.269),
                ],
                vec![
                    (-1.465, -89.627),
                    (-6.089, 1.959),
                    (-6.092, 1.849),
                    (-1.410, -84.648),
                ],
                vec![
                    (-1.477, -90.985),
                    (-6.063, 0.594),
                    (-6.066, 0.471),
                    (-1.424, -85.964),
                ],
                vec![
                    (-1.481, -92.320),
                    (-6.052, -0.794),
                    (-6.055, -0.900),
                    (-1.441, -87.215),
                ],
                vec![
                    (-1.488, -93.626),
                    (-6.056, -2.034),
                    (-6.060, -2.165),
                    (-1.466, -88.427),
                ],
                vec![
                    (-1.504, -94.858),
                    (-6.038, -3.236),
                    (-6.039, -3.361),
                    (-1.474, -89.665),
                ],
                vec![
                    (-1.520, -96.087),
                    (-6.023, -4.482),
                    (-6.024, -4.590),
                    (-1.486, -90.966),
                ],
                vec![
                    (-1.522, -97.307),
                    (-6.009, -5.661),
                    (-6.008, -5.786),
                    (-1.490, -92.174),
                ],
                vec![
                    (-1.518, -98.492),
                    (-6.002, -6.874),
                    (-6.003, -6.991),
                    (-1.485, -93.349),
                ],
                vec![
                    (-1.517, -99.685),
                    (-5.991, -8.055),
                    (-5.995, -8.196),
                    (-1.487, -94.556),
                ],
                vec![
                    (-1.539, -100.827),
                    (-5.988, -9.359),
                    (-5.989, -9.478),
                    (-1.508, -95.800),
                ],
                vec![
                    (-1.549, -101.915),
                    (-6.011, -10.524),
                    (-6.016, -10.644),
                    (-1.520, -96.906),
                ],
                vec![
                    (-1.535, -103.121),
                    (-6.030, -11.560),
                    (-6.035, -11.671),
                    (-1.513, -97.983),
                ],
                vec![
                    (-1.527, -104.305),
                    (-6.029, -12.498),
                    (-6.032, -12.599),
                    (-1.483, -99.050),
                ],
                vec![
                    (-1.536, -105.323),
                    (-6.025, -13.680),
                    (-6.030, -13.823),
                    (-1.482, -100.338),
                ],
                vec![
                    (-1.543, -106.386),
                    (-6.035, -14.827),
                    (-6.036, -14.953),
                    (-1.508, -101.481),
                ],
                vec![
                    (-1.532, -107.502),
                    (-6.043, -15.925),
                    (-6.049, -16.028),
                    (-1.519, -102.578),
                ],
                vec![
                    (-1.523, -108.630),
                    (-6.073, -17.002),
                    (-6.074, -17.148),
                    (-1.520, -103.680),
                ],
                vec![
                    (-1.517, -109.630),
                    (-6.085, -18.045),
                    (-6.088, -18.163),
                    (-1.507, -104.645),
                ],
                vec![
                    (-1.517, -110.526),
                    (-6.108, -19.017),
                    (-6.115, -19.129),
                    (-1.502, -105.568),
                ],
                vec![
                    (-1.523, -111.551),
                    (-6.133, -19.965),
                    (-6.132, -20.109),
                    (-1.491, -106.576),
                ],
                vec![
                    (-1.520, -112.602),
                    (-6.146, -20.916),
                    (-6.148, -21.090),
                    (-1.497, -107.712),
                ],
                vec![
                    (-1.505, -113.579),
                    (-6.160, -21.929),
                    (-6.164, -22.133),
                    (-1.493, -108.803),
                ],
                vec![
                    (-1.496, -114.428),
                    (-6.187, -22.916),
                    (-6.190, -23.068),
                    (-1.490, -109.691),
                ],
                vec![
                    (-1.492, -115.261),
                    (-6.202, -23.910),
                    (-6.201, -24.079),
                    (-1.478, -110.630),
                ],
                vec![
                    (-1.479, -116.159),
                    (-6.233, -24.888),
                    (-6.235, -25.011),
                    (-1.486, -111.506),
                ],
                vec![
                    (-1.450, -117.227),
                    (-6.269, -25.884),
                    (-6.268, -25.983),
                    (-1.472, -112.436),
                ],
                vec![
                    (-1.415, -118.310),
                    (-6.304, -26.742),
                    (-6.300, -26.933),
                    (-1.449, -113.459),
                ],
                vec![
                    (-1.406, -119.173),
                    (-6.318, -27.627),
                    (-6.319, -27.787),
                    (-1.436, -114.468),
                ],
                vec![
                    (-1.416, -120.155),
                    (-6.333, -28.537),
                    (-6.336, -28.705),
                    (-1.442, -115.341),
                ],
                vec![
                    (-1.412, -121.119),
                    (-6.354, -29.574),
                    (-6.357, -29.733),
                    (-1.429, -116.311),
                ],
                vec![
                    (-1.375, -122.107),
                    (-6.378, -30.598),
                    (-6.386, -30.749),
                    (-1.411, -117.337),
                ],
                vec![
                    (-1.348, -122.983),
                    (-6.417, -31.525),
                    (-6.420, -31.689),
                    (-1.405, -118.305),
                ],
                vec![
                    (-1.340, -123.747),
                    (-6.459, -32.495),
                    (-6.472, -32.682),
                    (-1.397, -119.061),
                ],
                vec![
                    (-1.351, -124.603),
                    (-6.527, -33.267),
                    (-6.525, -33.467),
                    (-1.388, -119.865),
                ],
                vec![
                    (-1.339, -125.547),
                    (-6.566, -34.045),
                    (-6.567, -34.205),
                    (-1.374, -120.594),
                ],
                vec![
                    (-1.313, -126.503),
                    (-6.593, -34.907),
                    (-6.594, -35.053),
                    (-1.332, -121.429),
                ],
                vec![
                    (-1.287, -127.265),
                    (-6.622, -35.784),
                    (-6.631, -36.002),
                    (-1.303, -122.280),
                ],
                vec![
                    (-1.265, -127.923),
                    (-6.640, -36.573),
                    (-6.655, -36.827),
                    (-1.275, -123.026),
                ],
                vec![
                    (-1.276, -128.868),
                    (-6.676, -37.454),
                    (-6.683, -37.685),
                    (-1.248, -124.052),
                ],
                vec![
                    (-1.276, -129.931),
                    (-6.720, -38.438),
                    (-6.734, -38.649),
                    (-1.253, -124.964),
                ],
                vec![
                    (-1.258, -130.950),
                    (-6.775, -39.360),
                    (-6.782, -39.607),
                    (-1.251, -125.870),
                ],
                vec![
                    (-1.250, -131.674),
                    (-6.823, -40.202),
                    (-6.838, -40.386),
                    (-1.227, -126.610),
                ],
                vec![
                    (-1.245, -132.330),
                    (-6.872, -40.937),
                    (-6.868, -41.097),
                    (-1.203, -127.499),
                ],
                vec![
                    (-1.264, -133.188),
                    (-6.919, -41.663),
                    (-6.921, -41.828),
                    (-1.186, -128.275),
                ],
                vec![
                    (-1.241, -133.962),
                    (-6.979, -42.437),
                    (-6.979, -42.740),
                    (-1.181, -129.251),
                ],
                vec![
                    (-1.244, -134.873),
                    (-7.035, -43.207),
                    (-7.046, -43.432),
                    (-1.178, -130.173),
                ],
                vec![
                    (-1.233, -135.444),
                    (-7.079, -43.963),
                    (-7.079, -44.149),
                    (-1.163, -130.927),
                ],
                vec![
                    (-1.204, -137.506),
                    (-7.295, -45.675),
                    (-7.287, -45.895),
                    (-1.120, -132.380),
                ],
                vec![
                    (-1.196, -138.266),
                    (-7.331, -46.540),
                    (-7.340, -46.723),
                    (-1.101, -132.999),
                ],
                vec![
                    (-1.184, -139.237),
                    (-7.406, -47.224),
                    (-7.392, -47.498),
                    (-1.081, -133.941),
                ],
                vec![
                    (-1.156, -139.806),
                    (-7.439, -47.942),
                    (-7.442, -48.261),
                    (-1.066, -134.851),
                ],
                vec![
                    (-1.141, -140.499),
                    (-7.512, -48.823),
                    (-7.507, -49.026),
                    (-1.057, -135.763),
                ],
                vec![
                    (-1.151, -141.070),
                    (-7.544, -49.468),
                    (-7.552, -49.677),
                    (-1.033, -136.450),
                ],
                vec![
                    (-1.139, -141.988),
                    (-7.619, -50.241),
                    (-7.621, -50.511),
                    (-1.060, -137.163),
                ],
                vec![
                    (-1.126, -142.624),
                    (-7.678, -51.043),
                    (-7.661, -51.286),
                    (-1.045, -138.047),
                ],
                vec![
                    (-1.090, -143.317),
                    (-7.754, -51.827),
                    (-7.754, -51.818),
                    (-1.059, -138.695),
                ],
                vec![
                    (-1.090, -143.804),
                    (-7.812, -52.472),
                    (-7.801, -52.722),
                    (-0.973, -139.574),
                ],
                vec![
                    (-1.082, -144.444),
                    (-7.886, -53.018),
                    (-7.910, -53.153),
                    (-0.968, -140.498),
                ],
                vec![
                    (-1.091, -145.084),
                    (-7.937, -53.366),
                    (-7.948, -53.552),
                    (-0.981, -140.970),
                ],
                vec![
                    (-1.073, -145.770),
                    (-7.979, -53.983),
                    (-7.999, -54.149),
                    (-0.994, -141.596),
                ],
                vec![
                    (-1.063, -146.276),
                    (-8.043, -54.559),
                    (-8.026, -54.782),
                    (-1.008, -142.399),
                ],
                vec![
                    (-1.042, -146.836),
                    (-8.078, -55.281),
                    (-8.074, -55.531),
                    (-1.001, -143.284),
                ],
                vec![
                    (-1.033, -147.673),
                    (-8.132, -56.080),
                    (-8.110, -56.310),
                    (-0.981, -143.668),
                ],
                vec![
                    (-1.042, -148.451),
                    (-8.160, -56.728),
                    (-8.174, -56.898),
                    (-0.965, -144.102),
                ],
                vec![
                    (-1.024, -148.900),
                    (-8.249, -57.285),
                    (-8.226, -57.521),
                    (-0.941, -144.796),
                ],
                vec![
                    (-0.995, -149.415),
                    (-8.286, -57.910),
                    (-8.272, -58.101),
                    (-0.938, -145.596),
                ],
                vec![
                    (-0.989, -150.029),
                    (-8.310, -58.664),
                    (-8.327, -58.674),
                    (-0.907, -146.361),
                ],
                vec![
                    (-0.991, -150.499),
                    (-8.371, -59.225),
                    (-8.362, -59.245),
                    (-0.863, -146.713),
                ],
                vec![
                    (-1.005, -150.999),
                    (-8.404, -59.719),
                    (-8.422, -59.990),
                    (-0.836, -147.464),
                ],
                vec![
                    (-0.991, -151.557),
                    (-8.484, -60.340),
                    (-8.449, -60.444),
                    (-0.834, -148.351),
                ],
                vec![
                    (-0.992, -152.156),
                    (-8.521, -61.244),
                    (-8.512, -61.261),
                    (-0.813, -149.016),
                ],
                vec![
                    (-0.994, -152.469),
                    (-8.563, -61.662),
                    (-8.583, -62.009),
                    (-0.825, -149.822),
                ],
                vec![
                    (-0.996, -152.775),
                    (-8.620, -62.219),
                    (-8.625, -62.491),
                    (-0.844, -150.308),
                ],
                vec![
                    (-0.998, -153.136),
                    (-8.656, -62.852),
                    (-8.703, -63.102),
                    (-0.866, -150.828),
                ],
                vec![
                    (-0.956, -153.739),
                    (-8.707, -63.513),
                    (-8.698, -63.648),
                    (-0.869, -151.174),
                ],
                vec![
                    (-0.964, -154.217),
                    (-8.762, -64.131),
                    (-8.749, -64.117),
                    (-0.803, -151.983),
                ],
                vec![
                    (-0.949, -154.618),
                    (-8.809, -64.854),
                    (-8.772, -65.287),
                    (-0.774, -152.579),
                ],
                vec![
                    (-0.920, -155.032),
                    (-8.834, -65.647),
                    (-8.851, -65.891),
                    (-0.753, -153.294),
                ],
                vec![
                    (-0.903, -155.591),
                    (-8.860, -66.464),
                    (-8.894, -66.700),
                    (-0.741, -154.031),
                ],
                vec![
                    (-0.907, -156.265),
                    (-8.938, -67.399),
                    (-8.959, -67.342),
                    (-0.720, -154.847),
                ],
                vec![
                    (-0.896, -156.802),
                    (-9.031, -67.658),
                    (-9.033, -68.226),
                    (-0.692, -155.857),
                ],
                vec![
                    (-0.856, -157.201),
                    (-9.084, -68.597),
                    (-9.143, -68.901),
                    (-0.706, -156.318),
                ],
                vec![
                    (-0.874, -157.546),
                    (-9.202, -69.240),
                    (-9.200, -69.721),
                    (-0.727, -156.732),
                ],
                vec![
                    (-0.922, -158.150),
                    (-9.258, -69.906),
                    (-9.279, -70.164),
                    (-0.807, -157.422),
                ],
                vec![
                    (-0.947, -158.704),
                    (-9.308, -70.926),
                    (-9.333, -70.844),
                    (-0.835, -158.104),
                ],
                vec![
                    (-0.895, -159.033),
                    (-9.413, -71.526),
                    (-9.401, -71.758),
                    (-0.804, -158.610),
                ],
                vec![
                    (-0.854, -159.318),
                    (-9.522, -72.082),
                    (-9.547, -72.352),
                    (-0.752, -158.691),
                ],
                vec![
                    (-0.838, -159.873),
                    (-9.683, -72.618),
                    (-9.692, -73.056),
                    (-0.775, -159.259),
                ],
                vec![
                    (-0.849, -160.766),
                    (-9.746, -73.258),
                    (-9.704, -73.203),
                    (-0.850, -160.019),
                ],
                vec![
                    (-0.846, -161.398),
                    (-9.808, -73.008),
                    (-9.869, -73.463),
                    (-0.825, -161.307),
                ],
                vec![
                    (-0.756, -161.361),
                    (-9.948, -73.728),
                    (-9.931, -73.058),
                    (-0.713, -161.524),
                ],
                vec![
                    (-0.761, -162.046),
                    (-10.012, -74.283),
                    (-9.980, -73.262),
                    (-0.727, -162.222),
                ],
                vec![
                    (-0.760, -162.772),
                    (-10.097, -74.685),
                    (-10.090, -73.887),
                    (-0.778, -163.039),
                ],
                vec![
                    (-0.759, -163.211),
                    (-10.197, -74.987),
                    (-10.190, -74.236),
                    (-0.827, -163.492),
                ],
                vec![
                    (-0.743, -163.534),
                    (-10.288, -75.174),
                    (-10.306, -74.376),
                    (-0.845, -163.829),
                ],
                vec![
                    (-0.718, -164.022),
                    (-10.406, -75.414),
                    (-10.419, -74.530),
                    (-0.869, -164.190),
                ],
                vec![
                    (-0.713, -164.578),
                    (-10.501, -75.436),
                    (-10.514, -74.649),
                    (-0.882, -164.538),
                ],
                vec![
                    (-0.721, -164.938),
                    (-10.563, -75.321),
                    (-10.556, -74.375),
                    (-0.872, -164.822),
                ],
                vec![
                    (-0.740, -165.270),
                    (-10.616, -75.215),
                    (-10.615, -74.492),
                    (-0.882, -164.933),
                ],
                vec![
                    (-0.727, -165.532),
                    (-10.636, -75.164),
                    (-10.632, -74.424),
                    (-0.853, -165.150),
                ],
                vec![
                    (-0.704, -166.050),
                    (-10.623, -75.065),
                    (-10.634, -74.288),
                    (-0.849, -165.342),
                ],
                vec![
                    (-0.701, -166.646),
                    (-10.619, -75.230),
                    (-10.623, -74.467),
                    (-0.840, -165.459),
                ],
                vec![
                    (-0.696, -167.209),
                    (-10.617, -75.602),
                    (-10.618, -74.787),
                    (-0.809, -165.838),
                ],
                vec![
                    (-0.716, -167.810),
                    (-10.631, -75.931),
                    (-10.637, -75.128),
                    (-0.774, -166.238),
                ],
                vec![
                    (-0.741, -168.274),
                    (-10.612, -76.177),
                    (-10.622, -75.507),
                    (-0.745, -166.740),
                ],
                vec![
                    (-0.764, -168.664),
                    (-10.584, -76.727),
                    (-10.596, -76.030),
                    (-0.731, -167.341),
                ],
                vec![
                    (-0.765, -168.885),
                    (-10.590, -77.366),
                    (-10.609, -76.537),
                    (-0.708, -167.767),
                ],
                vec![
                    (-0.772, -169.145),
                    (-10.629, -77.899),
                    (-10.640, -77.188),
                    (-0.683, -168.348),
                ],
                vec![
                    (-0.750, -169.444),
                    (-10.667, -78.526),
                    (-10.673, -77.889),
                    (-0.677, -169.053),
                ],
                vec![
                    (-0.736, -169.760),
                    (-10.715, -79.048),
                    (-10.721, -78.526),
                    (-0.659, -169.607),
                ],
                vec![
                    (-0.729, -169.988),
                    (-10.774, -79.624),
                    (-10.776, -79.055),
                    (-0.645, -170.279),
                ],
                vec![
                    (-0.726, -170.132),
                    (-10.837, -80.185),
                    (-10.837, -79.496),
                    (-0.652, -170.933),
                ],
                vec![
                    (-0.700, -170.491),
                    (-10.903, -80.593),
                    (-10.902, -79.850),
                    (-0.681, -171.309),
                ],
                vec![
                    (-0.666, -170.890),
                    (-10.965, -81.065),
                    (-10.973, -80.269),
                    (-0.694, -171.561),
                ],
                vec![
                    (-0.659, -171.335),
                    (-11.040, -81.499),
                    (-11.050, -80.629),
                    (-0.707, -171.762),
                ],
                vec![
                    (-0.626, -171.867),
                    (-11.113, -81.675),
                    (-11.129, -80.926),
                    (-0.693, -172.083),
                ],
                vec![
                    (-0.607, -172.055),
                    (-11.179, -81.751),
                    (-11.191, -80.954),
                    (-0.685, -172.412),
                ],
                vec![
                    (-0.604, -172.623),
                    (-11.229, -82.010),
                    (-11.237, -81.134),
                    (-0.692, -172.867),
                ],
                vec![
                    (-0.581, -173.185),
                    (-11.233, -82.302),
                    (-11.249, -81.467),
                    (-0.727, -173.284),
                ],
                vec![
                    (-0.555, -173.548),
                    (-11.270, -82.562),
                    (-11.284, -81.615),
                    (-0.719, -173.825),
                ],
                vec![
                    (-0.529, -174.035),
                    (-11.295, -82.750),
                    (-11.294, -81.725),
                    (-0.699, -174.420),
                ],
                vec![
                    (-0.493, -174.425),
                    (-11.329, -82.919),
                    (-11.331, -81.878),
                    (-0.714, -174.634),
                ],
                vec![
                    (-0.474, -174.909),
                    (-11.354, -83.028),
                    (-11.362, -82.069),
                    (-0.722, -174.945),
                ],
                vec![
                    (-0.463, -175.232),
                    (-11.350, -83.348),
                    (-11.367, -82.421),
                    (-0.717, -175.277),
                ],
                vec![
                    (-0.432, -175.548),
                    (-11.368, -83.709),
                    (-11.363, -82.693),
                    (-0.739, -175.601),
                ],
                vec![
                    (-0.393, -176.181),
                    (-11.396, -84.163),
                    (-11.398, -83.231),
                    (-0.752, -175.840),
                ],
                vec![
                    (-0.385, -176.735),
                    (-11.445, -84.475),
                    (-11.452, -83.539),
                    (-0.736, -176.095),
                ],
                vec![
                    (-0.378, -177.418),
                    (-11.479, -84.621),
                    (-11.486, -83.766),
                    (-0.728, -176.458),
                ],
                vec![
                    (-0.357, -178.081),
                    (-11.505, -84.919),
                    (-11.501, -84.040),
                    (-0.745, -176.797),
                ],
                vec![
                    (-0.373, -178.550),
                    (-11.532, -85.329),
                    (-11.527, -84.379),
                    (-0.737, -177.162),
                ],
                vec![
                    (-0.398, -179.219),
                    (-11.555, -85.876),
                    (-11.537, -84.887),
                    (-0.712, -177.517),
                ],
                vec![
                    (-0.406, -179.387),
                    (-11.592, -86.304),
                    (-11.596, -85.116),
                    (-0.697, -178.130),
                ],
                vec![
                    (-0.388, -179.751),
                    (-11.608, -86.455),
                    (-11.617, -85.489),
                    (-0.708, -178.407),
                ],
                vec![
                    (-0.351, 179.972),
                    (-11.662, -86.823),
                    (-11.666, -85.790),
                    (-0.737, -178.606),
                ],
                vec![
                    (-0.309, 179.552),
                    (-11.715, -87.019),
                    (-11.706, -85.940),
                    (-0.744, -178.758),
                ],
                vec![
                    (-0.265, 179.038),
                    (-11.732, -87.209),
                    (-11.729, -86.175),
                    (-0.744, -178.952),
                ],
                vec![
                    (-0.268, 178.546),
                    (-11.731, -87.406),
                    (-11.728, -86.540),
                    (-0.754, -179.276),
                ],
                vec![
                    (-0.257, 178.084),
                    (-11.753, -87.778),
                    (-11.749, -87.009),
                    (-0.785, -179.361),
                ],
                vec![
                    (-0.252, 177.567),
                    (-11.757, -88.069),
                    (-11.774, -87.224),
                    (-0.760, -179.478),
                ],
                vec![
                    (-0.264, 177.133),
                    (-11.802, -88.337),
                    (-11.792, -87.409),
                    (-0.743, -179.786),
                ],
                vec![
                    (-0.284, 176.662),
                    (-11.827, -88.625),
                    (-11.825, -87.942),
                    (-0.737, 179.938),
                ],
                vec![
                    (-0.281, 176.114),
                    (-11.838, -88.888),
                    (-11.841, -88.009),
                    (-0.737, 179.650),
                ],
                vec![
                    (-0.284, 175.439),
                    (-11.849, -89.096),
                    (-11.828, -88.048),
                    (-0.699, 179.298),
                ],
                vec![
                    (-0.315, 174.679),
                    (-11.814, -89.502),
                    (-11.828, -88.608),
                    (-0.682, 178.625),
                ],
                vec![
                    (-0.346, 174.031),
                    (-11.835, -89.972),
                    (-11.824, -88.968),
                    (-0.720, 178.373),
                ],
                vec![
                    (-0.338, 173.956),
                    (-11.827, -90.109),
                    (-11.831, -89.305),
                    (-0.713, 178.598),
                ],
                vec![
                    (-0.273, 173.583),
                    (-11.855, -90.206),
                    (-11.862, -89.369),
                    (-0.648, 178.595),
                ],
                vec![
                    (-0.266, 173.005),
                    (-11.846, -90.457),
                    (-11.845, -89.570),
                    (-0.609, 178.294),
                ],
                vec![
                    (-0.232, 172.456),
                    (-11.843, -90.902),
                    (-11.855, -90.128),
                    (-0.612, 177.855),
                ],
                vec![
                    (-0.223, 172.030),
                    (-11.847, -91.056),
                    (-11.850, -90.397),
                    (-0.575, 177.768),
                ],
                vec![
                    (-0.227, 171.491),
                    (-11.866, -91.933),
                    (-11.857, -91.056),
                    (-0.561, 177.593),
                ],
                vec![
                    (-0.211, 170.829),
                    (-11.908, -92.002),
                    (-11.879, -91.092),
                    (-0.489, 177.305),
                ],
                vec![
                    (-0.233, 170.119),
                    (-11.866, -92.554),
                    (-11.893, -91.468),
                    (-0.476, 176.579),
                ],
                vec![
                    (-0.204, 169.606),
                    (-11.915, -92.825),
                    (-11.903, -92.017),
                    (-0.448, 176.373),
                ],
                vec![
                    (-0.235, 168.777),
                    (-11.933, -93.385),
                    (-11.912, -92.605),
                    (-0.420, 175.689),
                ],
                vec![
                    (-0.271, 167.832),
                    (-11.947, -93.891),
                    (-11.939, -93.206),
                    (-0.472, 175.246),
                ],
                vec![
                    (-0.306, 167.102),
                    (-11.957, -94.149),
                    (-11.954, -93.210),
                    (-0.448, 175.324),
                ],
                vec![
                    (-0.258, 166.964),
                    (-11.928, -94.426),
                    (-11.944, -93.481),
                    (-0.419, 175.523),
                ],
                vec![
                    (-0.229, 165.980),
                    (-11.950, -94.223),
                    (-11.941, -93.532),
                    (-0.339, 175.625),
                ],
                vec![
                    (-0.213, 165.242),
                    (-11.921, -94.718),
                    (-11.931, -93.936),
                    (-0.281, 175.172),
                ],
                vec![
                    (-0.198, 164.306),
                    (-11.915, -95.109),
                    (-11.905, -94.295),
                    (-0.217, 174.767),
                ],
                vec![
                    (-0.188, 163.445),
                    (-11.865, -95.670),
                    (-11.879, -94.820),
                    (-0.205, 174.577),
                ],
                vec![
                    (-0.181, 162.394),
                    (-11.874, -96.178),
                    (-11.869, -95.263),
                    (-0.135, 174.270),
                ],
                vec![
                    (-0.234, 161.079),
                    (-11.820, -96.781),
                    (-11.834, -95.848),
                    (-0.079, 173.670),
                ],
                vec![
                    (-0.311, 159.862),
                    (-11.797, -97.531),
                    (-11.792, -96.706),
                    (-0.026, 173.252),
                ],
                vec![
                    (-0.473, 158.035),
                    (-11.767, -99.069),
                    (-11.777, -98.139),
                    (0.031, 172.884),
                ],
                vec![
                    (-0.865, 156.756),
                    (-11.916, -100.603),
                    (-11.916, -99.695),
                    (0.055, 172.980),
                ],
                vec![
                    (-1.383, 157.170),
                    (-12.183, -101.839),
                    (-12.185, -100.941),
                    (0.148, 172.980),
                ],
                vec![
                    (-1.786, 159.984),
                    (-12.561, -101.601),
                    (-12.577, -100.705),
                    (0.301, 173.255),
                ],
                vec![
                    (-1.716, 163.563),
                    (-12.816, -99.966),
                    (-12.805, -98.952),
                    (0.502, 172.921),
                ],
                vec![
                    (-1.393, 165.231),
                    (-12.859, -97.814),
                    (-12.853, -96.821),
                    (0.657, 171.744),
                ],
                vec![
                    (-1.030, 165.580),
                    (-12.700, -96.201),
                    (-12.669, -95.453),
                    (0.747, 170.716),
                ],
                vec![
                    (-0.759, 164.875),
                    (-12.514, -95.622),
                    (-12.508, -94.678),
                    (0.780, 169.512),
                ],
                vec![
                    (-0.585, 164.044),
                    (-12.361, -95.610),
                    (-12.335, -94.626),
                    (0.755, 168.895),
                ],
                vec![
                    (-0.439, 163.431),
                    (-12.203, -95.395),
                    (-12.209, -94.420),
                    (0.799, 168.418),
                ],
                vec![
                    (-0.318, 162.847),
                    (-12.067, -95.224),
                    (-12.068, -94.311),
                    (0.841, 167.973),
                ],
            ],
            String::from("test_3"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:48 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Array1 = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Vg(V) = -1.500\n\
            Vg(mA) = -0.000\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n",
            ),
        );
        comp_array_f64(
            &exemplar,
            &calc.passivity(),
            F64Margin::default(),
            "passivity5()",
        );
        assert_eq!(false, calc.is_passive());
    }

    #[test]
    fn network_ports() {
        let exemplar: Array1<(usize, usize)> = array![
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ];

        let exemplar_ne: Array1<(usize, usize)> = array![(0, 0), (0, 1), (1, 0), (1, 1)];

        let mut calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0), c64::from(50.0),],
            RFParameter::S,
            Points::new(array![[
                [c64(0.958, -0.263), c64(-0.846, 0.158), c64(1.473, 0.230)],
                [c64(0.004, 0.022), c64(0.544, -0.129), c64(-30.321, 4.378)],
                [c64(4.234, 84.212), c64(0.457, 0.287), c64(3.489, -2.893)],
            ]]),
            String::from(""),
            String::from(""),
        );

        assert_eq!(&exemplar, calc.ports());
        assert_ne!(&exemplar_ne, calc.ports());
        assert_eq!("", calc.port_name(0));
        assert_eq!("", calc.port_name(1));
        assert_eq!("", calc.port_name(2));

        calc.set_port_names(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
        assert_eq!("1", calc.port_name(0));
        assert_eq!("2", calc.port_name(1));
        assert_eq!("3", calc.port_name(2));

        calc.set_port_name(1, "port2".to_string());
        assert_eq!("1", calc.port_name(0));
        assert_eq!("port2", calc.port_name(1));
        assert_eq!("3", calc.port_name(2));
    }

    #[test]
    fn network_s_db() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_db = Pointsf64::new(array![
            [
                [
                    -0.0571232931409698157602471654702417,
                    -1.3036938210270008034803505896139
                ],
                [
                    -33.0102999566398124343011222238879,
                    -5.05042981341051474077798143321121
                ],
            ],
            [
                [
                    7.10808722678577439452626820799157,
                    10.2582363703478204445299539985994
                ],
                [
                    11.5269507556086664924691854963622,
                    51.9169985529863755091065898539251
                ],
            ],
            [
                [
                    26.5811015830929560104699014097416,
                    27.7122202597899446941970466073987
                ],
                [
                    -14.3808805387243224903439848474195,
                    -6.71178171541068598914939921950404
                ],
            ],
        ]);

        comp_points_c64(
            &exemplar_s,
            &calc.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
        comp_points_f64(
            &exemplar_s_db,
            &calc.net_db(RFParameter::S),
            F64Margin::default(),
            "net_db(S)",
        );
        comp_points_c64(&exemplar_s, &calc.s(), F64Margin::default(), "s()");
        comp_points_f64(&exemplar_s_db, &calc.s_db(), F64Margin::default(), "s_db()");
    }

    #[test]
    fn network_s_deg() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_deg = Pointsf64::new(array![
            [
                [
                    -15.351227009735157493813896682283,
                    169.42124106201960483185916007312
                ],
                [
                    79.6951535312339672595779107954925,
                    -13.3402769069429650908860377064099
                ],
            ],
            [
                [
                    -25.6719967136523418134102583838897,
                    94.1896222142384810674002307865579
                ],
                [
                    112.142888514762848850576625256636,
                    -89.9821278079241522410312200526232
                ],
            ],
            [
                [
                    -1.1307792814754272842611967996569,
                    92.2216280653444960010340518233173
                ],
                [
                    43.7269699799432878929141211369787,
                    -44.5612966323272438891616918249177
                ],
            ],
        ]);

        comp_points_c64(
            &exemplar_s,
            &calc.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
        comp_points_f64(
            &exemplar_s_deg,
            &calc.net_deg(RFParameter::S),
            F64Margin::default(),
            "net_deg(S)",
        );
        comp_points_c64(&exemplar_s, &calc.s(), F64Margin::default(), "s()");
        comp_points_f64(
            &exemplar_s_deg,
            &calc.s_deg(),
            F64Margin::default(),
            "s_deg()",
        );
    }

    #[test]
    fn network_s_im() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_im = Pointsf64::new(array![
            [[-0.263, 0.158], [0.022, -0.129],],
            [[-0.982, 3.249], [3.492, -394.321],],
            [[-0.421, 24.282], [0.132, -0.324],],
        ]);

        comp_points_c64(
            &exemplar_s,
            &calc.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
        comp_points_f64(
            &exemplar_s_im,
            &calc.net_im(RFParameter::S),
            F64Margin::default(),
            "net_im(S)",
        );
        comp_points_c64(&exemplar_s, &calc.s(), F64Margin::default(), "s()");
        comp_points_f64(&exemplar_s_im, &calc.s_im(), F64Margin::default(), "s_im()");
    }

    #[test]
    fn network_s_mag() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_mag = Pointsf64::new(array![
            [
                [
                    0.993445016092989383214754191494358,
                    0.8606276779188547320363076992136
                ],
                [
                    0.0223606797749978957228246600636754,
                    0.559085861026729993118859201916716
                ],
            ],
            [
                [
                    2.26675384636267918218365739653485,
                    3.25770548085611949025013186816706
                ],
                [
                    3.7700537131452119493274643947394,
                    394.321019183608852673148079774108
                ],
            ],
            [
                [
                    21.3331545252923161462373382375707,
                    24.3002651837382219396266802638806
                ],
                [
                    0.190965965554074595232556839140833,
                    0.461754263651132967274505211463413
                ],
            ],
        ]);

        comp_points_c64(
            &exemplar_s,
            &calc.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
        comp_points_f64(
            &exemplar_s_mag,
            &calc.net_mag(RFParameter::S),
            F64Margin::default(),
            "net_mag(S)",
        );
        comp_points_c64(&exemplar_s, &calc.s(), F64Margin::default(), "s()");
        comp_points_f64(
            &exemplar_s_mag,
            &calc.s_mag(),
            F64Margin::default(),
            "s_mag()",
        );
    }

    #[test]
    fn network_s_rad() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_rad = Pointsf64::new(array![
            [
                [
                    -0.267929455540962111948947391268623,
                    2.95695847934725672395589308518909
                ],
                [
                    1.39094282700241833476498088656927,
                    -0.232831755153919938547207556795139
                ],
            ],
            [
                [
                    -0.448060868214397288426676208406736,
                    1.64391902884805336903629339885268
                ],
                [
                    1.9572626372795453631888787638813,
                    -1.57048439819862423569595683285167
                ],
            ],
            [
                [
                    -0.0197358215750819296148809534818378,
                    1.60957105128986980679885216395864
                ],
                [
                    0.763179598070729232409291077103769,
                    -0.7777413451919714619155793725139
                ],
            ],
        ]);

        comp_points_c64(
            &exemplar_s,
            &calc.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
        comp_points_f64(
            &exemplar_s_rad,
            &calc.net_rad(RFParameter::S),
            F64Margin::default(),
            "net_rad(S)",
        );
        comp_points_c64(&exemplar_s, &calc.s(), F64Margin::default(), "s()");
        comp_points_f64(
            &exemplar_s_rad,
            &calc.s_rad(),
            F64Margin::default(),
            "s_rad()",
        );
    }

    #[test]
    fn network_s_re() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_re = Pointsf64::new(array![
            [[0.958, -0.846], [0.004, 0.544],],
            [[2.043, -0.238], [-1.421, 0.123],],
            [[21.329, -0.942], [0.138, 0.329],],
        ]);

        comp_points_c64(
            &exemplar_s,
            &calc.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
        comp_points_f64(
            &exemplar_s_re,
            &calc.net_re(RFParameter::S),
            F64Margin::default(),
            "net_re(S)",
        );
        comp_points_c64(&exemplar_s, &calc.s(), F64Margin::default(), "s()");
        comp_points_f64(&exemplar_s_re, &calc.s_re(), F64Margin::default(), "s_re()");
    }

    #[test]
    fn network_s_to_1port() {
        let param = RFParameter::S;

        // 1 PortVal
        let exemplar = points![[[c64(0.958, -0.263)]]];

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_s = exemplar;

        let exemplar_y = points![[[c64(6.695989913226831e-5, 0.0026954088117833435)]]];

        let exemplar_z = points![[[c64(9.210804562051559, -370.77241904332254)]]];

        let margin = MARGIN;
        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    fn network_s_to_2port() {
        let param = RFParameter::S;
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

        // 2 PortVal, 3 Frequncy
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = Points::new(array![
            [
                [
                    c64(6.202488000000001, -19.779434),
                    c64(-105.52439999999999, -3423.8283)
                ],
                [
                    c64(0.062430240000000005, 0.014948679999999995),
                    c64(8.786488000000002, -0.49143400000000115)
                ],
            ],
            [
                [
                    c64(127.68634379565486, -106.20979554227536),
                    c64(-6411.591799734475, 5278.017618105008)
                ],
                [
                    c64(-0.6179030615363564, 1.3314279726354992),
                    c64(31.240691781573684, -66.41332603711804)
                ],
            ],
            [
                [
                    c64(40.7174714544258, -2.0678821158275724),
                    c64(2112.736167324778, -4017.3775296150047)
                ],
                [
                    c64(-0.7401111928265877, 0.017303677196446158),
                    c64(-38.39835612043435, 74.3110254195459)
                ],
            ],
        ]);

        let exemplar_g = Points::new(array![
            [
                [
                    c64(2.130487608213253e-4, 0.003089512451043385),
                    c64(1.768990320428292, -0.5640273242688661)
                ],
                [
                    c64(0.014434566504766778, 0.04603113387718689),
                    c64(156.07949988012558, -54.278885080962404)
                ],
            ],
            [
                [
                    c64(-0.007986727642840415, 0.0037839540883934483),
                    c64(-0.004830664666515159, -0.0019321765151291675)
                ],
                [
                    c64(0.004628946068864235, 0.003850368026334074),
                    c64(-50.00122293572251, -0.25532915888001423)
                ],
            ],
            [
                [
                    c64(-0.018151513013219368, -4.968754496812585e-4),
                    c64(-1.9471381808274046, -2.4393116449214203)
                ],
                [
                    c64(0.024496300246171044, 0.0012440718780805742),
                    c64(56.75212590395926, -95.78249051609758)
                ],
            ],
        ]);

        let exemplar_h = Points::new(array![
            [
                [
                    c64(9.754118397254564, -389.12416366807344),
                    c64(-0.2233565554706719, 4.3678697329944365)
                ],
                [
                    c64(-0.11345619746535836, -0.006345679063715907),
                    c64(0.006988238111543526, 0.002092182656837212)
                ],
            ],
            [
                [
                    c64(-102.25781538984043, -48.43887663577415),
                    c64(0.008051284343097117, 0.008590117728274544)
                ],
                [
                    c64(-0.0057995859683741295, -0.012329105785842618),
                    c64(-0.01999889824635471, 1.0353877691989483e-4)
                ],
            ],
            [
                [
                    c64(-54.26394074814746, -0.39146339072870934),
                    c64(0.5493477836957078, -1.4186428387747134)
                ],
                [
                    c64(0.005488174726287244, 0.010621076858418059),
                    c64(0.004245643228549894, 0.007765812258925287)
                ],
            ],
        ]);

        let exemplar_s = exemplar;

        let exemplar_t = Points::new(array![
            [
                [
                    c64(6.988976000000001, 23.729132000000003),
                    c64(-3.9080000000000004, -44.256)
                ],
                [
                    c64(1.3239999999999996, 24.968000000000004),
                    c64(8.000000000000002, -44.0)
                ],
            ],
            [
                [
                    c64(159.02701232436794, -172.37743628663426),
                    c64(-0.4455154518952489, -0.4037578874160514)
                ],
                [
                    c64(96.89116746597642, -39.39271161774127),
                    c64(-0.09997674713938806, -0.24568529275914364)
                ],
            ],
            [
                [
                    c64(-1.4650241855873656, 75.86275501809806),
                    c64(79.18805528134254, -78.79582099374794)
                ],
                [
                    c64(-0.07222770648239558, 2.416913458374465),
                    c64(3.7841395195788086, -3.61961171437973)
                ],
            ],
        ]);

        let exemplar_y = Points::new(array![
            [
                [
                    c64(6.437819859727872e-5, 0.0025682600587126577),
                    c64(0.011232204669595628, 2.924421351492062e-4)
                ],
                [
                    c64(8.993248472233786e-6, -2.917935437506947e-4),
                    c64(0.005715730624512929, 0.001987727318193308)
                ],
            ],
            [
                [
                    c64(-0.007987028866551884, 0.0037834047644999032),
                    c64(9.680573280190289e-5, 3.814825071879078e-5)
                ],
                [
                    c64(9.296745811563372e-5, 7.653074262542338e-5),
                    c64(-0.019998989345705667, 1.0212400473991743e-4)
                ],
            ],
            [
                [
                    c64(-0.01842748510783592, 1.3293700574377318e-4),
                    c64(0.00993450797186877, -0.026215048434335596)
                ],
                [
                    c64(-1.0254519219319226e-4, -1.9499015412256825e-4),
                    c64(0.004578597588387434, 0.007727454665379356)
                ],
            ],
        ]);

        let exemplar_z = Points::new(array![
            [
                [
                    c64(22.214615781667156, -322.14377491810205),
                    c64(142.4004511071262, 582.3988699153475)
                ],
                [
                    c64(15.14930157983583, -3.6274417900757743),
                    c64(131.3265083109445, -39.31735563178437)
                ],
            ],
            [
                [
                    c64(-102.25481599514548, -48.44631573102483),
                    c64(-0.4003518931087395, -0.43160225964937365)
                ],
                [
                    c64(-0.28679588343874085, -0.6179740568004651),
                    c64(-50.00141431550363, -0.25886852459163934)
                ],
            ],
            [
                [
                    c64(-55.050575493627136, 1.5069421173698982),
                    c64(-110.86697887528577, -131.3512853282026)
                ],
                [
                    c64(-1.3504101703265112, -0.031572366283079384),
                    c64(54.19970554227091, -99.13803752984421)
                ],
            ],
        ]);

        let mut exemplars: HashMap<RFParameter, Points> = HashMap::new();
        exemplars.insert(RFParameter::A, exemplar_a);
        exemplars.insert(RFParameter::G, exemplar_g);
        exemplars.insert(RFParameter::H, exemplar_h);
        exemplars.insert(RFParameter::S, exemplar_s.clone());
        exemplars.insert(RFParameter::SPower, exemplar_s.clone());
        exemplars.insert(RFParameter::SPseudo, exemplar_s.clone());
        exemplars.insert(RFParameter::STraveling, exemplar_s.clone());
        exemplars.insert(RFParameter::T, exemplar_t);
        exemplars.insert(RFParameter::Y, exemplar_y);
        exemplars.insert(RFParameter::Z, exemplar_z);

        compare_2ports(&calc, exemplars);
    }

    #[test]
    fn network_s_to_3port() {
        let param = RFParameter::S;

        let exemplar = Points::new(array![[
            [c64(0.958, -0.263), c64(-0.846, 0.158), c64(1.473, 0.230)],
            [c64(0.004, 0.022), c64(0.544, -0.129), c64(-30.321, 4.378)],
            [c64(4.234, 84.212), c64(0.457, 0.287), c64(3.489, -2.893)],
        ]]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_s = exemplar;

        let exemplar_y = Points::new(array![[
            [
                c64(-0.01979363538912805, -3.679768738843324e-4),
                c64(-1.3997430248235537e-5, -9.099790614032492e-5),
                c64(3.282222069630684e-5, -4.696694148470769e-4)
            ],
            [
                c64(-0.04869282181919289, -0.011977978173420245),
                c64(-0.02214838809265807, -0.0014508726872149143),
                c64(1.8537746629572412e-4, -0.001159959857770211)
            ],
            [
                c64(-0.0024215654928461455, -7.523224794308155e-4),
                c64(-0.0013962656575253964, -2.663674795964231e-4),
                c64(-0.019986777807612287, -5.7984945881732476e-5)
            ],
        ]]);

        let exemplar_z = Points::new(array![[
            [
                c64(-50.53068434585749, 0.4419235157446797),
                c64(0.06033794406037343, 0.12934254686112553),
                c64(-0.06108950957660311, 1.1860216769506107)
            ],
            [
                c64(112.51442557969453, 19.05407191039559),
                c64(-45.00350038086035, 2.4680923131099726),
                c64(0.3584149111093307, 0.021004509325415748)
            ],
            [
                c64(-1.4702476888594347, -0.9778592208137421),
                c64(3.175532470820473, 0.40019548882166295),
                c64(-50.00579897879361, -0.002565924324619274)
            ],
        ]]);

        let margin = MARGIN;
        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_s_to_a() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::S,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.a();
    }

    #[test]
    #[should_panic(expected = "G parameters do not exist for network with 1 port(s)")]
    fn network_s_to_g() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::S,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.g();
    }

    #[test]
    #[should_panic(expected = "H parameters do not exist for network with 1 port(s)")]
    fn network_s_to_h() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::S,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.h();
    }

    #[test]
    #[should_panic(expected = "T parameters do not exist for network with 1 port(s)")]
    fn network_s_to_t() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::S,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.t();
    }

    #[test]
    fn network_t_to_2port() {
        let param = RFParameter::T;
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = Points::new(array![
            [
                [
                    c64(0.33, -0.10600000000000001),
                    c64(-31.599999999999998, 6.75)
                ],
                [
                    c64(0.00436, -1.9999999999999947e-5),
                    c64(1.172, -0.28600000000000003)
                ],
            ],
            [
                [
                    c64(0.25350000000000006, -194.281),
                    c64(-18.425000000000004, -9839.550000000001)
                ],
                [
                    c64(-0.031030000000000002, -3.9309600000000002),
                    c64(1.9125, -201.02200000000002)
                ],
            ],
            [
                [c64(10.427, 11.8345), c64(-552.0, 606.175)],
                [c64(-0.19920000000000002, -0.24053), c64(11.231, -12.5795)],
            ],
        ]);

        let exemplar_g = Points::new(array![
            [
                [
                    c64(0.011994073383498702, 0.0037920356928814016),
                    c64(-1.5766089598455084, 0.24713166744356396)
                ],
                [
                    c64(2.746886861556902, 0.8823333555303988),
                    c64(-92.75737497502828, -9.340247719251511)
                ],
            ],
            [
                [
                    c64(0.020233130461588148, -1.8611752344291309e-4),
                    c64(-4.116608106547478, 1.940530382049762)
                ],
                [
                    c64(6.716090874263961e-6, 0.005147174955198724),
                    c64(50.64576158645125, -0.16092001051140048)
                ],
            ],
            [
                [
                    c64(-0.019791204372495267, -6.052548051889096e-4),
                    c64(0.06063514515277476, 0.9166673419669595)
                ],
                [
                    c64(0.04191291740849899, -0.04757057840902285),
                    c64(5.700164957597985, 51.6655219918775)
                ],
            ],
        ]);

        let exemplar_h = Points::new(array![
            [
                [
                    c64(-26.77355742142945, -0.7740933639324433),
                    c64(0.44674819222471107, -0.10316042408168315)
                ],
                [
                    c64(-0.8052879660294906, -0.1965122510959337),
                    c64(0.003514985776910498, 8.406876554576813e-4)
                ],
            ],
            [
                [
                    c64(48.94232564709039, -0.5572882460629204),
                    c64(3.9628581685727124, -1.9079682285888873)
                ],
                [
                    c64(-4.732329292727238e-5, -0.004974129668405829),
                    c64(0.019551636319537043, -3.403732151760235e-4)
                ],
            ],
            [
                [
                    c64(-48.61428446249878, -0.4779976311996644),
                    c64(0.8580073053026979, 0.04608903010019478)
                ],
                [
                    c64(-0.03949304569565779, -0.044234953995951135),
                    c64(0.002772818782071094, -0.018310865117170037)
                ],
            ],
        ]);

        let exemplar_s = Points::new(array![
            [
                [
                    c64(-1.5375603451309596, -0.07416412595936357),
                    c64(0.9625186306094178, -0.22887701590328144)
                ],
                [
                    c64(1.7403711725430853, 0.41269831113613603),
                    c64(0.0021178781548226505, -0.03993895904049242)
                ],
            ],
            [
                [
                    c64(-0.008239667486325092, -6.00998985342353e-4),
                    c64(2.0291927440451167, -0.9540811006959242)
                ],
                [
                    c64(7.910524066400982e-7, 0.002536004683241709),
                    c64(0.008856852439349883, 0.0036009002998824815)
                ],
            ],
            [
                [
                    c64(-38.351941918327334, 36.03638546644967),
                    c64(31.378370866300532, -0.3315648611508466)
                ],
                [
                    c64(1.5430289329650075, 1.5195786452299769),
                    c64(-0.012353611578814082, -0.41338167219311783)
                ],
            ],
        ]);

        let exemplar_t = exemplar;

        let exemplar_y = Points::new(array![
            [
                [
                    c64(-0.03731908851691253, 0.0010789921680645705),
                    c64(0.01656092604076629, -0.004331890798254036)
                ],
                [
                    c64(0.03026464806572026, 0.006464758684924424),
                    c64(-0.010672598282289674, 0.001074682328941288)
                ],
            ],
            [
                [
                    c64(0.02042956374064228, 2.326239220209597e-4),
                    c64(-0.0814033026024069, 0.03805710293149084)
                ],
                [
                    c64(1.903073228305826e-7, -1.0163030764492043e-4),
                    c64(0.01974478955665645, 6.273637997483575e-5)
                ],
            ],
            [
                [
                    c64(-0.020568097284233762, 2.0223483465507019e-4),
                    c64(0.017656898533430877, 7.74444689316051e-4)
                ],
                [
                    c64(8.212426545263248e-4, 9.018419675860415e-4),
                    c64(0.0021097516066510193, -0.019122502391011445)
                ],
            ],
        ]);

        let exemplar_z = Points::new(array![
            [
                [
                    c64(75.79800105207785, -23.96422935297212),
                    c64(113.5814876380852, -56.51430510257759)
                ],
                [
                    c64(229.35297211993685, 1.0520778537611755),
                    c64(269.1025775907417, -64.36191478169384)
                ],
            ],
            [
                [
                    c64(49.419707635510186, 0.4545946862674464),
                    c64(204.32372387576785, -94.02905596804213)
                ],
                [
                    c64(-0.0020079711364625785, 0.2543749345339651),
                    c64(51.131117845088255, 0.8901384360901887)
                ],
            ],
            [
                [
                    c64(-50.480283855814946, 1.5437885333291603),
                    c64(4.476019870356524, 46.18001978204391)
                ],
                [
                    c64(-2.0423370545346717, 2.4660809825663885),
                    c64(8.084578260714984, 53.38813449272201)
                ],
            ],
        ]);

        let mut exemplars: HashMap<RFParameter, Points> = HashMap::new();
        exemplars.insert(RFParameter::A, exemplar_a);
        exemplars.insert(RFParameter::G, exemplar_g);
        exemplars.insert(RFParameter::H, exemplar_h);
        exemplars.insert(RFParameter::S, exemplar_s.clone());
        exemplars.insert(RFParameter::SPower, exemplar_s.clone());
        exemplars.insert(RFParameter::SPseudo, exemplar_s.clone());
        exemplars.insert(RFParameter::STraveling, exemplar_s.clone());
        exemplars.insert(RFParameter::T, exemplar_t);
        exemplars.insert(RFParameter::Y, exemplar_y);
        exemplars.insert(RFParameter::Z, exemplar_z);

        compare_2ports(&calc, exemplars);
    }

    #[test]
    fn network_y_to_1port() {
        let param = RFParameter::Y;

        let exemplar = Points::new(array![[[c64(0.958, -0.263)]]]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_s = Points::new(array![[[c64(-0.9618584453026511, 0.010256880250923072)]]]);

        let exemplar_y = exemplar;

        let exemplar_z = Points::new(array![[[c64(0.9706839268724422, 0.26648212188669346)]]]);

        let margin = MARGIN;
        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    fn network_y_to_2port() {
        let param = RFParameter::Y;
        // 2 PortVal, 3 Frequncy
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = Points::new(array![
            [
                [
                    c64(1.3239999999999996, 24.968000000000004),
                    c64(-8.000000000000002, 44.0)
                ],
                [
                    c64(6.988976000000001, 23.729132000000003),
                    c64(3.9080000000000004, 44.256)
                ],
            ],
            [
                [
                    c64(96.89116746597642, -39.39271161774127),
                    c64(0.09997674713938806, 0.24568529275914364)
                ],
                [
                    c64(159.02701232436794, -172.37743628663426),
                    c64(0.4455154518952489, 0.4037578874160514)
                ],
            ],
            [
                [
                    c64(-0.07222770648239558, 2.416913458374465),
                    c64(-3.7841395195788086, 3.61961171437973)
                ],
                [
                    c64(-1.4650241855873656, 75.86275501809806),
                    c64(-79.18805528134254, 78.79582099374794)
                ],
            ],
        ]);

        let exemplar_g = Points::new(array![
            [
                [
                    c64(0.9625186306094178, -0.22887701590328144),
                    c64(-1.5375603451309596, -0.07416412595936357)
                ],
                [
                    c64(0.0021178781548226505, -0.03993895904049242),
                    c64(1.7403711725430853, 0.41269831113613603)
                ],
            ],
            [
                [
                    c64(2.0291927440451167, -0.9540811006959242),
                    c64(-0.008239667486325092, -6.00998985342353e-4)
                ],
                [
                    c64(0.008856852439349883, 0.0036009002998824815),
                    c64(7.910524066400982e-7, 0.002536004683241709)
                ],
            ],
            [
                [
                    c64(31.378370866300532, -0.3315648611508466),
                    c64(-38.351941918327334, 36.03638546644967)
                ],
                [
                    c64(-0.012353611578814082, -0.41338167219311783),
                    c64(1.5430289329650075, 1.5195786452299769)
                ],
            ],
        ]);

        let exemplar_h = Points::new(array![
            [
                [
                    c64(0.9706839268724422, 0.26648212188669346),
                    c64(0.8633027773921836, 0.07207581467029678)
                ],
                [
                    c64(-0.001979870974017487, 0.0224209748787405),
                    c64(0.5458675431868223, -0.10971903563869077)
                ],
            ],
            [
                [
                    c64(0.39761214735276523, 0.1911185162508152),
                    c64(0.7155757503688567, -1.2463556598814403)
                ],
                [
                    c64(-1.2323927201361262, 1.1168822069634479),
                    c64(3.4584408230318444, -390.05113808702043)
                ],
            ],
            [
                [
                    c64(0.04686626414341519, 9.25064335148286e-4),
                    c64(0.06661043300916779, -1.1371352153266978)
                ],
                [
                    c64(0.006345435959551723, 0.006314005745181268),
                    c64(0.4882940881783893, -0.4721320825578742)
                ],
            ],
        ]);

        let exemplar_s = Points::new(array![
            [
                [
                    c64(-0.9614083414074007, 0.009035414350549562),
                    c64(0.05681907427865709, 0.015737835221420228)
                ],
                [
                    c64(4.159505533221463e-4, -0.0014742917111007538),
                    c64(-0.931897386653905, 0.01324869884154277)
                ],
            ],
            [
                [
                    c64(-0.9839576651419523, 0.007469120090701472),
                    c64(1.2769817947395006e-4, 7.117801494528487e-5)
                ],
                [
                    c64(1.1519531894791628e-4, 1.2391389821847143e-4),
                    c64(-0.9999990884585707, 1.0253038106212201e-4)
                ],
            ],
            [
                [
                    c64(-0.9987139933278567, 2.5705213831050958e-5),
                    c64(0.04740456047838831, -0.04539691148010188)
                ],
                [
                    c64(-2.0298357760962746e-5, -5.154064989860888e-4),
                    c64(-0.9577418866259986, 0.039252150528032284)
                ],
            ],
        ]);

        let exemplar_t = Points::new(array![
            [
                [
                    c64(-172.0284, -559.0563000000001),
                    c64(-176.09640000000002, -602.4323)
                ],
                [c64(173.5124, 583.1443), c64(177.2604, 628.2803)],
            ],
            [
                [
                    c64(-3927.0079664177338, 4289.938973447766),
                    c64(-3927.4514823346863, 4289.540129266205)
                ],
                [
                    c64(4023.8971343487674, -4329.336598771363),
                    c64(4024.3446493356055, -4328.9279271780915)
                ],
            ],
            [
                [
                    c64(-2.9666954590325436, -1855.998704343534),
                    c64(76.14567703191842, -1934.7221331029943)
                ],
                [
                    c64(2.9701505429417243, 1858.343225567621),
                    c64(-76.2935875287924, 1937.2114387956565)
                ],
            ],
        ]);

        let exemplar_y = exemplar;

        let exemplar_z = Points::new(array![
            [
                [
                    c64(0.9833390626156057, 0.2338279001727904),
                    c64(1.4946015066547766, 0.43245298899150947)
                ],
                [
                    c64(0.011421435247022799, -0.03877831954295688),
                    c64(1.760808278638465, 0.3539213655183655)
                ],
            ],
            [
                [
                    c64(0.4035870224366391, 0.18975760273287182),
                    c64(0.0032113787399704777, 0.001806094940502799)
                ],
                [
                    c64(0.002891212495572124, 0.0031339317167705934),
                    c64(2.2730192220551214e-5, 0.0025635648542889945)
                ],
            ],
            [
                [
                    c64(0.031865528080802376, 3.3671249022544614e-4),
                    c64(1.2342387832408936, -1.1354048751434689)
                ],
                [
                    c64(-2.5446358440632175e-4, -0.013176784898676817),
                    c64(1.058426060538316, 1.0233933039404715)
                ],
            ],
        ]);

        let mut exemplars: HashMap<RFParameter, Points> = HashMap::new();
        exemplars.insert(RFParameter::A, exemplar_a);
        exemplars.insert(RFParameter::G, exemplar_g);
        exemplars.insert(RFParameter::H, exemplar_h);
        exemplars.insert(RFParameter::S, exemplar_s.clone());
        exemplars.insert(RFParameter::SPower, exemplar_s.clone());
        exemplars.insert(RFParameter::SPseudo, exemplar_s.clone());
        exemplars.insert(RFParameter::STraveling, exemplar_s.clone());
        exemplars.insert(RFParameter::T, exemplar_t);
        exemplars.insert(RFParameter::Y, exemplar_y);
        exemplars.insert(RFParameter::Z, exemplar_z);

        compare_2ports(&calc, exemplars);
    }

    #[test]
    fn network_y_to_3port() {
        let param = RFParameter::Y;

        // 3 PortVal
        let exemplar = Points::new(array![[
            [c64(0.958, -0.263), c64(-0.846, 0.158), c64(1.473, 0.230)],
            [c64(0.004, 0.022), c64(0.544, -0.129), c64(-30.321, 4.378)],
            [c64(4.234, 84.212), c64(0.457, 0.287), c64(3.489, -2.893)],
        ]]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_s = Points::new(array![[
            [
                c64(-0.9998020654843073, -2.530137423705384e-4),
                c64(-2.010724204231428e-5, -7.093267722797153e-5),
                c64(2.7300999728120196e-5, -4.720929337281659e-4)
            ],
            [
                c64(-0.0468818126822702, -0.00954057185948405),
                c64(-1.0020893279268488, -0.0011866636239758967),
                c64(-6.135645815505928e-6, -5.743577190740908e-4)
            ],
            [
                c64(-8.969201850864128e-4, -1.0740090230100282e-4),
                c64(-0.001333375738284437, -2.0573200269127044e-4),
                c64(-1.0000006530163605, -1.079426505110717e-5)
            ],
        ]]);

        let exemplar_y = exemplar;

        let exemplar_z = Points::new(array![[
            [
                c64(0.0049505482271746445, -0.006277586868079991),
                c64(-5.05139977548531e-4, -0.0017634882308923848),
                c64(6.805419547393476e-4, -0.011803426938288317)
            ],
            [
                c64(-1.1711093979459362, -0.23744162620243292),
                c64(-0.05219414542656583, -0.029551423941518985),
                c64(-2.338269177300081e-4, -0.014070662368167343)
            ],
            [
                c64(-0.021669227998620977, -0.002403588627800365),
                c64(-0.03330252100014572, -0.005117230099113714),
                c64(-1.8557328737234102e-5, -2.55194829519355e-4)
            ],
        ]]);

        let margin = MARGIN;
        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_y_to_a() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Y,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.a();
    }

    #[test]
    #[should_panic(expected = "G parameters do not exist for network with 1 port(s)")]
    fn network_y_to_g() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Y,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.g();
    }

    #[test]
    #[should_panic(expected = "H parameters do not exist for network with 1 port(s)")]
    fn network_y_to_h() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Y,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.h();
    }

    #[test]
    #[should_panic(expected = "T parameters do not exist for network with 1 port(s)")]
    fn network_y_to_t() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Y,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.t();
    }

    #[test]
    fn network_z_to_1port() {
        let param = RFParameter::Z;

        // 1 PortVal
        let exemplar = Points::new(array![[[c64(0.958, -0.263)]]]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_s = Points::new(array![[[c64(-0.9623481369389654, -0.010127900624336668)]]]);

        let exemplar_y = Points::new(array![[[c64(0.9706839268724422, 0.26648212188669346)]]]);

        let exemplar_z = exemplar;

        let margin = MARGIN;
        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    fn network_z_to_2port() {
        let param = RFParameter::Z;
        // 2 PortVal, 3 Frequncy
        let exemplar = Points::new(array![
            [
                [c64(0.958, -0.263), c64(-0.846, 0.158)],
                [c64(0.004, 0.022), c64(0.544, -0.129)],
            ],
            [
                [c64(2.043, -0.982), c64(-0.238, 3.249)],
                [c64(-1.421, 3.492), c64(0.123, -394.321)],
            ],
            [
                [c64(21.329, -0.421), c64(-0.942, 24.282)],
                [c64(0.138, 0.132), c64(0.329, -0.324)],
            ],
        ]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0, 2.0, 3.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_a = Points::new(array![
            [
                [
                    c64(-3.9080000000000004, -44.256),
                    c64(-6.988976000000001, -23.729132000000003)
                ],
                [
                    c64(8.000000000000002, -44.0),
                    c64(-1.3239999999999996, -24.968000000000004)
                ],
            ],
            [
                [
                    c64(-0.4455154518952489, -0.4037578874160514),
                    c64(-159.02701232436794, 172.37743628663426)
                ],
                [
                    c64(-0.09997674713938806, -0.24568529275914364),
                    c64(-96.89116746597642, 39.39271161774127)
                ],
            ],
            [
                [
                    c64(79.18805528134254, -78.79582099374794),
                    c64(1.4650241855873656, -75.86275501809806)
                ],
                [
                    c64(3.7841395195788086, -3.61961171437973),
                    c64(0.07222770648239558, -2.416913458374465)
                ],
            ],
        ]);

        let exemplar_g = Points::new(array![
            [
                [
                    c64(0.9706839268724422, 0.26648212188669346),
                    c64(0.8633027773921836, 0.07207581467029678)
                ],
                [
                    c64(-0.001979870974017487, 0.0224209748787405),
                    c64(0.5458675431868223, -0.10971903563869077)
                ],
            ],
            [
                [
                    c64(0.39761214735276523, 0.1911185162508152),
                    c64(0.7155757503688567, -1.2463556598814403)
                ],
                [
                    c64(-1.2323927201361262, 1.1168822069634479),
                    c64(3.4584408230318444, -390.05113808702043)
                ],
            ],
            [
                [
                    c64(0.04686626414341519, 9.25064335148286e-4),
                    c64(0.06661043300916779, -1.1371352153266978)
                ],
                [
                    c64(0.006345435959551723, 0.006314005745181268),
                    c64(0.4882940881783893, -0.4721320825578742)
                ],
            ],
        ]);

        let exemplar_h = Points::new(array![
            [
                [
                    c64(0.9625186306094178, -0.22887701590328144),
                    c64(-1.5375603451309596, -0.07416412595936357)
                ],
                [
                    c64(0.0021178781548226505, -0.03993895904049242),
                    c64(1.7403711725430853, 0.41269831113613603)
                ],
            ],
            [
                [
                    c64(2.0291927440451167, -0.9540811006959242),
                    c64(-0.008239667486325092, -6.00998985342353e-4)
                ],
                [
                    c64(0.008856852439349883, 0.0036009002998824815),
                    c64(7.910524066400982e-7, 0.002536004683241709)
                ],
            ],
            [
                [
                    c64(31.378370866300532, -0.3315648611508466),
                    c64(-38.351941918327334, 36.03638546644967)
                ],
                [
                    c64(-0.012353611578814082, -0.41338167219311783),
                    c64(1.5430289329650075, 1.5195786452299769)
                ],
            ],
        ]);

        let exemplar_s = Points::new(array![
            [
                [
                    c64(-0.9623430870772148, -0.010114135749597068),
                    c64(-0.032892112564098006, 0.005881025767514882)
                ],
                [
                    c64(1.4871251487033453e-4, 8.553179086165596e-4),
                    c64(-0.9784561857601861, -0.005035627286912867)
                ],
            ],
            [
                [
                    c64(-0.9212169334924274, -0.03518038771460153),
                    c64(-0.015738542584796452, 5.511049630169696e-4)
                ],
                [
                    c64(-0.017523514701019295, -0.0050075143069553425),
                    c64(0.9681312578391027, -0.24960142383274408)
                ],
            ],
            [
                [
                    c64(-0.4005804553925762, -0.009508052006542934),
                    c64(-0.03514926515147758, 0.6753273395247927)
                ],
                [
                    c64(0.0037914227535220508, 0.003723840176853586),
                    c64(-0.9849648487680371, -0.014538021436056158)
                ],
            ],
        ]);

        let exemplar_t = Points::new(array![
            [
                [
                    c64(-202.54611024000002, 1065.62529132),
                    c64(-201.36188976000003, 1090.11870868)
                ],
                [
                    c64(198.77788976000002, -1109.40670868),
                    c64(197.31411024000002, -1134.84929132)
                ],
            ],
            [
                [
                    c64(-44.578652657207456, 23.912834821274856),
                    c64(49.13197456228161, -12.032328070733726)
                ],
                [
                    c64(47.31367745179957, -27.764141434423593),
                    c64(-52.75803026066422, 15.07611890905036)
                ],
            ],
            [
                [
                    c64(-54.98799673741362, 50.642553183613025),
                    c64(-55.03092396018427, 51.54221154162553)
                ],
                [
                    c64(134.14675153504442, -127.921119076999),
                    c64(134.24827972523855, -131.85528763573544)
                ],
            ],
        ]);

        let exemplar_y = Points::new(array![
            [
                [
                    c64(0.9833390626156057, 0.2338279001727904),
                    c64(1.4946015066547766, 0.43245298899150947)
                ],
                [
                    c64(0.011421435247022799, -0.03877831954295688),
                    c64(1.760808278638465, 0.3539213655183655)
                ],
            ],
            [
                [
                    c64(0.4035870224366391, 0.18975760273287182),
                    c64(0.0032113787399704777, 0.001806094940502799)
                ],
                [
                    c64(0.002891212495572124, 0.0031339317167705934),
                    c64(2.2730192220551214e-5, 0.0025635648542889945)
                ],
            ],
            [
                [
                    c64(0.031865528080802376, 3.3671249022544614e-4),
                    c64(1.2342387832408936, -1.1354048751434689)
                ],
                [
                    c64(-2.5446358440632175e-4, -0.013176784898676817),
                    c64(1.058426060538316, 1.0233933039404715)
                ],
            ],
        ]);

        let exemplar_z = exemplar;

        let margin = MARGIN;
        comp_points_c64(&exemplar_a, &calc.net(RFParameter::A), margin, "net(A)");
        comp_points_c64(&exemplar_a, &calc.a(), margin, "a()");
        new_net.set_net(calc.a(), RFParameter::A);
        comp_points_c64(&exemplar_a, &new_net.a(), margin, "a(new_net)");

        comp_points_c64(&exemplar_g, &calc.net(RFParameter::G), margin, "net(G)");
        comp_points_c64(&exemplar_g, &calc.g(), margin, "g()");
        new_net.set_net(calc.g(), RFParameter::G);
        comp_points_c64(&exemplar_g, &new_net.g(), margin, "g(new_net)");

        comp_points_c64(&exemplar_h, &calc.net(RFParameter::H), margin, "net(H)");
        comp_points_c64(&exemplar_h, &calc.h(), margin, "h()");
        new_net.set_net(calc.h(), RFParameter::H);
        comp_points_c64(&exemplar_h, &new_net.h(), margin, "h(new_net)");

        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_t, &calc.net(RFParameter::T), margin, "net(T)");
        comp_points_c64(&exemplar_t, &calc.t(), margin, "t()");
        new_net.set_net(calc.t(), RFParameter::T);
        comp_points_c64(&exemplar_t, &new_net.t(), margin, "t(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    fn network_z_to_3port() {
        let param = RFParameter::Z;

        // 3 PortVal
        let exemplar = Points::new(array![[
            [c64(0.958, -0.263), c64(-0.846, 0.158), c64(1.473, 0.230)],
            [c64(0.004, 0.022), c64(0.544, -0.129), c64(-30.321, 4.378)],
            [c64(4.234, 84.212), c64(0.457, 0.287), c64(3.489, -2.893)],
        ]]);

        let calc = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0), c64::from(50.0), c64::from(50.0),],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );
        let mut new_net = calc.clone();

        let exemplar_s = Points::new(array![[
            [
                c64(-0.9363584033146554, -0.06735806039174684),
                c64(-0.03284074046511246, 0.004493615742204726),
                c64(0.03385323741509071, 0.01724739584282606)
            ],
            [
                c64(0.19290070897274017, 1.8292358295080133),
                c64(-0.9591992622451238, 0.030710182797377566),
                c64(-1.1091336360676787, 0.0665741119414386)
            ],
            [
                c64(-0.10855886998740481, 3.0313601637570065),
                c64(0.023178639236865643, 0.06285161257517914),
                c64(-0.8272250394291196, -0.1481078232509937)
            ],
        ]]);

        let exemplar_y = Points::new(array![[
            [
                c64(0.0049505482271746445, -0.006277586868079991),
                c64(-5.05139977548531e-4, -0.0017634882308923848),
                c64(6.805419547393476e-4, -0.011803426938288317)
            ],
            [
                c64(-1.1711093979459362, -0.23744162620243292),
                c64(-0.05219414542656583, -0.029551423941518985),
                c64(-2.338269177300081e-4, -0.014070662368167343)
            ],
            [
                c64(-0.021669227998620977, -0.002403588627800365),
                c64(-0.03330252100014572, -0.005117230099113714),
                c64(-1.8557328737234102e-5, -2.55194829519355e-4)
            ],
        ]]);

        let exemplar_z = exemplar;

        let margin = MARGIN;
        comp_points_c64(&exemplar_s, &calc.net(RFParameter::S), margin, "net(S)");
        comp_points_c64(&exemplar_s, &calc.s(), margin, "s()");
        new_net.set_net(calc.s(), RFParameter::S);
        comp_points_c64(&exemplar_s, &new_net.s(), margin, "s(new_net)");

        comp_points_c64(&exemplar_y, &calc.net(RFParameter::Y), margin, "net(Y)");
        comp_points_c64(&exemplar_y, &calc.y(), margin, "y()");
        new_net.set_net(calc.y(), RFParameter::Y);
        comp_points_c64(&exemplar_y, &new_net.y(), margin, "y(new_net)");

        comp_points_c64(&exemplar_z, &calc.net(RFParameter::Z), margin, "net(Z)");
        comp_points_c64(&exemplar_z, &calc.z(), margin, "z()");
        new_net.set_net(calc.z(), RFParameter::Z);
        comp_points_c64(&exemplar_z, &new_net.z(), margin, "z(new_net)");
    }

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_z_to_a() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Z,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.a();
    }

    #[test]
    #[should_panic(expected = "G parameters do not exist for network with 1 port(s)")]
    fn network_z_to_g() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Z,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.g();
    }

    #[test]
    #[should_panic(expected = "H parameters do not exist for network with 1 port(s)")]
    fn network_z_to_h() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Z,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.h();
    }

    #[test]
    #[should_panic(expected = "T parameters do not exist for network with 1 port(s)")]
    fn network_z_to_t() {
        let net = Network::new(
            Frequency::new_scaled(array![1.0], Scale::Giga),
            array![c64::from(50.0)],
            RFParameter::Z,
            Points::new(array![[[c64(0.958, -0.263)]]]),
            String::from(""),
            String::from(""),
        );

        net.t();
    }

    // #[test]
    // fn port_for_loop() {
    //     let tree = PortVal::Children(vec![
    //         PortVal::Leaf((0, 0)),
    //         PortVal::Leaf((0, 1)),
    //         PortVal::Leaf((1, 0)),
    //         PortVal::Leaf((1, 1)),
    //     ]);

    //     for &node in &tree {
    //         let _: (usize, usize) = node;
    //     }
    // }

    // #[test]
    // fn port_iterator() {
    //     let tree = PortVal::Children(vec![
    //         PortVal::Leaf((0, 0)),
    //         PortVal::Leaf((0, 1)),
    //         PortVal::Children(vec![
    //             PortVal::Leaf((1, 0)),
    //             PortVal::Leaf((1, 1)),
    //             PortVal::Children(vec![]),
    //         ]),
    //     ]);

    //     let nums: Vec<(usize, usize)> = tree.iter().copied().collect();
    //     assert_eq!(nums, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    //     assert_ne!(nums, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    // }
}
