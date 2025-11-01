#![allow(unused_assignments)]
#![allow(unused_variables)]
use crate::minimize::{Minimizer, MinimizerResult, myfloat::ObjFn};
use crate::myfloat::MyFloat;
use ndarray::prelude::*;

/// Result of Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadBoundedResult {
    pub xmin: Array1<MyFloat>,
    pub fmin: MyFloat,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: MyFloat,
    pub history: Array1<MyFloat>,
}

impl MinimizerResult<Array1<MyFloat>> for NelderMeadBoundedResult {
    fn converged(&self) -> bool {
        self.converged
    }

    fn fmin(&self) -> f64 {
        self.fmin.to_f64()
    }

    fn fn_evals(&self) -> usize {
        self.fn_evals
    }

    fn iters(&self) -> usize {
        self.iters
    }

    fn xmin(&self) -> Array1<MyFloat> {
        self.xmin.clone()
    }
}

pub struct NelderMeadBounded {
    scale: Array1<MyFloat>,
    x: Array1<MyFloat>,
    x_scaled: Array1<MyFloat>,
    lb: Array1<MyFloat>,
    lb_vec: Vec<MyFloat>,
    ub: Array1<MyFloat>,
    ub_vec: Vec<MyFloat>,
    res: Option<Array1<MyFloat>>,
    simplex: Option<Array2<MyFloat>>,
    f: Box<dyn ObjFn<MyFloat>>,
    n: usize,
    iters: usize,
    max_iters: usize, // Maximum iterations
    tol: Option<MyFloat>,
    target_tol: MyFloat, // Convergent tolerance
    alpha: MyFloat,      // Reflection coefficient
    beta: MyFloat,       // Contraction coefficient
    gamma: MyFloat,      // Expansion coefficient
    rho: MyFloat,        // Scaling coefficient
    mu: MyFloat,
    verbosity: u32,
}

impl NelderMeadBounded {
    pub fn new<F>(
        x: Array1<MyFloat>,
        scale: Array1<MyFloat>,
        lb: Array1<MyFloat>,
        ub: Array1<MyFloat>,
        f: F,
    ) -> Self
    where
        F: ObjFn<MyFloat> + 'static,
    {
        let mut lb_vec: Vec<MyFloat> = vec![];
        let mut ub_vec: Vec<MyFloat> = vec![];
        for i in 0..x.len() {
            lb_vec.push(lb[i].clone());
            ub_vec.push(ub[i].clone());
        }
        NelderMeadBounded {
            x_scaled: Array1::from_shape_fn(x.len(), |i| &x[i] * &scale[i]),
            x: x.clone(),
            lb_vec,
            ub_vec,
            scale,
            lb,
            ub,
            res: None,
            simplex: None,
            f: Box::new(f),
            n: x.len(),
            iters: 0,
            max_iters: 1,
            tol: None,
            target_tol: 1e-12.into(),
            alpha: 1.0.into(),
            beta: 0.5.into(),
            gamma: 2.0.into(),
            rho: 0.5.into(),
            mu: 10.0.into(),
            verbosity: 0,
        }
    }

    fn check_bounds(&self, x: &MyFloat, lb: &MyFloat, ub: &MyFloat, scale: &MyFloat) -> MyFloat {
        if *x < (lb * scale) {
            return (1.0 + 0.05 / self.iters as f64) * lb * scale;
        } else if *x > (ub * scale) {
            return (1.0 - 0.05 / self.iters as f64) * ub * scale;
        }
        x.clone()
    }

    fn res_all(&self) -> Option<Array1<MyFloat>> {
        self.res.clone()
    }

    fn x_scaled(&self) -> Array1<MyFloat> {
        self.x_scaled.clone()
    }

    fn simplex(&self) -> Option<Array2<MyFloat>> {
        self.simplex.clone()
    }

    pub fn get_mu(&self) -> MyFloat {
        self.mu.clone()
    }

    pub fn set_mu(&mut self, mu: &MyFloat) {
        self.mu = mu.clone();
    }

    pub fn set_target_tolerance(&mut self, tol: &MyFloat) {
        self.target_tol = tol.clone();
    }

    pub fn set_x(&mut self, x: &Array1<MyFloat>) {
        self.x = x.clone();
        self.x_scaled = Array1::from_shape_fn(self.x.len(), |i| &self.x[i] * &self.scale[i]);
        self.res = None;
        self.simplex = None;
        self.tol = None;
    }

    pub fn set_verbosity(&mut self, verbose: u32) {
        self.verbosity = verbose;
    }

    fn calc_obj(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        let mut sum = MyFloat::new(0.0);
        for i in 0..x.len() {
            sum += (&self.ub[i] - &x[i]).ln() + (&x[i] - &self.lb[i]).ln();
        }
        &self.f.call(x) - &self.mu * &sum
    }

    fn x(&self) -> &Array1<MyFloat> {
        &self.x
    }

    fn final_value(&self) -> Option<MyFloat> {
        self.res.as_ref().map(|x| x[0].clone())
    }

    fn tolerance(&self) -> Option<MyFloat> {
        self.tol.clone()
    }

    fn iterations(&self) -> usize {
        self.iters
    }

    fn name(&self) -> &str {
        "NelderMeadBounded"
    }
}

impl Minimizer<Array1<MyFloat>> for NelderMeadBounded {
    /// Solve using Nelder-Mead algorithm
    fn minimize(&mut self, max_iters: Option<usize>) -> Box<dyn MinimizerResult<Array1<MyFloat>>> {
        self.max_iters = match max_iters {
            Some(x) => x,
            _ => 1000,
        };
        self.iters = 0;

        // Generate initial simplex veritces
        let c = MyFloat::new(1.0);
        let b = &c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
        let a = &c / 2f64.sqrt();
        let _ncols = self.n;
        let nrows = self.n + 1;
        let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
            if i == j && i < self.n {
                self.check_bounds(
                    &(&self.x_scaled[j] + &a + &b),
                    &self.lb[j],
                    &self.ub[j],
                    &self.scale[j],
                )
            } else if i < self.n {
                self.check_bounds(
                    &(&self.x_scaled[j] + &b),
                    &self.lb[j],
                    &self.ub[j],
                    &self.scale[j],
                )
            } else {
                self.x_scaled[j].clone()
            }
        });

        // Evaluate function at simplex vertices
        let mut res: Array1<MyFloat> = simplex
            .rows()
            .into_iter()
            .map(|x| self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / &self.scale[i])))
            .collect();
        let mut prev_tol = MyFloat::new(1.0);

        while self.iters < self.max_iters {
            if self.verbosity > 1 {
                println!(
                    "iteration: {}\terr: {}\ttol: {}",
                    self.iters,
                    res[0],
                    self.tol.clone().unwrap_or(1.0.into())
                );
            }
            self.iters += 1;

            // Sort points from best to worst
            let mut order: Vec<usize> = (0..res.len()).collect();
            order.sort_by(|&a, &b| {
                res[a]
                    .partial_cmp(&res[b])
                    .unwrap_or(std::cmp::Ordering::Greater)
            });
            let tmp_res = res.clone();
            let tmp_simplex = simplex.clone();
            for i in 0..order.len() {
                res[i] = tmp_res[order[i]].clone();
                for j in 0..self.n {
                    simplex[(i, j)] = tmp_simplex[(order[i], j)].clone();
                }
            }
            let shape = match simplex.shape() {
                [a, b] => (*a, *b),
                _ => panic!("Shape must be 2-dimensional"),
            };
            self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
                simplex[(i, j)].clone()
            }));

            // Determine if convergence criteria met
            let tol = 2.0 * (&res[res.len() - 1] - &res[0]).abs()
                / (res[res.len() - 1].abs() + res[0].abs() + 1e-10);
            // if tol.to_f64() < self.target_tol || (&prev_tol - &tol).abs().to_f64() < 1e-15 {
            //     break;
            // }
            if tol < self.target_tol {
                break;
            }
            prev_tol = tol.clone();

            let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)].clone()); // Best point
            let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)].clone()); // Lousy point
            let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)].clone()); // Worst point

            // Calculate the average of the n best points
            let mut x_avg = Array1::<MyFloat>::zeros(self.n);
            for i in 0..self.n {
                let mut sum = MyFloat::new(0.0);
                for j in 0..self.n {
                    sum += &simplex[(j, i)].clone();
                }
                x_avg[i] = 1.0 / self.n as f64 * &sum;
            }

            // Calculate reflection point
            let x_r = Array1::from_shape_fn(self.n, |i| {
                self.check_bounds(
                    &(&x_avg[i] + &self.alpha * (&x_avg[i] - &x_w[i])),
                    &self.lb[i],
                    &self.ub[i],
                    &self.scale[i],
                )
            });
            let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_r[i] / &self.scale[i]));

            //
            // Determine simplex adjustment
            //
            if f_r <= res[0] {
                // Perform expansion
                if self.verbosity > 1 {
                    println!("performing expansion");
                }
                let x_e = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &(&x_r[i] + &self.gamma * (&x_r[i] - &x_avg[i])),
                        &self.lb[i],
                        &self.ub[i],
                        &self.scale[i],
                    )
                });
                let f_e =
                    self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_e[i] / &self.scale[i]));
                if f_e < res[0] {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_e[i].clone();
                    }
                    res[nrows - 1] = f_e.clone();
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_r[i].clone();
                    }
                    res[nrows - 1] = f_r.clone();
                }
            } else if f_r > res[nrows - 1] {
                // Perform inside contraction
                if self.verbosity > 1 {
                    println!("performing inside contraction");
                }
                let x_ic = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &(&x_avg[i] - &self.beta * (&x_avg[i] - &x_w[i])),
                        &self.lb[i],
                        &self.ub[i],
                        &self.scale[i],
                    )
                });
                let f_ic = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                    &x_ic[i] / &self.scale[i]
                }));
                if f_ic > res[nrows - 1] {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] = self.check_bounds(
                                &(&x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j])),
                                &self.lb[j],
                                &self.ub[j],
                                &self.scale[j],
                            );
                        }

                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            &simplex[(i + 1, j)] / &self.scale[j]
                        }));
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_ic[i].clone();
                    }
                    res[nrows - 1] = f_ic.clone();
                }
            } else if f_r > res[nrows - 2] {
                // Perform outside contraction
                if self.verbosity > 1 {
                    println!("performing outside contraction");
                }
                let x_oc = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &(&x_avg[i] + &self.beta * (&x_avg[i] - &x_w[i])),
                        &self.lb[i],
                        &self.ub[i],
                        &self.scale[i],
                    )
                });
                let f_oc = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                    &x_oc[i] / &self.scale[i]
                }));
                if f_oc > f_r {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] = self.check_bounds(
                                &(&x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j])),
                                &self.lb[j],
                                &self.ub[j],
                                &self.scale[j],
                            );
                        }
                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            &simplex[(i + 1, j)] / &self.scale[j]
                        }));
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_oc[i].clone();
                    }
                    res[nrows - 1] = f_oc.clone();
                }
            } else {
                for i in 0..self.n {
                    simplex[(nrows - 1, i)] = x_r[i].clone();
                }
                res[nrows - 1] = f_r.clone();
            }

            self.tol = Some(tol.clone());
        }

        let shape = match simplex.shape() {
            [a, b] => (*a, *b),
            _ => panic!("Shape must be 2-dimensional"),
        };
        self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
            simplex[(i, j)].clone()
        }));
        self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)].clone());
        self.x = Array1::from_shape_fn(self.n, |i| &self.x_scaled[i] / &self.scale[i]);
        // self.res = Some(res);
        self.res = Some(Array1::from_shape_fn(res.len(), |i| res[i].clone()));

        if self.verbosity > 0 {
            println!("x: {:?}", self.x());
            println!("res: {:?}", self.res_all().unwrap());
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap());
        }

        Box::new(NelderMeadBoundedResult {
            xmin: self.x.clone(),
            fmin: self.calc_obj(&self.x.clone()),
            iters: 0,
            fn_evals: 0,
            converged: true,
            final_simplex_size: 0.0.into(),
            history: array![],
        })
    }
}

#[cfg(test)]
mod minimize_myfloat_neldermeadbounded_tests {
    use super::*;
    use crate::file::read_touchstone;
    use crate::frequency::*;
    use crate::mycomplex::MyComplex;
    use crate::network::{Network, NetworkBuilder};
    use crate::points::{Points, Pts};
    use crate::util::*;
    use float_cmp::F64Margin;

    fn calc_feed_y(freq: Frequency, x: &Array1<MyFloat>) -> Array3<MyComplex> {
        let mut yfeed = Array3::<MyComplex>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut zs = Array1::<MyComplex>::zeros(freq.npts());
        let mut zm = Array1::<MyComplex>::zeros(freq.npts());
        let mut zp = Array1::<MyComplex>::zeros(freq.npts());
        let mut zall = Array1::<MyComplex>::zeros(freq.npts());
        for i in 0..freq.npts() {
            zs[i] = &x[3] * MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[2]
                / (&x[3] + MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[2]);
            zm[i] = &x[1] + MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[0] + &zs[i];
            zp[i] = &x[5] - MyComplex::new(0.0.into(), 1.0.into()) / (w[i] * &x[4]);
            zall[i] = &zm[i] * &zp[i] / (&zm[i] + &zp[i]);
        }

        for i in 0..freq.npts() {
            yfeed[[i, 0, 1]] = -1.0 / &zall[i];
            yfeed[[i, 1, 0]] = -1.0 / &zall[i];
            yfeed[[i, 0, 0]] =
                MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[6] + 1.0 / &zall[i];
            yfeed[[i, 1, 1]] =
                MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[7] + 1.0 / &zall[i];
        }

        yfeed
    }

    fn calc_err(model: &Network, meas: &Network) -> MyFloat {
        let mut err = MyFloat::new(0.0);
        let meas_h = meas.h();
        let model_h = model.h();
        let meas_y = meas.y();
        let model_y = model.y();
        let meas_z = meas.z();
        let model_z = model.z();
        for i in 0..meas.freq().npts() {
            for port in [(0, 0), (0, 1), (1, 1)].iter() {
                err += ((model_h[[i, port.0, port.1]].re - meas_h[[i, port.0, port.1]].re)
                    / meas_h[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_h[[i, port.0, port.1]].im - meas_h[[i, port.0, port.1]].im)
                        / meas_h[[i, port.0, port.1]].im)
                        .powi(2);
                err += ((model_y[[i, port.0, port.1]].re - meas_y[[i, port.0, port.1]].re)
                    / meas_y[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_y[[i, port.0, port.1]].im - meas_y[[i, port.0, port.1]].im)
                        / meas_y[[i, port.0, port.1]].im)
                        .powi(2);
                err += ((model_z[[i, port.0, port.1]].re - meas_z[[i, port.0, port.1]].re)
                    / meas_z[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_z[[i, port.0, port.1]].im - meas_z[[i, port.0, port.1]].im)
                        / meas_z[[i, port.0, port.1]].im)
                        .powi(2);
            }
        }

        err
    }

    fn eval_f_simplex(x: &Array1<MyFloat>, meas: &Network) -> MyFloat {
        let model_y = calc_feed_y(meas.freq().clone(), x);
        let model_y_c64 =
            Points::from_shape_fn(model_y.dim(), |(i, j, k)| model_y[[i, j, k]].clone().into());
        let model = NetworkBuilder::new()
            .freq(meas.freq().clone())
            .z0(meas.z0().clone())
            .y(model_y_c64)
            .build();

        calc_err(&model, meas)
    }

    fn calc_xfmr_z(
        freq: &Frequency,
        ls: &MyFloat,
        rs: &MyFloat,
        km: &MyFloat,
        qp: &MyFloat,
        x: &Array1<MyFloat>,
    ) -> Array3<MyComplex> {
        let mut z = Array3::<MyComplex>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut m = Array1::<MyFloat>::zeros(freq.npts());
        let mut n = Array1::<MyFloat>::zeros(freq.npts());
        let mut rp = Array1::<MyFloat>::zeros(freq.npts());
        for i in 0..freq.npts() {
            m[i] = km * (&x[0] * ls).sqrt();
            n[i] = (ls / &x[0]).sqrt();
            rp[i] = w[i] * &x[0] / qp;
        }

        for i in 0..freq.npts() {
            z[[i, 0, 1]] = -MyComplex::new(0.0.into(), 1.0.into()) * freq.w_pt(i) * &m[i];
            z[[i, 1, 0]] = -MyComplex::new(0.0.into(), 1.0.into()) * freq.w_pt(i) * &m[i];
            z[[i, 0, 0]] = MyComplex::new(rp[i].clone(), w[i] * &x[0]);
            z[[i, 1, 1]] = MyComplex::new(rs.clone(), -w[i] * ls);
        }

        z
    }

    fn calc_err_xfmr(meas: &Network, model: &Network) -> MyFloat {
        let mut err = MyFloat::new(0.0);
        let meas_s = meas.s();
        let model_s = model.s();
        for i in 0..meas.freq().npts() {
            for port in [(0, 0)].iter() {
                err += ((model_s[[i, port.0, port.1]].re - meas_s[[i, port.0, port.1]].re)
                    / meas_s[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_s[[i, port.0, port.1]].im - meas_s[[i, port.0, port.1]].im)
                        / meas_s[[i, port.0, port.1]].im)
                        .powi(2);
            }
        }

        err
    }

    fn eval_f_xfmr(
        x: &Array1<MyFloat>,
        ls: &MyFloat,
        rs: &MyFloat,
        km: &MyFloat,
        qp: &MyFloat,
        meas: &Network,
    ) -> MyFloat {
        let model_z = calc_xfmr_z(&meas.freq(), ls, rs, km, qp, x);
        let model_z_c64 =
            Points::from_shape_fn(model_z.dim(), |(i, j, k)| model_z[[i, j, k]].clone().into());
        let model = NetworkBuilder::new()
            .freq(meas.freq().clone())
            .z0(meas.z0().clone())
            .z(model_z_c64)
            .build();

        calc_err_xfmr(meas, &model)
    }

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn nelder_mead_bounded_iter1_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            MyFloat::new(272695.2468393795),
            326794.52891594684.into(),
            342207.96817168326.into(),
            342220.223907122.into(),
            343582.48617426824.into(),
            344033.75696385157.into(),
            416206.242967765.into(),
            2403422.1818864923.into(),
            1206556.0229802025.into()
        ];
        let exemplar_tol = f64::NAN.into();
        let exemplar_x = array![
            MyFloat::new(1.0883883476483184e-11),
            1.7777669529663687e-01.into(),
            2.7677669529663682e-13.into(),
            1.7677769529663687e-01.into(),
            1.1767766952966369e-15.into(),
            1.0001767766952967e+03.into(),
            1.1767766952966369e-15.into(),
            1.1767766952966369e-15.into()
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let mut test =
            NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<MyFloat>| {
                eval_f_simplex(x, &net)
            });
        test.minimize(Some(1));

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter2_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            272695.2468393795.into(),
            326794.52891594684.into(),
            342207.96817168326.into(),
            342220.223907122.into(),
            343582.48617426824.into(),
            344033.75696385157.into(),
            416206.242967765.into(),
            1206556.0229802025.into(),
            6972.801065944916.into()
        ];
        let exemplar_tol = 1.5924016727932413.into();
        let exemplar_x = array![
            1.0883883476483184e-11.into(),
            1.7777669529663687e-01.into(),
            2.7677669529663682e-13.into(),
            1.7677769529663687e-01.into(),
            1.1767766952966369e-15.into(),
            1.0001767766952967e+03.into(),
            1.1767766952966369e-15.into(),
            1.1767766952966369e-15.into()
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let mut test =
            NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<MyFloat>| {
                eval_f_simplex(x, &net)
            });
        test.minimize(Some(2));

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter10_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            4827.569008555878.into(),
            6491.027837932648.into(),
            6870.00113002921.into(),
            6972.801065944916.into(),
            7474.561087533056.into(),
            22859.74949460588.into(),
            57889.9701887297.into(),
            103373.76960471414.into(),
            4890.356172590547.into()
        ];
        let exemplar_tol = 1.9304191405840867.into();
        let exemplar_x = array![
            1.2361669322101397e-11.into(),
            1.0055555555555556e-06.into(),
            5.849383897826444e-13.into(),
            1.0055555555555556e-06.into(),
            1.9006085977366987e-15.into(),
            1.0008701159555326e+03.into(),
            1.4565740713168557e-15.into(),
            2.5661741932665137e-15.into()
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        // let mut test =
        //     NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);
        let mut test =
            NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<MyFloat>| {
                eval_f_simplex(x, &net)
            });
        test.minimize(Some(10));

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter100_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            3444.854217601127.into(),
            3445.6419507426426.into(),
            3446.992313419479.into(),
            3447.1629865763994.into(),
            3447.6709128994753.into(),
            3448.537541981722.into(),
            3449.1735500221007.into(),
            3449.6304958773794.into(),
            3446.6718686755007.into()
        ];
        let exemplar_tol = 0.0014246249880836607.into();
        let exemplar_x = array![
            2.3804481112658933e-11.into(),
            1.0059277453279175e-6.into(),
            7.013163931317152e-13.into(),
            0.7061273333553028.into(),
            4.947616036977013e-15.into(),
            1003.4892444833454.into(),
            5.401451482867253e-16.into(),
            1.1334435942547104e-14.into()
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let mut test =
            NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<MyFloat>| {
                eval_f_simplex(x, &net)
            });
        test.minimize(Some(100));

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    // #[test]
    // fn nelder_mead_bounded_xfmr() {
    //     let z0 = 50.0;
    //     let freq = FrequencyBuilder::new()
    //         .freqs_scaled(array![275.0], Scale::Giga)
    //         .build();
    //     let src = ImpedanceBuilder::new()
    //         .kind(ImpedanceType::Z)
    //         .category(ComplexNumberType::ReIm)
    //         .mode(ImpedanceMode::Se)
    //         // .re(42.4)
    //         // .im(-19.6)
    //         .re(50.0)
    //         .im(0.0)
    //         .z0(z0)
    //         .freq(freq.unitval(0))
    //         .build();
    //     let load = ImpedanceBuilder::new()
    //         .kind(ImpedanceType::Z)
    //         .category(ComplexNumberType::ReIm)
    //         .mode(ImpedanceMode::Se)
    //         // .re(212.3)
    //         // .im(-43.2)
    //         .re(100.0)
    //         .im(0.0)
    //         .z0(z0)
    //         .freq(freq.unitval(0))
    //         .build();
    //     let x: Array1<MyFloat> = array![1e-11];
    //     let scale: Array1<MyFloat> = array![1e12];
    //     let lb: Array1<MyFloat> = array![1e-15];
    //     let ub: Array1<MyFloat> = array![1e-9];
    //     let exemplar_res = array![3444.854217601127,];
    //     let exemplar_tol = 0.0014246249880836607;
    //     let exemplar_x = array![3.498285705078592e-12];
    //     let exemplar_km = 0.4;
    //     let exemplar_qp = 25.0;
    //     let exemplar_qs = 15.0;
    //     let ls = -(((exemplar_qs * load.z().im.powi(2) + exemplar_qs * load.z().re.powi(2))
    //         * src.z().im)
    //         / ((2.0 * exemplar_qs * load.z().im + 2.0 * load.z().re) * src.z().im
    //             + exemplar_qs * load.z().im.powi(2)
    //             + exemplar_qs * load.z().re.powi(2)))
    //         / freq.w()[0];

    //     let net = NetworkBuilder::new()
    //         .freq(freq)
    //         .z0(array![src.z(), load.z()])
    //         .s(points![[
    //             [c64(0.0, 0.0), c64(1.0, 0.0)],
    //             [c64(1.0, 0.0), c64(0.0, 0.0)]
    //         ]])
    //         .build();

    //     let mut test = NelderMeadBounded::new(x, scale, lb, ub, move |x| {
    //         eval_f_xfmr(x, ls, exemplar_km, exemplar_qp, exemplar_qs, &net)
    //     });
    //     test.minimize(Some(50));

    //     let margin = MARGIN;
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter200_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<MyFloat> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<MyFloat> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test =
    //         NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);

    //     let margin = MARGIN;
    //     let exemplar_res = array![
    //         818.7670362019819,
    //         819.7858499276429,
    //         819.8191081224865,
    //         819.8818753756948,
    //         820.072179669824,
    //         820.400571229628,
    //         820.6580990875366,
    //         820.6632007827918,
    //         819.4152758850769
    //     ];
    //     let exemplar_tol = 0.0026640279313826782;
    //     let exemplar_x = array![
    //         2.5558359727448667e-11,
    //         1.0009436575297008e-6,
    //         8.640548821790739e-13,
    //         0.04575718526325763,
    //         1.1871308027782806e-15,
    //         987.0763620759078,
    //         1.311157416982076e-16,
    //         9.087060407601609e-15
    //     ];
    //     let mu = 1.0;
    //     test.set_mu(mu);
    //     assert_eq!(mu, test.get_mu());
    //     test.minimize(Some(200));
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x, mu=1.0)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=1.0)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=1.0)",
    //         "",
    //     );

    //     let exemplar_res = array![
    //         555.311267334037,
    //         555.3896833543715,
    //         555.5822722884807,
    //         555.9175385731802,
    //         556.5515771850778,
    //         556.820749296749,
    //         556.9013339933028,
    //         556.9337932429075,
    //         556.1600388642868
    //     ];
    //     let exemplar_tol = 0.003641404147217269;
    //     let exemplar_x = array![
    //         2.528067770783543e-11,
    //         1.0006951943403888e-6,
    //         1.0169226355037147e-12,
    //         0.023668253316077553,
    //         1.1089003073348476e-14,
    //         1013.2824818461836,
    //         9.197901642318022e-16,
    //         8.668218207170575e-15
    //     ];
    //     test.set_x(x.clone());
    //     let mu = 0.1;
    //     test.set_mu(mu);
    //     assert_eq!(mu, test.get_mu());
    //     test.minimize(Some(200));
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x, mu=0.1)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=0.1)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=0.1)",
    //         "",
    //     );

    //     let exemplar_res = array![
    //         542.4623317818372,
    //         542.6035903708896,
    //         542.7839845275608,
    //         542.9156097461646,
    //         542.9335968602808,
    //         542.9503436944373,
    //         543.0027100742718,
    //         543.017174454764,
    //         542.5970783474999
    //     ];
    //     let exemplar_tol = 0.0011096407061036788;
    //     let exemplar_x = array![
    //         2.427259390407293e-11,
    //         1.009829574420645e-6,
    //         6.351662277864478e-13,
    //         0.026075078478467278,
    //         7.621742679101595e-15,
    //         1006.9703272139681,
    //         1.1713222130218483e-15,
    //         8.873892874525603e-15
    //     ];
    //     test.set_x(x.clone());
    //     let mu = 0.01;
    //     test.set_mu(mu);
    //     assert_eq!(mu, test.get_mu());
    //     test.minimize(Some(200));
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x, mu=0.01)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=0.01)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=0.01)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter500_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<MyFloat> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<MyFloat> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
    //     let exemplar_res = array![
    //         3203.2879708700866,
    //         3203.2949572112325,
    //         3203.2954429260863,
    //         3203.2955724351086,
    //         3203.299995287214,
    //         3203.300056320635,
    //         3203.3004882934383,
    //         3203.3007569470956,
    //         3203.2930641437715
    //     ];
    //     let exemplar_tol = 0.000005526963054275563;
    //     let exemplar_x = array![
    //         2.745441131485568e-11,
    //         5.802690745506808e-6,
    //         2.1735312351394178e-11,
    //         0.02148541121361614,
    //         4.0575542618309124e-13,
    //         2851.5938365163875,
    //         1.6942970452531605e-15,
    //         7.731480438940903e-15
    //     ];

    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test =
    //         NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);
    //     test.minimize(Some(500));

    //     let margin = MARGIN;
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter1000_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<MyFloat> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<MyFloat> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
    //     let exemplar_res = array![
    //         3203.2829155938794,
    //         3203.2829156375756,
    //         3203.282916068842,
    //         3203.2829162901735,
    //         3203.282916290638,
    //         3203.2829162913545,
    //         3203.282916374713,
    //         3203.28291642157,
    //         3203.282916681512
    //     ];
    //     let exemplar_tol = 0.0000000003395367987351268;
    //     let exemplar_x = array![
    //         2.7406879533217748e-11,
    //         5.842620278464627e-6,
    //         2.193783307045718e-11,
    //         0.0214722393586234,
    //         4.091786190272255e-13,
    //         2866.7924256179062,
    //         1.6948944506279059e-15,
    //         7.754134539035408e-15
    //     ];

    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test =
    //         NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);
    //     test.minimize(Some(1000));

    //     let margin = MARGIN;
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }
}
