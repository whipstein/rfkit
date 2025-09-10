#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{Brent, DBrent, ObjectiveDerFn, ObjectiveFn};
use crate::myfloat::MyFloat;
use ndarray::prelude::*;
use num::{One, Zero};
use rand::Rng;

#[derive(Clone)]
struct F1dim {
    p: Array1<MyFloat>,
    xi: Array1<MyFloat>,
    n: usize,
    f: Box<dyn ObjectiveFn>,
    xt: Array1<MyFloat>,
}

impl F1dim {
    pub fn new<F>(p: Array1<MyFloat>, xi: Array1<MyFloat>, f: F) -> Self
    where
        F: ObjectiveFn + 'static,
    {
        let n = p.len();
        let xt = Array1::zeros(n);
        F1dim {
            p,
            xi,
            n,
            f: Box::new(f),
            xt,
        }
    }

    pub fn new_boxed(p: Array1<MyFloat>, xi: Array1<MyFloat>, f: Box<dyn ObjectiveFn>) -> Self {
        let n = p.len();
        let xt = Array1::zeros(n);
        F1dim { p, xi, n, f, xt }
    }

    pub fn operator(&mut self, x: &MyFloat) -> MyFloat {
        azip!((xt in &mut self.xt, p in &self.p, xi in &self.xi) *xt = p + x * xi);

        self.f.call(&self.xt)
    }
}

impl ObjectiveFn for F1dim {
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        self.operator(&x[0])
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        self.operator(x)
    }
}

#[derive(Clone)]
struct Df1dim {
    p: Array1<MyFloat>,
    xi: Array1<MyFloat>,
    n: usize,
    f: Box<dyn ObjectiveDerFn>,
    xt: Array1<MyFloat>,
    dft: Array1<MyFloat>,
}

impl Df1dim {
    pub fn new<F>(p: Array1<MyFloat>, xi: Array1<MyFloat>, f: F) -> Self
    where
        F: ObjectiveDerFn + 'static,
    {
        let n = p.len();
        let xt = Array1::zeros(n);
        Df1dim {
            p,
            xi,
            n,
            f: Box::new(f),
            dft: xt.clone(),
            xt,
        }
    }

    pub fn new_boxed(p: Array1<MyFloat>, xi: Array1<MyFloat>, f: Box<dyn ObjectiveDerFn>) -> Self {
        let n = p.len();
        let xt = Array1::zeros(n);
        Df1dim {
            p,
            xi,
            n,
            f,
            dft: xt.clone(),
            xt,
        }
    }

    pub fn operator(&mut self, x: &MyFloat) -> MyFloat {
        azip!((xt in &mut self.xt, p in &self.p, xi in &self.xi) *xt = p + x * xi);

        self.f.call(&self.xt)
    }
}

impl ObjectiveFn for Df1dim {
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        self.operator(&x[0])
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        self.operator(x)
    }
}

impl ObjectiveDerFn for Df1dim {
    fn df(&self, _x: &Array1<MyFloat>) -> MyFloat {
        let mut df1 = MyFloat::zero();
        for j in 0..self.n {
            df1 += &self.dft[j] * &self.xi[j];
        }
        df1
    }

    fn df_scalar(&self, x: &MyFloat) -> MyFloat {
        self.df(&array![x.clone()])
    }
}

#[derive(Clone)]
struct Line {
    p: Array1<MyFloat>,
    xi: Array1<MyFloat>,
    f: Box<dyn ObjectiveFn>,
    n: usize,
}

impl Line {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjectiveFn + 'static,
    {
        Line {
            p: Array1::zeros(0),
            xi: Array1::zeros(0),
            n: 0,
            f: Box::new(f),
        }
    }

    pub fn new_boxed(f: Box<dyn ObjectiveFn>) -> Self {
        Line {
            p: Array1::zeros(0),
            xi: Array1::zeros(0),
            n: 0,
            f: f,
        }
    }

    pub fn set_p(&mut self, p: &Array1<MyFloat>) {
        self.n = p.len();
        self.p = p.clone();
        self.xi = Array1::zeros(self.n);
    }

    pub fn linmin(&mut self) -> MyFloat {
        println!("\n\np = {:?}\nxi = {:?}\n\n", self.p, self.xi);

        self.n = self.p.len();

        let f1dim = F1dim::new_boxed(self.p.clone(), self.xi.clone(), self.f.clone());
        let ax = MyFloat::zero();
        let xx = MyFloat::one();
        let mut brent = Brent::new(f1dim);
        brent.bracket(&ax, &xx);
        let result = brent.minimize(&ax, &xx, &MyFloat::new(1e-12), 1000);
        match result {
            Ok((xmin, _)) => {
                azip!((xi in &mut self.xi) *xi *= &xmin);
                azip!((p in &mut self.p, xi in &self.xi) *p += xi);
            }
            Err(e) => panic!("{}", e),
        }

        brent.fmin().clone()
    }
}

impl ObjectiveFn for Line {
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        self.f.call(x)
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        self.f.call_scalar(x)
    }
}

#[derive(Clone)]
struct DLine {
    p: Array1<MyFloat>,
    xi: Array1<MyFloat>,
    f: Box<dyn ObjectiveDerFn>,
    n: usize,
}

impl DLine {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjectiveDerFn + 'static,
    {
        DLine {
            p: Array1::zeros(0),
            xi: Array1::zeros(0),
            n: 0,
            f: Box::new(f),
        }
    }

    pub fn fill_p(&mut self, n: usize) {
        let mut rng = rand::rng();
        self.n = n;
        self.p = Array1::from_shape_fn(n, |_| MyFloat::new(rng.random_range(0.0..10.0)));
        self.xi = Array1::zeros(self.n);
    }

    pub fn set_p(&mut self, p: &Array1<MyFloat>) {
        self.n = p.len();
        self.p = p.clone();
        self.xi = Array1::zeros(self.n);
    }

    pub fn linmin(&mut self, tol: &MyFloat, max_iter: usize) -> MyFloat {
        let mut ax = MyFloat::zero();
        let mut xx = MyFloat::zero();

        self.n = self.p.len();

        let mut df1dim = Df1dim::new_boxed(self.p.clone(), self.xi.clone(), self.f.clone());
        ax = MyFloat::zero();
        xx = MyFloat::zero();
        let mut dbrent = DBrent::new(df1dim.clone());
        dbrent.bracket.bracket(&mut ax, &mut xx, &mut df1dim);
        let result = dbrent.minimize(&ax, &xx, tol, max_iter);
        match result {
            Ok((xmin, fmin)) => {
                for j in 0..self.n {
                    self.xi[j] *= &xmin;
                    self.p[j] += &self.xi[j];
                }
                fmin
            }
            Err(e) => panic!("{}", e),
        }
    }
}

pub struct Powell {
    iter: usize,
    fret: MyFloat,
    line: Line,
    ftol: MyFloat,
    f: Box<dyn ObjectiveFn>,
}

impl Powell {
    const TOL: f64 = 3e-8;

    pub fn new<F>(f: F) -> Self
    where
        F: ObjectiveFn + Clone + 'static,
    {
        let boxed = Box::new(f);
        Powell {
            iter: 0,
            fret: MyFloat::zero(),
            line: Line::new_boxed(boxed.clone()),
            ftol: MyFloat::zero(),
            f: boxed,
        }
    }

    pub fn minimize(&mut self, x0: &Array1<MyFloat>) -> Array1<MyFloat> {
        const ITMAX: usize = 200;
        const TINY: f64 = 1e-25;

        self.line.set_p(x0);
        println!("\n\nline.p = {:?}\n\n", self.line.p);
        let n = self.line.p.len();
        let mut ximat = Array2::<MyFloat>::eye(n);
        let mut fptt = MyFloat::zero();
        let mut pt = Array1::<MyFloat>::zeros(n);
        let mut ptt = Array1::<MyFloat>::zeros(n);
        self.line.xi = Array1::<MyFloat>::zeros(n);
        self.fret = self.line.call(x0);

        pt = self.line.p.clone();

        self.iter = 0;
        loop {
            let fp = self.fret.clone();
            let mut ibig = 0;
            let mut del = MyFloat::zero();

            for i in 0..n {
                for j in 0..n {
                    self.line.xi[j] = ximat[[j, i]].clone();
                }
                fptt = self.fret.clone();
                self.fret = self.line.linmin();
                if &fptt - &self.fret > del {
                    del = &fptt - &self.fret;
                    ibig = i + 1;
                }
            }
            println!("\n\nfret = {:?}\nfp = {:?}\n\n", self.fret, fp);

            if 2.0 * (&fp - &self.fret) <= &self.ftol * (fp.abs() + self.fret.abs()) + TINY {
                return self.line.p.clone();
            }

            if self.iter == ITMAX {
                return self.line.p.clone();
            }

            for j in 0..n {
                ptt[j] = 2.0 * &self.line.p[j] - &pt[j];
                self.line.xi[j] = &self.line.p[j] - &pt[j];
                pt[j] = self.line.p[j].clone();
            }

            fptt = self.line.call(&ptt);

            if fptt < fp {
                let t = 2.0
                    * (&fp - 2.0 * &self.fret + &fptt)
                    * (&fp - &self.fret - &del)
                    * (&fp - &self.fret - &del)
                    - &del * (&fp - &fptt) * (&fp - &fptt);
                if t < 0.0 {
                    self.fret = self.line.linmin();

                    for j in 0..n {
                        ximat[[j, ibig - 1]] = ximat[[j, n - 1]].clone();
                        ximat[[j, n - 1]] = self.line.xi[j].clone();
                    }
                }
            }

            self.iter += 1;
        }
    }

    pub fn xmin(&self) -> &Array1<MyFloat> {
        &self.line.p
    }

    pub fn fmin(&self) -> &MyFloat {
        &self.fret
    }

    pub fn iter(&self) -> usize {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::{minimize::MultiDimFn, util::*};
    use float_cmp::F64Margin;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };
    const DEFAULT_TOL: f64 = 1e-10;
    const DEFAULT_MAX_ITER: usize = 1000;
    const DEFAULT_LINE_TOL: f64 = 1e-8;

    mod powell_tests {
        use super::*;

        #[test]
        fn powell_sphere() {
            let exemplar_res = MyFloat::zero();
            let exemplar_x = array![MyFloat::zero(), MyFloat::zero()];
            let mut rng = rand::rng();
            let x0 = array![
                MyFloat::new(rng.random_range(-10.0..10.0)),
                MyFloat::new(rng.random_range(-10.0..10.0))
            ];

            let objective = MultiDimFn::new(|x: &Array1<MyFloat>| -> MyFloat {
                let mut sum = MyFloat::zero();
                for i in 0..x.len() {
                    sum += &x[i] * &x[i]
                }
                sum
            });
            let mut test = Powell::new(objective);
            test.minimize(&x0);

            println!(
                "\n\nx = {}\nf = {}\niters = {}\n\n",
                test.xmin(),
                test.fmin(),
                test.iter()
            );
            let margin = MARGIN;
            comp_array_myfloat(&exemplar_x, test.xmin(), margin, "solve()");
            comp_myfloat(&exemplar_res, test.fmin(), margin, "solve(res)", "");
        }
    }
}
