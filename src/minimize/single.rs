#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{ObjectiveDerFn, ObjectiveFn};
use crate::myfloat::MyFloat;
use num::Zero;

#[derive(Clone, Debug)]
pub struct Bracket {
    ax: MyFloat,
    bx: MyFloat,
    cx: MyFloat,
    fa: MyFloat,
    fb: MyFloat,
    fc: MyFloat,
    iter: usize,
}

impl Bracket {
    pub fn new() -> Self {
        Bracket {
            ax: MyFloat::new(-1.0),
            bx: MyFloat::new(1.0),
            cx: MyFloat::new(2.0),
            fa: MyFloat::zero(),
            fb: MyFloat::zero(),
            fc: MyFloat::zero(),
            iter: 0,
        }
    }

    pub fn bracket_boxed(&mut self, a: &MyFloat, b: &MyFloat, f: &mut Box<dyn ObjectiveFn>) {
        const ITMAX: usize = 100;
        const GOLD: f64 = 1.618034;
        const GLIMIT: f64 = 100.0;
        const TINY: f64 = 1e-20;

        self.ax = a.clone();
        self.bx = b.clone();
        let mut fu = MyFloat::zero();
        self.fa = f.call_scalar(&self.ax);
        self.fb = f.call_scalar(&self.bx);

        if self.fb > self.fa {
            std::mem::swap(&mut self.ax, &mut self.bx);
            std::mem::swap(&mut self.fa, &mut self.fb);
        }

        self.cx = if self.ax == self.bx {
            &self.bx + GOLD
        } else {
            &self.bx + GOLD * (&self.bx - &self.ax)
        };
        self.fc = f.call_scalar(&self.cx);

        self.iter = 0;
        while self.fb >= self.fc {
            self.iter += 1;
            let r = (&self.bx - &self.ax) * (&self.fb - &self.fc);
            let q = (&self.bx - &self.cx) * (&self.fb - &self.fa);
            let mut u = &self.bx
                - ((&self.bx - &self.cx) * &q - (&self.bx - &self.ax) * &r)
                    / (2.0 * (&q - &r).abs().max(&MyFloat::new(TINY)) * (&q - &r).signum());
            let ulim = &self.bx + GLIMIT * (&self.cx - &self.bx);

            fu = f.call_scalar(&u);
            if (&self.bx - &u) * (&u - &self.cx) > 0.0 {
                fu = f.call_scalar(&u);
                if fu < self.fc {
                    self.ax = self.bx.clone();
                    self.bx = u.clone();
                    self.fa = self.fb.clone();
                    self.fb = fu.clone();
                    if self.ax > self.cx {
                        std::mem::swap(&mut self.ax, &mut self.cx);
                        std::mem::swap(&mut self.fa, &mut self.fc);
                    }
                    return;
                } else if fu > self.fb {
                    self.cx = u.clone();
                    self.fc = fu.clone();
                    if self.ax > self.cx {
                        std::mem::swap(&mut self.ax, &mut self.cx);
                        std::mem::swap(&mut self.fa, &mut self.fc);
                    }
                    return;
                }
                u = &self.cx + GOLD * (&self.cx - &self.bx);
                fu = f.call_scalar(&u);
            } else if (&self.cx - &u) * (&u - &ulim) > 0.0 {
                fu = f.call_scalar(&u);
                if fu < self.fc {
                    (self.bx, self.cx, u) =
                        (self.cx.clone(), u.clone(), &u + GOLD * (&u - &self.cx));
                    (self.fb, self.fc, fu) = (self.fc.clone(), fu.clone(), f.call_scalar(&u));
                }
            } else if (&u - &ulim) * (&ulim - &self.cx) >= 0.0 {
                u = ulim.clone();
                fu = f.call_scalar(&u);
            } else {
                u = &self.cx + GOLD * (&self.cx - &self.bx);
                fu = f.call_scalar(&u);
            }
            (self.ax, self.bx, self.cx) = (self.bx.clone(), self.cx.clone(), u.clone());
            (self.fa, self.fb, self.fc) = (self.fb.clone(), self.fc.clone(), fu.clone());

            if self.fb < self.fc {
                break;
            }
        }

        if self.ax > self.cx {
            std::mem::swap(&mut self.ax, &mut self.cx);
            std::mem::swap(&mut self.fa, &mut self.fc);
        }
    }

    pub fn bracket<F>(&mut self, a: &MyFloat, b: &MyFloat, f: &mut F)
    where
        F: ObjectiveFn,
    {
        const ITMAX: usize = 100;
        const GOLD: f64 = 1.618034;
        const GLIMIT: f64 = 100.0;
        const TINY: f64 = 1e-20;

        self.ax = a.clone();
        self.bx = b.clone();
        let mut fu = MyFloat::zero();
        self.fa = f.call_scalar(&self.ax);
        self.fb = f.call_scalar(&self.bx);

        if self.fb > self.fa {
            std::mem::swap(&mut self.ax, &mut self.bx);
            std::mem::swap(&mut self.fa, &mut self.fb);
        }

        self.cx = if self.ax == self.bx {
            &self.bx + GOLD
        } else {
            &self.bx + GOLD * (&self.bx - &self.ax)
        };
        self.fc = f.call_scalar(&self.cx);

        self.iter = 0;
        while self.fb >= self.fc {
            self.iter += 1;
            let r = (&self.bx - &self.ax) * (&self.fb - &self.fc);
            let q = (&self.bx - &self.cx) * (&self.fb - &self.fa);
            let mut u = &self.bx
                - ((&self.bx - &self.cx) * &q - (&self.bx - &self.ax) * &r)
                    / (2.0 * (&q - &r).abs().max(&MyFloat::new(TINY)) * (&q - &r).signum());
            let ulim = &self.bx + GLIMIT * (&self.cx - &self.bx);

            fu = f.call_scalar(&u);
            if (&self.bx - &u) * (&u - &self.cx) > 0.0 {
                fu = f.call_scalar(&u);
                if fu < self.fc {
                    self.ax = self.bx.clone();
                    self.bx = u.clone();
                    self.fa = self.fb.clone();
                    self.fb = fu.clone();
                    if self.ax > self.cx {
                        std::mem::swap(&mut self.ax, &mut self.cx);
                        std::mem::swap(&mut self.fa, &mut self.fc);
                    }
                    return;
                } else if fu > self.fb {
                    self.cx = u.clone();
                    self.fc = fu.clone();
                    if self.ax > self.cx {
                        std::mem::swap(&mut self.ax, &mut self.cx);
                        std::mem::swap(&mut self.fa, &mut self.fc);
                    }
                    return;
                }
                u = &self.cx + GOLD * (&self.cx - &self.bx);
                fu = f.call_scalar(&u);
            } else if (&self.cx - &u) * (&u - &ulim) > 0.0 {
                fu = f.call_scalar(&u);
                if fu < self.fc {
                    (self.bx, self.cx, u) =
                        (self.cx.clone(), u.clone(), &u + GOLD * (&u - &self.cx));
                    (self.fb, self.fc, fu) = (self.fc.clone(), fu.clone(), f.call_scalar(&u));
                }
            } else if (&u - &ulim) * (&ulim - &self.cx) >= 0.0 {
                u = ulim.clone();
                fu = f.call_scalar(&u);
            } else {
                u = &self.cx + GOLD * (&self.cx - &self.bx);
                fu = f.call_scalar(&u);
            }
            (self.ax, self.bx, self.cx) = (self.bx.clone(), self.cx.clone(), u.clone());
            (self.fa, self.fb, self.fc) = (self.fb.clone(), self.fc.clone(), fu.clone());

            if self.iter >= ITMAX {
                break;
            }
        }

        if self.ax > self.cx {
            std::mem::swap(&mut self.ax, &mut self.cx);
            std::mem::swap(&mut self.fa, &mut self.fc);
        }
    }
}

pub struct Golden {
    xmin: MyFloat,
    fmin: MyFloat,
    f: Box<dyn ObjectiveFn>,
    pub bracket: Bracket,
    iter: usize,
}

impl Golden {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjectiveFn + 'static,
    {
        Golden {
            xmin: MyFloat::zero(),
            fmin: MyFloat::zero(),
            f: Box::new(f),
            bracket: Bracket::new(),
            iter: 0,
        }
    }

    fn set_x(&mut self, x1: &MyFloat, x2: &MyFloat, cat: bool) -> MyFloat {
        const R: f64 = 0.61803399;
        const C: f64 = 1.0 - R;

        if cat {
            self.bracket.bx = x1.clone();
            &self.bracket.bx + C * (&self.bracket.cx - &self.bracket.bx)
        } else {
            self.bracket.bx = x2.clone();
            &self.bracket.bx - C * (&self.bracket.bx - &self.bracket.cx)
        }
    }

    pub fn minimize(
        &mut self,
        a: &MyFloat,
        b: &MyFloat,
        tol: &MyFloat,
        max_iter: usize,
    ) -> Result<(MyFloat, MyFloat), String> {
        const PHI: f64 = 1.618033988749895; // Golden ratio
        const RESPHI: f64 = 2.0 - PHI; // 1 / PHI

        if *a >= *b {
            return Err("Invalid interval: a must be less than b".to_string());
        }

        if *tol <= MyFloat::zero() {
            return Err("Tolerance must be positive".to_string());
        }

        let mut x1 = a.clone();
        let mut x2 = b.clone();
        let mut x3 = &x1 + RESPHI * (&x2 - &x1);
        let mut x4 = &x1 + &x2 - &x3;

        let mut f3 = self.f.call_scalar(&x3);
        let mut f4 = self.f.call_scalar(&x4);

        for iter in 0..max_iter {
            if (&x2 - &x1).abs() < *tol {
                self.iter = iter;
                self.set_xmin((&x1 + &x2) * 0.5);
                return Ok((self.xmin.clone(), self.fmin.clone()));
            }

            if f3 < f4 {
                x2 = x4.clone();
                x4 = x3.clone();
                f4 = f3.clone();
                x3 = &x1 + RESPHI * (&x2 - &x1);
                f3 = self.f.call_scalar(&x3);
            } else {
                x1 = x3.clone();
                x3 = x4.clone();
                f3 = f4.clone();
                x4 = &x1 + &x2 - &x3;
                f4 = self.f.call_scalar(&x4);
            }
        }

        self.iter = max_iter;
        self.set_xmin((&x1 + &x2) * 0.5);
        Ok((self.xmin.clone(), self.fmin.clone()))
    }

    pub fn xmin(&self) -> &MyFloat {
        &self.xmin
    }

    pub fn fmin(&self) -> &MyFloat {
        &self.fmin
    }

    pub fn iter(&self) -> usize {
        self.iter
    }

    pub fn set_xmin(&mut self, xmin: MyFloat) {
        self.xmin = xmin;
        self.fmin = self.f.call_scalar(&self.xmin);
    }
}

pub struct Brent {
    xmin: MyFloat,
    fmin: MyFloat,
    f: Box<dyn ObjectiveFn>,
    bracket: Bracket,
    iter: usize,
}

impl Brent {
    const TOL: f64 = 3e-8;

    pub fn new<F>(f: F) -> Self
    where
        F: ObjectiveFn + 'static,
    {
        Brent {
            xmin: MyFloat::zero(),
            fmin: MyFloat::zero(),
            f: Box::new(f),
            bracket: Bracket::new(),
            iter: 0,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjectiveFn>) -> Self {
        Brent {
            xmin: MyFloat::zero(),
            fmin: MyFloat::zero(),
            f: f,
            bracket: Bracket::new(),
            iter: 0,
        }
    }

    pub fn bracket(&mut self, a: &MyFloat, b: &MyFloat) {
        self.bracket.bracket_boxed(a, b, &mut self.f);
    }

    pub fn minimize(
        &mut self,
        a: &MyFloat,
        b: &MyFloat,
        tol: &MyFloat,
        max_iter: usize,
    ) -> Result<(MyFloat, MyFloat), String> {
        if *tol <= MyFloat::zero() {
            return Err("Tolerance must be positive".to_string());
        }

        let mut ax = a.clone();
        let mut bx = b.clone();
        let mut fa = self.f.call_scalar(&ax);
        let mut fb = self.f.call_scalar(&bx);

        // Check that f(a) and f(b) have opposite signs
        if &fa * &fb > 0.0 {
            return Err("Function values at endpoints must have opposite signs".to_string());
        }

        // Ensure |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut ax, &mut bx);
            std::mem::swap(&mut fa, &mut fb);
        }

        let mut c = ax.clone();
        let mut fc = fa.clone();
        let mut d = &bx - &ax;
        let mut e = d.clone();

        for iter in 0..max_iter {
            // Check for convergence
            if fb.abs() < *tol {
                self.iter = iter;
                self.set_xmin(bx);
                return Ok((self.xmin.clone(), self.fmin.clone()));
            }

            // Check if we've converged to within tolerance
            if (&bx - &ax).abs() < *tol {
                self.iter = iter;
                self.set_xmin(bx);
                return Ok((self.xmin.clone(), self.fmin.clone()));
            }

            // Try inverse quadratic interpolation if we have three distinct points
            let mut s;
            if fa != fc && fb != fc {
                // Inverse quadratic interpolation
                s = &ax * &fb * &fc / ((&fa - &fb) * (&fa - &fc))
                    + &bx * &fa * &fc / ((&fb - &fa) * (&fb - &fc))
                    + &c * &fa * &fb / ((&fc - &fa) * (&fc - &fb));
            } else {
                // Secant method
                s = &bx - &fb * (&bx - &ax) / (&fb - &fa);
            }

            // Check if we should use bisection instead
            let use_bisection =
                // s is not between (3a + b)/4 and b
                !((3.0 * &ax + &bx) / 4.0 < s && s < bx) ||
                // Previous step used bisection and |s - b| >= |b - c| / 2
                (e.abs() < *tol && (&s - &bx).abs() >= (&bx - &c).abs() / 2.0) ||
                // Previous step didn't use bisection and |s - b| >= |c - d| / 2
                (e.abs() >= *tol && (&s - &bx).abs() >= (&c - &d).abs() / 2.0) ||
                // Previous step used bisection and |b - c| < tol
                (e.abs() < *tol && (&bx - &c).abs() < *tol) ||
                // Previous step didn't use bisection and |c - d| < tol
                (e.abs() >= *tol && (&c - &d).abs() < *tol);

            if use_bisection {
                s = (&ax + &bx) / 2.0;
                d = e.clone();
                e = (&bx - &ax).abs();
            } else {
                d = e.clone();
                e = (&c - &bx).abs();
            }

            // Update c to be the previous value of b
            c = bx.clone();
            fc = fb.clone();

            // Calculate new function value
            let fs = self.f.call_scalar(&s);

            // Update the interval
            if &fs * &fa < 0.0 {
                bx = s.clone();
                fb = fs.clone();
            } else {
                ax = s.clone();
                fa = fs.clone();
            }

            // Ensure |f(a)| >= |f(b)|
            if fa.abs() < fb.abs() {
                std::mem::swap(&mut ax, &mut bx);
                std::mem::swap(&mut fa, &mut fb);
            }
        }

        self.iter = max_iter;
        self.set_xmin(bx);
        Ok((self.xmin.clone(), self.fmin.clone()))
    }

    pub fn xmin(&self) -> &MyFloat {
        &self.xmin
    }

    pub fn fmin(&self) -> &MyFloat {
        &self.fmin
    }

    pub fn iter(&self) -> usize {
        self.iter
    }

    pub fn set_xmin(&mut self, xmin: MyFloat) {
        self.xmin = xmin;
        self.fmin = self.f.call_scalar(&self.xmin);
    }
}

pub struct DBrent {
    xmin: MyFloat,
    fmin: MyFloat,
    f: Box<dyn ObjectiveDerFn>,
    pub bracket: Bracket,
    iter: usize,
}

impl DBrent {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjectiveDerFn + 'static,
    {
        DBrent {
            xmin: MyFloat::zero(),
            fmin: MyFloat::zero(),
            f: Box::new(f),
            bracket: Bracket::new(),
            iter: 0,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjectiveDerFn>) -> Self {
        DBrent {
            xmin: MyFloat::zero(),
            fmin: MyFloat::zero(),
            f: f,
            bracket: Bracket::new(),
            iter: 0,
        }
    }

    pub fn minimize(
        &mut self,
        a: &MyFloat,
        b: &MyFloat,
        tol: &MyFloat,
        max_iter: usize,
    ) -> Result<(MyFloat, MyFloat), String> {
        const ZEPS: f64 = std::f64::EPSILON * 1e-3;

        if *a >= *b {
            return Err("Invalid interval: a must be less than b".to_string());
        }

        self.bracket.ax = a.clone();
        self.bracket.cx = b.clone();

        let mut ok1 = false;
        let mut ok2 = false;
        let mut d = MyFloat::zero();
        let mut d1 = MyFloat::zero();
        let mut d2 = MyFloat::zero();
        let mut du = MyFloat::zero();
        let mut e = MyFloat::zero();
        let mut fu = MyFloat::zero();
        let mut olde = MyFloat::zero();
        let mut tol1 = MyFloat::zero();
        let mut tol2 = MyFloat::zero();
        let mut u = MyFloat::zero();
        let mut u1 = MyFloat::zero();
        let mut u2 = MyFloat::zero();
        let mut xm = MyFloat::zero();

        let (a, b) = if self.bracket.ax < self.bracket.cx {
            (&mut self.bracket.ax, &mut self.bracket.cx)
        } else {
            (&mut self.bracket.cx, &mut self.bracket.ax)
        };
        let mut x = self.bracket.bx.clone();
        let mut w = self.bracket.bx.clone();
        let mut v = self.bracket.bx.clone();
        let mut fw = self.f.call_scalar(&x);
        let mut fv = fw.clone();
        let mut fx = fw.clone();
        let mut dw = self.f.df_scalar(&x);
        let mut dv = dw.clone();
        let mut dx = dw.clone();

        for iter in 0..max_iter {
            xm = 0.5 * (a.clone() + b.clone());
            tol1 = tol * x.abs() + ZEPS;
            tol2 = 2.0 * &tol1;

            if (&x - &xm).abs() <= &tol2 - 0.5 * (b.clone() - a.clone()) {
                self.set_xmin(x);
                self.iter = iter;
                return Ok((self.xmin.clone(), self.fmin.clone()));
            }

            if e.abs() > tol1 {
                d1 = 2.0 * (b.clone() - a.clone());
                d2 = d1.clone();
                if dw != dx {
                    d1 = (&w - &x) * &dx / (&dx - &dw);
                }
                if dv != dx {
                    d2 = (&v - &x) * &dx / (&dx - &dv);
                }
                u1 = &x + &d1;
                u2 = &x + &d2;
                ok1 = (a.clone() - &u1) * (&u1 - b.clone()) > MyFloat::zero()
                    && &dx * &d1 <= MyFloat::zero();
                ok2 = (a.clone() - &u2) * (&u2 - b.clone()) > MyFloat::zero()
                    && &dx * &d2 <= MyFloat::zero();
                olde = e.clone();
                e = d.clone();

                if ok1 || ok2 {
                    if ok1 && ok2 {
                        d = if d1.abs() < d2.abs() {
                            d1.clone()
                        } else {
                            d2.clone()
                        };
                    } else if ok1 {
                        d = d1.clone();
                    } else {
                        d = d2.clone();
                    }

                    if d.abs() <= (0.5 * &olde).abs() {
                        u = &x + &d;
                        if &u - a.clone() < tol2 || b.clone() - &u < tol2 {
                            d = &tol1 * (&xm - &x).signum();
                        }
                    } else {
                        e = if dx >= MyFloat::zero() {
                            a.clone() - &x
                        } else {
                            b.clone() - &x
                        };
                        d = 0.5 * &e;
                    }
                } else {
                    e = if dx >= MyFloat::zero() {
                        a.clone() - &x
                    } else {
                        b.clone() - &x
                    };
                    d = 0.5 * &e;
                }
            } else {
                e = if dx >= MyFloat::zero() {
                    a.clone() - &x
                } else {
                    b.clone() - &x
                };
                d = 0.5 * &e;
            }

            if d.abs() >= tol1 {
                u = &x + &d;
                fu = self.f.call_scalar(&u);
            } else {
                u = &x + &tol1 * d.signum();
                fu = self.f.call_scalar(&u);

                if fu > fx {
                    self.set_xmin(x);
                    self.iter = iter;
                    return Ok((self.xmin.clone(), self.fmin.clone()));
                }
            }

            du = self.f.df_scalar(&u);

            if fu <= fx {
                if u >= x {
                    *a = x.clone();
                } else {
                    *b = x.clone();
                }

                v = w.clone();
                fv = fw.clone();
                dv = dw.clone();

                w = x.clone();
                fw = fx.clone();
                dw = dx.clone();

                x = u.clone();
                fx = fu.clone();
                dx = du.clone();
            } else {
                if u < x {
                    *a = u.clone();
                } else {
                    *b = u.clone();
                }

                if fu <= fw || w == x {
                    v = w.clone();
                    fv = fw.clone();
                    dv = dw.clone();

                    w = u.clone();
                    fw = fu.clone();
                    dw = du.clone();
                } else if fu < fv || v == x || v == w {
                    v = u.clone();
                    fv = fu.clone();
                    dv = du.clone();
                }
            }
        }

        self.set_xmin(x);
        self.iter = max_iter;

        Ok((self.xmin.clone(), self.fmin.clone()))
    }

    pub fn xmin(&self) -> &MyFloat {
        &self.xmin
    }

    pub fn fmin(&self) -> &MyFloat {
        &self.fmin
    }

    pub fn iter(&self) -> usize {
        self.iter
    }

    pub fn set_xmin(&mut self, xmin: MyFloat) {
        self.xmin = xmin;
        self.fmin = self.f.call_scalar(&self.xmin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimize::{SingleDimDerFn, SingleDimFn};
    use crate::util::*;
    use float_cmp::F64Margin;
    use rand::Rng;
    use std::f64::consts::PI;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };
    const DEFAULT_TOL: f64 = 1e-6;
    const DEFAULT_TOL_BRENT: f64 = 1e-10;
    const DEFAULT_MAX_ITER: usize = 100;

    mod bracket_tests {
        use super::*;

        #[test]
        fn bracket_new() {
            let bracket = Bracket::new();
            assert_eq!(bracket.ax, -1.0);
            assert_eq!(bracket.bx, 1.0);
            assert_eq!(bracket.cx, 2.0);
            assert_eq!(bracket.fa, 0.0);
            assert_eq!(bracket.fb, 0.0);
            assert_eq!(bracket.fc, 0.0);
            assert_eq!(bracket.iter, 0);
        }

        #[test]
        fn bracket_simple_quadratic() {
            // f(x) = (x - 2)^2, minimum at x = 2
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 2.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::zero(), &MyFloat::new(1.0), &mut obj);

            // Should bracket the minimum at x = 2
            // ax < bx < cx and f(ax) > f(bx) < f(cx)
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            // The minimum should be between ax and cx
            assert!(bracket.ax < 2.0);
            assert!(bracket.cx > 2.0);
        }

        #[test]
        fn bracket_shifted_quadratic() {
            // f(x) = 3(x + 1)^2 - 5, minimum at x = -1
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { 3.0 * (x + 1.0).powi(2) - 5.0 });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(-2.0), &MyFloat::zero(), &mut obj);

            // Should bracket the minimum at x = -1
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            // The minimum should be between ax and cx
            assert!(bracket.ax < -1.0);
            assert!(bracket.cx > -1.0);
        }

        #[test]
        fn bracket_quartic_function() {
            // f(x) = x^4 - 4x^2 + 5, has minima at x = ±√2
            // Testing the positive minimum
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(4) - 4.0 * x.powi(2) + 5.0 });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(1.0), &MyFloat::new(2.0), &mut obj);

            // Should bracket a minimum around x = √2 ≈ 1.414
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa > bracket.fb);
            assert!(bracket.fc > bracket.fb);

            let sqrt2 = 2.0_f64.sqrt();
            assert!(bracket.ax < sqrt2);
            assert!(bracket.cx > sqrt2);
        }

        #[test]
        fn bracket_sine_function() {
            // f(x) = sin(x), minimum at x = -π/2 in interval [-π, 0]
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.sin() });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(-PI), &MyFloat::new(-1.0), &mut obj);

            // Should bracket the minimum at x = -π/2
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < -PI / 2.0);
            assert!(bracket.cx > -PI / 2.0);
        }

        #[test]
        fn bracket_cosine_function() {
            // f(x) = cos(x), minimum at x = π in interval [π/2, 3π/2]
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.cos() });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(
                &MyFloat::new(PI / 2.0),
                &MyFloat::new(3.0 * PI / 4.0),
                &mut obj,
            );

            // Should bracket the minimum at x = π
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < PI);
            assert!(bracket.cx > PI);
        }

        #[test]
        fn bracket_exponential_function() {
            // f(x) = e^x, strictly increasing function
            // Bracketing should handle this gracefully
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.exp() });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(-2.0), &MyFloat::new(1.0), &mut obj);

            // For a monotonic function, the algorithm should still produce valid output
            // though it won't find a true minimum
            assert!(bracket.ax <= bracket.bx);
            assert!(bracket.bx <= bracket.cx);
        }

        #[test]
        fn bracket_absolute_value() {
            // f(x) = |x - 3|, minimum at x = 3
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 3.0).abs() });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(1.0), &MyFloat::new(5.0), &mut obj);

            // Should bracket the minimum at x = 3
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 3.0);
            assert!(bracket.cx > 3.0);
        }

        #[test]
        fn bracket_reverse_initial_points() {
            // Test when initial b > a (algorithm should swap them)
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 2.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            // Start with b > a
            bracket.bracket(&MyFloat::new(3.0), &MyFloat::new(1.0), &mut obj);

            // Should still bracket the minimum properly
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);
        }

        #[test]
        fn bracket_narrow_parabola() {
            // f(x) = 100(x - 1.5)^2, very narrow parabola
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { 100.0 * (x - 1.5).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(1.0), &MyFloat::new(2.0), &mut obj);

            // Should bracket the minimum at x = 1.5
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 1.5);
            assert!(bracket.cx > 1.5);
        }

        #[test]
        fn bracket_wide_parabola() {
            // f(x) = 0.01(x - 10)^2, very wide parabola
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { 0.01 * (x - 10.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(5.0), &MyFloat::new(15.0), &mut obj);

            // Should bracket the minimum at x = 10
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 10.0);
            assert!(bracket.cx > 10.0);
        }

        #[test]
        fn bracket_polynomial_with_local_minimum() {
            // f(x) = x^4 - 2x^3 + x^2 + 1
            // This has a local minimum around x ≈ 1.36
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat {
                x.powi(4) - 2.0 * x.powi(3) + x.powi(2) + 1.0
            });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(0.5), &MyFloat::new(2.0), &mut obj);

            // Should bracket the local minimum
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa > bracket.fb);
            assert!(bracket.fc > bracket.fb);
        }

        #[test]
        fn bracket_very_close_initial_points() {
            // Test with initial points very close together
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 5.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(4.999), &MyFloat::new(5.001), &mut obj);

            // Should still be able to bracket the minimum
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 5.0);
            assert!(bracket.cx > 5.0);
        }

        #[test]
        fn bracket_large_scale_function() {
            // f(x) = (x - 1000)^2, minimum at x = 1000
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 1000.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(500.0), &MyFloat::new(1500.0), &mut obj);

            // Should bracket the minimum at x = 1000
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 1000.0);
            assert!(bracket.cx > 1000.0);
        }

        #[test]
        fn bracket_small_scale_function() {
            // f(x) = (x - 0.001)^2, minimum at x = 0.001
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 0.001).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(-0.001), &MyFloat::new(0.002), &mut obj);

            // Should bracket the minimum at x = 0.001
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 0.001);
            assert!(bracket.cx > 0.001);
        }

        #[test]
        fn bracket_iteration_count() {
            // Test that iteration count is tracked properly
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 1.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(5.0), &MyFloat::new(8.0), &mut obj);

            // Should have performed some iterations
            assert!(bracket.iter > 0);
            // But not too many for a simple quadratic
            assert!(bracket.iter < 50);
        }

        #[test]
        fn bracket_boxed_objective() {
            // Test the bracket_boxed method with Box<dyn ObjectiveFnF64>
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 3.0).powi(2) + 1.0 });
            let mut boxed_obj: Box<dyn ObjectiveFn> = Box::new(objective);
            let mut bracket = Bracket::new();

            bracket.bracket_boxed(&MyFloat::new(1.0), &MyFloat::new(5.0), &mut boxed_obj);

            // Should bracket the minimum at x = 3
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 3.0);
            assert!(bracket.cx > 3.0);
        }

        #[test]
        fn bracket_function_with_plateau() {
            // Function with a flat region: f(x) = (x^2 - 1)^2
            // Has minima at x = ±1
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x.powi(2) - 1.0).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            // Test bracketing the positive minimum
            bracket.bracket(&MyFloat::new(0.5), &MyFloat::new(1.5), &mut obj);

            // Should bracket one of the minima at x = 1
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);
        }

        #[test]
        fn bracket_negative_quadratic() {
            // f(x) = -(x - 2)^2 + 10, maximum at x = 2
            // For minimization, this becomes minimum at negative values
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { -(x - 2.0).powi(2) + 10.0 });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            // This function has a maximum, not minimum, so bracketing behavior will be different
            bracket.bracket(&MyFloat::zero(), &MyFloat::new(4.0), &mut obj);

            // The algorithm should still complete, though it may not find a meaningful bracket
            // for a function that has a maximum instead of minimum
            assert!(bracket.iter <= 100); // Should terminate
        }

        #[test]
        fn bracket_identical_start_points() {
            // Test with identical starting points
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::new(1.0), &MyFloat::new(1.0), &mut obj);

            // Algorithm should handle this gracefully
            // The golden ratio expansion should still work
            assert!(bracket.cx != bracket.bx);
        }

        #[test]
        fn bracket_steep_function() {
            // Very steep function: f(x) = 1000 * (x - 0.5)^2
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { 1000.0 * (x - 0.5).powi(2) });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::zero(), &MyFloat::new(1.0), &mut obj);

            // Should bracket the minimum at x = 0.5
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);

            assert!(bracket.ax < 0.5);
            assert!(bracket.cx > 0.5);
        }

        #[test]
        fn bracket_rational_function() {
            // f(x) = (x - 1)^2 / (1 + x^2), minimum around x = 1
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat {
                (x - 1.0).powi(2) / (1.0 + x.powi(2))
            });
            let mut bracket = Bracket::new();
            let mut obj = objective;

            bracket.bracket(&MyFloat::zero(), &MyFloat::new(2.0), &mut obj);

            // Should bracket the minimum near x = 1
            assert!(bracket.ax < bracket.bx);
            assert!(bracket.bx < bracket.cx);
            assert!(bracket.fa >= bracket.fb);
            assert!(bracket.fc >= bracket.fb);
        }
    }

    mod golden_tests {
        use super::*;

        #[test]
        fn golden_quadratic_function() {
            // f(x) = (x - 2)^2 + 1, minimum at x = 2, f_min = 1
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 2.0).powi(2) + 1.0 });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(0.0);
            let c = MyFloat::new(4.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2.0).abs() < DEFAULT_TOL);
            assert!((f_min - 1.0).abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_shifted_quadratic() {
            // f(x) = 3(x + 1)^2 - 5, minimum at x = -1, f_min = -5
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { 3.0 * (x + 1.0).powi(2) - 5.0 });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(-3.0);
            let c = MyFloat::new(1.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - (-1.0)).abs() < DEFAULT_TOL);
            assert!((f_min - (-5.0)).abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_quartic_function() {
            // f(x) = x^4 - 4x^2 + 5, minimum at x = ±√2, f_min = 1
            // Testing the positive minimum
            let objective =
                SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(4) - 4.0 * x.powi(2) + 5.0 });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(1.0);
            let c = MyFloat::new(2.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2_f64.sqrt()).abs() < DEFAULT_TOL);
            assert!((f_min - 1.0).abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_sine_function() {
            // f(x) = sin(x), minimum at x = -π/2 in interval [-π, 0]
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.sin() });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(-PI);
            let c = MyFloat::new(0.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - (-PI / 2.0)).abs() < DEFAULT_TOL);
            assert!((f_min - (-1.0)).abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_cosine_function() {
            // f(x) = cos(x), minimum at x = π in interval [π/2, 3π/2]
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.cos() });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(PI / 2.0);
            let c = MyFloat::new(3.0 * PI / 2.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - PI).abs() < DEFAULT_TOL);
            assert!((f_min - (-1.0)).abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_exponential_function() {
            // f(x) = e^x, strictly increasing, minimum at left boundary
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.exp() });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(-2.0);
            let c = MyFloat::new(1.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            // For monotonic function, should converge to left boundary
            assert!(x_min < -1.9);
        }

        #[test]
        fn golden_absolute_value() {
            // f(x) = |x - 3|, minimum at x = 3
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 3.0).abs() });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(1.0);
            let c = MyFloat::new(5.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 3.0).abs() < DEFAULT_TOL);
            assert!(f_min.abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_narrow_interval() {
            // Test on a very narrow interval
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 2.0).powi(2) + 1.0 });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(1.99);
            let c = MyFloat::new(2.01);
            let tol = MyFloat::new(DEFAULT_TOL);

            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2.0).abs() < DEFAULT_TOL);
            assert!((f_min - 1.0).abs() < DEFAULT_TOL);
        }

        #[test]
        fn golden_invalid_interval() {
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(2) });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(2.0);
            let c = MyFloat::new(1.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            // Test with a >= b
            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Invalid interval"));

            // Test with a == b
            let a = MyFloat::new(1.0);
            let c = MyFloat::new(1.0);
            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Invalid interval"));
        }

        #[test]
        fn golden_invalid_tolerance() {
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(2) });
            let mut test = Golden::new(objective);

            // Test with zero tolerance
            let a = MyFloat::new(0.0);
            let c = MyFloat::new(2.0);
            let result = test.minimize(&a, &c, &MyFloat::new(0.0), DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Tolerance must be positive"));

            // Test with negative tolerance
            let result = test.minimize(&a, &c, &MyFloat::new(-1e-6), DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Tolerance must be positive"));
        }

        #[test]
        fn golden_max_iterations() {
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 1.0).powi(2) });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(0.0);
            let c = MyFloat::new(2.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            // Test with very few iterations - should still return a result
            let result = test.minimize(&a, &c, &tol, 5);
            assert!(result.is_ok());

            // The result might not be as accurate, but should be reasonable
            let (x_min, _) = result.unwrap();
            assert!((x_min - 1.0).abs() < 0.1); // Relaxed tolerance
        }

        #[test]
        fn golden_function_with_multiple_local_minima() {
            // f(x) = sin(x) + 0.1*x, test in interval where it's unimodal
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.sin() + 0.1 * x });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(-PI);
            let c = MyFloat::new(0.0);
            let tol = MyFloat::new(DEFAULT_TOL);

            // This function has a minimum around x ≈ -1.37 in the interval [-π, 0]
            let result = test.minimize(&a, &c, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            // Verify it found a reasonable minimum
            assert!(x_min > -2.0 && x_min < -1.0);
            assert!(f_min < -0.8); // Should be a decent minimum value
        }

        #[test]
        fn golden_floating_point_edge_cases() {
            // Test with very large numbers
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 1e6).powi(2) });
            let mut test = Golden::new(objective);
            let a = MyFloat::new(5e5);
            let c = MyFloat::new(1.5e6);
            let result = test.minimize(&a, &c, &MyFloat::new(1e-3), DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - 1e6).abs() < 1e3);

            let a = MyFloat::new(0.0);
            let c = MyFloat::new(4.0);

            // Test with very small numbers
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { (x - 1e-6).powi(2) });
            let mut test = Golden::new(objective);
            let result = test.minimize(&a, &c, &MyFloat::new(1e-12), DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - 1e-6).abs() < 1e-9);
        }
    }

    mod brent_tests {
        use super::*;

        #[test]
        fn brent_sphere() {
            let exemplar_res = MyFloat::zero();
            let exemplar_x = MyFloat::zero();
            let mut rng = rand::rng();
            let lb = MyFloat::new(rng.random_range(-5.12..5.12));
            let ub = MyFloat::new(rng.random_range(-5.12..5.12));
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            // let objective = move |x: &MyFloat| -> MyFloat { x * x };
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x * x });
            let mut test = Brent::new(objective);
            _ = test.minimize(&lb, &ub, &tol, DEFAULT_MAX_ITER);

            println!(
                "\n\nx = {}\nf = {}\niters = {}\n\n",
                test.xmin(),
                test.fmin(),
                test.iter()
            );
            let margin = MARGIN;
            comp_myfloat(&exemplar_x, test.xmin(), margin, "solve()", "x");
            comp_myfloat(&exemplar_res, test.fmin(), margin, "solve(res)", "");
        }

        #[test]
        fn brent_simple_linear_root() {
            // f(x) = x - 2, root at x = 2
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x - 2.0 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(0.0);
            let b = MyFloat::new(4.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2.0).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_quadratic_root() {
            // f(x) = x^2 - 4, roots at x = ±2
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x * x - 4.0 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            // Test positive root
            let a = MyFloat::new(1.0);
            let b = MyFloat::new(3.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2.0).abs() < tol);
            assert!(f_min.abs() < tol);

            // Test negative root
            let a = MyFloat::new(-3.0);
            let b = MyFloat::new(-1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - (-2.0)).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_cubic_root() {
            // f(x) = x^3 - 8, root at x = 2
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(3) - 8.0 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(1.0);
            let b = MyFloat::new(3.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2.0).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_transcendental_function() {
            // f(x) = e^x - 2, root at x = ln(2)
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.exp() - 2.0 });
            let mut test = Brent::new(objective);
            let expected_root = 2.0_f64.ln();
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(0.0);
            let b = MyFloat::new(1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - expected_root).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_sine_root() {
            // f(x) = sin(x), root at x = π in interval [3, 4]
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.sin() });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(3.0);
            let b = MyFloat::new(4.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - PI).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_cosine_root() {
            // f(x) = cos(x), root at x = π/2
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.cos() });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(1.0);
            let b = MyFloat::new(2.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - PI / 2.0).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_polynomial_with_multiple_roots() {
            // f(x) = (x-1)(x-3)(x-5) = x^3 - 9x^2 + 23x - 15
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat {
                x.powi(3) - 9.0 * x.powi(2) + 23.0 * x - 15.0
            });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            // Test root at x = 1
            let a = MyFloat::new(0.0);
            let b = MyFloat::new(2.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());
            let (x_min, _) = result.unwrap();
            assert!((x_min - 1.0).abs() < tol);

            // Test root at x = 3
            let a = MyFloat::new(2.0);
            let b = MyFloat::new(4.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());
            let (x_min, _) = result.unwrap();
            assert!((x_min - 3.0).abs() < tol);

            // Test root at x = 5
            let a = MyFloat::new(4.0);
            let b = MyFloat::new(6.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());
            let (x_min, _) = result.unwrap();
            assert!((x_min - 5.0).abs() < tol);
        }

        #[test]
        fn brent_function_with_steep_slope() {
            // f(x) = 1000(x - 0.5), root at x = 0.5
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { 1000.0 * (x - 0.5) });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(0.0);
            let b = MyFloat::new(1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 0.5).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_function_with_shallow_slope() {
            // f(x) = 0.001(x - 100), root at x = 100
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { 0.001 * (x - 100.0) });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(99.0);
            let b = MyFloat::new(101.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 100.0).abs() < tol);
            assert!(f_min.abs() < tol);
        }

        #[test]
        fn brent_same_sign_endpoints() {
            // f(x) = x^2, both endpoints positive
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x * x });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(1.0);
            let b = MyFloat::new(2.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("opposite signs"));
        }

        #[test]
        fn brent_invalid_tolerance() {
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.clone() });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(0.0);

            // Zero tolerance
            let a = MyFloat::new(-1.0);
            let b = MyFloat::new(1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Tolerance must be positive"));

            // Negative tolerance
            let tol = MyFloat::new(-1e-6);
            let a = MyFloat::new(-1.0);
            let b = MyFloat::new(1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Tolerance must be positive"));
        }

        #[test]
        fn brent_root_at_endpoint() {
            // f(x) = x - 1, root at x = 1 (right endpoint)
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x - 1.0 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(0.0);
            let b = MyFloat::new(1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - 1.0).abs() < tol);
        }

        #[test]
        fn brent_very_narrow_interval() {
            // f(x) = x - 1.5, root at x = 1.5
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x - 1.5 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(1.49999);
            let b = MyFloat::new(1.50001);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - 1.5).abs() < tol);
        }

        #[test]
        fn brent_convergence_with_different_tolerances() {
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(3) - 2.0 });
            let mut test = Brent::new(objective);
            let expected_root = 2.0_f64.powf(1.0 / 3.0);

            let tolerances = [1e-3, 1e-6, 1e-9, 1e-12];

            for &tol in &tolerances {
                let a = MyFloat::new(1.0);
                let b = MyFloat::new(2.0);
                let result = test.minimize(&a, &b, &MyFloat::new(tol), DEFAULT_MAX_ITER);
                assert!(result.is_ok());

                let (x_min, f_min) = result.unwrap();
                assert!((x_min - expected_root).abs() < MyFloat::new(tol) * 10.0);
                assert!(f_min.abs() < MyFloat::new(tol) * 10.0);
            }
        }

        #[test]
        fn brent_max_iterations() {
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x - 1.0 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            // Test with very few iterations
            let a = MyFloat::new(0.0);
            let b = MyFloat::new(2.0);
            let result = test.minimize(&a, &b, &tol, 3);
            assert!(result.is_ok());

            // Should still find a reasonable approximation
            let (x_min, _) = result.unwrap();
            assert!((x_min - 1.0).abs() < 0.1);
        }

        #[test]
        fn brent_floating_point_precision() {
            // Test with numbers close to machine epsilon
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x - f64::EPSILON });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(0.0);
            let b = MyFloat::new(1e-15);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - f64::EPSILON).abs() < DEFAULT_TOL_BRENT);
        }

        #[test]
        fn brent_large_numbers() {
            // f(x) = x - 1e12, root at x = 1e12
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x - 1e12 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(1e6);

            let a = MyFloat::new(9e11);
            let b = MyFloat::new(1.1e12);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - 1e12).abs() < 1e6);
        }

        #[test]
        fn brent_convergence_rate() {
            // Test that Brent's method converges faster than bisection
            let objective = SingleDimFn::new(|x: &MyFloat| -> MyFloat { x.powi(3) - 2.0 });
            let mut test = Brent::new(objective);
            let tol = MyFloat::new(1e-15);

            // With tight tolerance, should converge in reasonable iterations
            let a = MyFloat::new(1.0);
            let b = MyFloat::new(2.0);
            let result = test.minimize(&a, &b, &tol, 50);
            assert!(result.is_ok());

            let (x_min, _) = result.unwrap();
            assert!((x_min - 2.0_f64.powf(1.0 / 3.0)).abs() < 1e-15);
        }
    }

    mod dbrent_tests {
        use super::*;

        #[test]
        fn dbrent_sphere() {
            let exemplar_res = MyFloat::zero();
            let exemplar_x = MyFloat::zero();
            let mut rng = rand::rng();
            let lb = MyFloat::new(rng.random_range(-5.12..5.12));
            let ub = MyFloat::new(rng.random_range(-5.12..5.12));
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            // let objective = move |x: &MyFloat| -> MyFloat { x * x };
            let objective = SingleDimDerFn::new(
                |x: &MyFloat| -> MyFloat { x * x },
                |x: &MyFloat| -> MyFloat { 2.0 * x },
            );
            let mut test = DBrent::new(objective);
            _ = test.minimize(&lb, &ub, &tol, DEFAULT_MAX_ITER);

            println!(
                "\n\nx = {}\nf = {}\niters = {}\n\n",
                test.xmin(),
                test.fmin(),
                test.iter()
            );
            let margin = MARGIN;
            comp_myfloat(&exemplar_x, test.xmin(), margin, "solve()", "x");
            comp_myfloat(&exemplar_res, test.fmin(), margin, "solve(res)", "");
        }

        #[test]
        fn dbrent_brent_minimize_quadratic() {
            // f(x) = (x - 3)^2 + 2, minimum at x = 3, f_min = 2
            let objective = SingleDimDerFn::new(
                |x: &MyFloat| -> MyFloat { (x - 3.0).powi(2) + 2.0 },
                |x: &MyFloat| -> MyFloat { 2.0 * (x - 3.0) },
            );
            let mut test = DBrent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(1.0);
            let b = MyFloat::new(5.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 3.0).abs() < DEFAULT_TOL_BRENT);
            assert!((f_min - 2.0).abs() < DEFAULT_TOL_BRENT);
        }

        #[test]
        fn dbrent_brent_minimize_cubic() {
            // f(x) = x^3 - 3x^2 + 2, f'(x) = 3x^2 - 6x = 3x(x - 2)
            // Critical points at x = 0 (local max) and x = 2 (local min)
            let objective = SingleDimDerFn::new(
                |x: &MyFloat| -> MyFloat { x.powi(3) - 3.0 * x.powi(2) + 2.0 },
                |x: &MyFloat| -> MyFloat { 3.0 * x.powi(2) - 6.0 * x },
            );
            let mut test = DBrent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            // Test finding minimum at x = 2
            let a = MyFloat::new(1.0);
            let b = MyFloat::new(3.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            assert!((x_min - 2.0).abs() < DEFAULT_TOL_BRENT);
            assert!((f_min - (-2.0)).abs() < DEFAULT_TOL_BRENT);
        }

        #[test]
        fn dbrent_brent_minimize_sine() {
            // f(x) = sin(x), f'(x) = cos(x)
            // Minimum at x = -π/2 in interval [-π, 0]
            let objective = SingleDimDerFn::new(
                |x: &MyFloat| -> MyFloat { x.sin() },
                |x: &MyFloat| -> MyFloat { x.cos() },
            );
            let mut test = DBrent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(-PI);
            let b = MyFloat::new(0.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_ok());

            let (x_min, f_min) = result.unwrap();
            println!(
                "\n\nx_min = {}\n-PI/2 = {}\ntol = {}\n\n",
                x_min,
                -PI / 2.0,
                (&x_min - (-PI / 2.0)).abs()
            );
            assert!((x_min - (-PI / 2.0)).abs() < 2.0 * DEFAULT_TOL_BRENT);
            assert!((f_min - (-1.0)).abs() < DEFAULT_TOL_BRENT);
        }

        #[test]
        fn dbrent_brent_minimize_invalid_interval() {
            let objective = SingleDimDerFn::new(
                |x: &MyFloat| -> MyFloat { x.powi(2) },
                |x: &MyFloat| -> MyFloat { 2.0 * x },
            );
            let mut test = DBrent::new(objective);
            let tol = MyFloat::new(DEFAULT_TOL_BRENT);

            let a = MyFloat::new(2.0);
            let b = MyFloat::new(1.0);
            let result = test.minimize(&a, &b, &tol, DEFAULT_MAX_ITER);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Invalid interval"));
        }
    }
}
