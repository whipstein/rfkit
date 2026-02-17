use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

/// Represents a two-word floating point type, represented as the sum of two
/// non-overlapping f64 values.
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct MyUsize {
    pub(crate) hi: u64,
    pub(crate) lo: u64,
}

impl MyUsize {
    /// Creates a new `TwoFloat` by adding two `f64` values using Algorithm 2
    /// from Joldes et al. (2017).
    pub fn new_add(a: u64, b: u64) -> Self {
        let s = a + b;
        let aa = s - b;
        let bb = s - aa;
        let da = a - aa;
        let db = b - bb;
        Self { hi: s, lo: da + db }
    }

    /// Creates a new `TwoFloat` by subtracting two `f64` values using
    /// Algorithm 2 from Joldes et al. (2017) modified for negative right-hand
    /// side.
    pub fn new_sub(a: u64, b: u64) -> Self {
        let s = a - b;
        let aa = s + b;
        let bb = s - aa;
        let da = a - aa;
        let db = b + bb;
        Self { hi: s, lo: da - db }
    }

    /// Creates a new `TwoFloat` by multiplying two `f64` values using
    /// Algorithm 3 from Joldes et al. (2017).
    pub fn new_mul(a: u64, b: u64) -> Self {
        let p = a * b;
        Self {
            hi: p,
            lo: a * b - p,
        }
    }

    /// Creates a new `TwoFloat` by dividing two `f64` values using Algorithm
    /// 15 from Joldes et al. (2017) modified for the left-hand-side having a
    /// zero value in the low word.
    pub fn new_div(a: u64, b: u64) -> Self {
        let th = a / b;
        let (ph, pl) = Self::new_mul(th, b).into();
        let dh = a - ph;
        let d = dh - pl;
        let tl = d / b;
        Self::fast_two_sum(th, tl)
    }

    fn fast_two_sum(a: u64, b: u64) -> MyUsize {
        // Joldes et al. (2017) Algorithm 1
        let s = a + b;
        let z = s - a;
        MyUsize { hi: s, lo: b - z }
    }
}

impl Add for MyUsize {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_add(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Add<&MyUsize> for MyUsize {
    type Output = MyUsize;

    fn add(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_add(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Add<MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn add(self, rhs: MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_add(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Add<&MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn add(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_add(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Add<u64> for MyUsize {
    type Output = Self;

    fn add(self, rhs: u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<&u64> for MyUsize {
    type Output = MyUsize;

    fn add(self, rhs: &u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, *rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<u64> for &MyUsize {
    type Output = MyUsize;

    fn add(self, rhs: u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<&u64> for &MyUsize {
    type Output = MyUsize;

    fn add(self, rhs: &u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(self.hi, *rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<MyUsize> for u64 {
    type Output = MyUsize;

    fn add(self, rhs: MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(rhs.hi, self).into();
        let v = rhs.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<&MyUsize> for u64 {
    type Output = MyUsize;

    fn add(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(rhs.hi, self).into();
        let v = rhs.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<MyUsize> for &u64 {
    type Output = MyUsize;

    fn add(self, rhs: MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(rhs.hi, *self).into();
        let v = rhs.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Add<&MyUsize> for &u64 {
    type Output = MyUsize;

    fn add(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_add(rhs.hi, *self).into();
        let v = rhs.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub for MyUsize {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_sub(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Sub<&MyUsize> for MyUsize {
    type Output = MyUsize;

    fn sub(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_sub(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Sub<MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn sub(self, rhs: MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_sub(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Sub<&MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn sub(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, rhs.hi).into();
        let (th, tl) = MyUsize::new_sub(self.lo, rhs.lo).into();
        let c = sl + th;
        let (vh, vl) = MyUsize::fast_two_sum(sh, c).into();
        let w = tl + vl;
        MyUsize::fast_two_sum(vh, w)
    }
}

impl Sub<u64> for MyUsize {
    type Output = Self;

    fn sub(self, rhs: u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<&u64> for MyUsize {
    type Output = MyUsize;

    fn sub(self, rhs: &u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, *rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<u64> for &MyUsize {
    type Output = MyUsize;

    fn sub(self, rhs: u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<&u64> for &MyUsize {
    type Output = MyUsize;

    fn sub(self, rhs: &u64) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self.hi, *rhs).into();
        let v = self.lo + sl;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<MyUsize> for u64 {
    type Output = MyUsize;

    fn sub(self, rhs: MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self, rhs.hi).into();
        let v = sl - rhs.lo;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<&MyUsize> for u64 {
    type Output = MyUsize;

    fn sub(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(self, rhs.hi).into();
        let v = sl - rhs.lo;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<MyUsize> for &u64 {
    type Output = MyUsize;

    fn sub(self, rhs: MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(*self, rhs.hi).into();
        let v = sl - rhs.lo;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Sub<&MyUsize> for &u64 {
    type Output = MyUsize;

    fn sub(self, rhs: &MyUsize) -> Self::Output {
        let (sh, sl) = MyUsize::new_sub(*self, rhs.hi).into();
        let v = sl - rhs.lo;
        MyUsize::fast_two_sum(sh, v)
    }
}

impl Mul for MyUsize {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, rhs.hi).into();
        let tl0 = self.lo * rhs.lo;
        let tl1 = self.hi * rhs.lo + tl0;
        let cl2 = self.lo * rhs.hi + tl1;
        let cl3 = cl1 + cl2;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<&MyUsize> for MyUsize {
    type Output = MyUsize;

    fn mul(self, rhs: &MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, rhs.hi).into();
        let tl0 = self.lo * rhs.lo;
        let tl1 = self.hi * rhs.lo + tl0;
        let cl2 = self.lo * rhs.hi + tl1;
        let cl3 = cl1 + cl2;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn mul(self, rhs: MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, rhs.hi).into();
        let tl0 = self.lo * rhs.lo;
        let tl1 = self.hi * rhs.lo + tl0;
        let cl2 = self.lo * rhs.hi + tl1;
        let cl3 = cl1 + cl2;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<&MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn mul(self, rhs: &MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, rhs.hi).into();
        let tl0 = self.lo * rhs.lo;
        let tl1 = self.hi * rhs.lo + tl0;
        let cl2 = self.lo * rhs.hi + tl1;
        let cl3 = cl1 + cl2;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<u64> for MyUsize {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, rhs).into();
        let cl3 = self.lo * rhs + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<&u64> for MyUsize {
    type Output = MyUsize;

    fn mul(self, rhs: &u64) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, *rhs).into();
        let cl3 = self.lo * rhs + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<u64> for &MyUsize {
    type Output = MyUsize;

    fn mul(self, rhs: u64) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, rhs).into();
        let cl3 = self.lo * rhs + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<&u64> for &MyUsize {
    type Output = MyUsize;

    fn mul(self, rhs: &u64) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(self.hi, *rhs).into();
        let cl3 = self.lo * rhs + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<MyUsize> for u64 {
    type Output = MyUsize;

    fn mul(self, rhs: MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(rhs.hi, self).into();
        let cl3 = rhs.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<&MyUsize> for u64 {
    type Output = MyUsize;

    fn mul(self, rhs: &MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(rhs.hi, self).into();
        let cl3 = rhs.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<MyUsize> for &u64 {
    type Output = MyUsize;

    fn mul(self, rhs: MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(rhs.hi, *self).into();
        let cl3 = rhs.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Mul<&MyUsize> for &u64 {
    type Output = MyUsize;

    fn mul(self, rhs: &MyUsize) -> Self::Output {
        let (ch, cl1) = MyUsize::new_mul(rhs.hi, *self).into();
        let cl3 = rhs.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Div for MyUsize {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = rh + (rhs.lo * th);
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        self * m
    }
}

impl Div<&MyUsize> for MyUsize {
    type Output = MyUsize;

    fn div(self, rhs: &MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = rh + (rhs.lo * th);
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        self * m
    }
}

impl Div<MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn div(self, rhs: MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = rh + (rhs.lo * th);
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        self * m
    }
}

impl Div<&MyUsize> for &MyUsize {
    type Output = MyUsize;

    fn div(self, rhs: &MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = rh + (rhs.lo * th);
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        self * m
    }
}

impl Div<u64> for MyUsize {
    type Output = Self;

    fn div(self, rhs: u64) -> Self::Output {
        let th = self.hi / rhs;
        let (ph, pl) = MyUsize::new_mul(th, rhs).into();
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / rhs;
        MyUsize::fast_two_sum(th, tl)
    }
}

impl Div<&u64> for MyUsize {
    type Output = MyUsize;

    fn div(self, rhs: &u64) -> Self::Output {
        let th = self.hi / rhs;
        let (ph, pl) = MyUsize::new_mul(th, *rhs).into();
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / rhs;
        MyUsize::fast_two_sum(th, tl)
    }
}

impl Div<u64> for &MyUsize {
    type Output = MyUsize;

    fn div(self, rhs: u64) -> Self::Output {
        let th = self.hi / rhs;
        let (ph, pl) = MyUsize::new_mul(th, rhs).into();
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / rhs;
        MyUsize::fast_two_sum(th, tl)
    }
}

impl Div<&u64> for &MyUsize {
    type Output = MyUsize;

    fn div(self, rhs: &u64) -> Self::Output {
        let th = self.hi / rhs;
        let (ph, pl) = MyUsize::new_mul(th, *rhs).into();
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / rhs;
        MyUsize::fast_two_sum(th, tl)
    }
}

impl Div<MyUsize> for u64 {
    type Output = MyUsize;

    fn div(self, rhs: MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = s - rh;
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        let (ch, cl1) = MyUsize::new_mul(m.hi, self).into();
        let cl3 = m.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Div<&MyUsize> for u64 {
    type Output = MyUsize;

    fn div(self, rhs: &MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = s - rh;
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        let (ch, cl1) = MyUsize::new_mul(m.hi, self).into();
        let cl3 = m.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Div<MyUsize> for &u64 {
    type Output = MyUsize;

    fn div(self, rhs: MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = s - rh;
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        let (ch, cl1) = MyUsize::new_mul(m.hi, *self).into();
        let cl3 = m.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl Div<&MyUsize> for &u64 {
    type Output = MyUsize;

    fn div(self, rhs: &MyUsize) -> Self::Output {
        let th = 1 / rhs.hi;
        let rh = 1 - rhs.hi * th;
        let s = rh - (rhs.lo * th);
        let z = s - rh;
        let (eh, el) = (s, 0 - (rhs.lo * th) - z);
        let e = MyUsize { hi: eh, lo: el };
        let d = e * th;
        let m = d + th;
        let (ch, cl1) = MyUsize::new_mul(m.hi, *self).into();
        let cl3 = m.lo * self + cl1;
        MyUsize::fast_two_sum(ch, cl3)
    }
}

impl From<u32> for MyUsize {
    fn from(value: u32) -> Self {
        Self {
            hi: value as u64,
            lo: 0,
        }
    }
}

impl From<u64> for MyUsize {
    fn from(value: u64) -> Self {
        Self {
            hi: value as u64,
            lo: 0,
        }
    }
}

impl From<MyUsize> for [u64; 2] {
    fn from(value: MyUsize) -> Self {
        [value.hi, value.lo]
    }
}

impl<'a> From<&'a MyUsize> for [u64; 2] {
    fn from(value: &'a MyUsize) -> Self {
        [value.hi, value.lo]
    }
}

impl From<MyUsize> for (u64, u64) {
    fn from(value: MyUsize) -> Self {
        (value.hi, value.lo)
    }
}

impl<'a> From<&'a MyUsize> for (u64, u64) {
    fn from(value: &'a MyUsize) -> Self {
        (value.hi, value.lo)
    }
}

impl PartialEq<MyUsize> for MyUsize {
    fn eq(&self, other: &MyUsize) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd<MyUsize> for MyUsize {
    fn partial_cmp(&self, other: &MyUsize) -> Option<Ordering> {
        let hi_cmp = self.hi.partial_cmp(&other.hi);
        if matches!(hi_cmp, Some(Ordering::Equal)) {
            self.lo.partial_cmp(&other.lo)
        } else {
            hi_cmp
        }
    }
}
