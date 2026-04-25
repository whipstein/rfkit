# Points Generic Zeros Implementation

## Summary

`rfkit-base` provides a generic `Points<T, D>::zeros` method that works with any ndarray dimension supported by the `Points` wrapper.

## Current Location

The implementation now lives in [rfkit-base/src/pts.rs](rfkit-base/src/pts.rs):

```rust
impl<T: Scalar, D: Dimension> Points<T, D> {
    pub fn zeros(shape: impl IntoDimension<Dim = D>) -> Self {
        Points(Array::<T, D>::from_elem(shape, T::zero()))
    }
}
```

## Examples

```rust
use ndarray::{Ix1, Ix2, Ix3};
use num_complex::Complex64;
use rfkit::prelude::*;

let zeros_1d: Points<f64, Ix1> = Points::zeros(10);
let zeros_2d: Points<f64, Ix2> = Points::zeros((3, 4));
let zeros_3d: Points<f64, Ix3> = Points::zeros((2, 3, 4));

let zeros_complex: Points<Complex64, Ix2> = Points::zeros((3, 3));
```

## Demo

The facade crate owns the runnable example:

```bash
cargo run -p rfkit --example ptsx_zeros_demo
```
