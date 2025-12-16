# Ptsx Generic Zeros Implementation

## Summary

Implemented a generic `zeros` method for `Pointsx<T, D>` that works with any dimension array.

## Changes Made

### 1. Updated `src/pts.rs`

- Made the inner `Array<T, D>` field public for easier access
- Added `Dimension` trait bound to the `Ptsx` trait
- Implemented a generic `zeros` method on `Pointsx<T, D>` that:
  - Works with any type `T` that implements `Clone + num_traits::Zero`
  - Works with any dimension `D` that implements `Dimension`
  - Accepts any shape that implements `ShapeBuilder<Dim = D>` (tuples, single values, etc.)

### 2. Key Implementation

```rust
impl<T, D> Pointsx<T, D>
where
    T: Clone + num_traits::Zero,
    D: Dimension,
{
    /// Create a new array with given dimensions filled with zeros
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        Pointsx(Array::from_elem(shape, T::zero()))
    }
}
```

## Features

The implementation supports:

### Multiple Dimensions
- 1D arrays (vectors): `Pointsx::<f64, Ix1>::zeros(10)`
- 2D arrays (matrices): `Pointsx::<f64, Ix2>::zeros((3, 4))`
- 3D arrays: `Pointsx::<f64, Ix3>::zeros((2, 3, 4))`
- 4D arrays: `Pointsx::<f64, Ix4>::zeros((2, 2, 2, 2))`
- 5D+ arrays: Works with any dimension supported by ndarray

### Multiple Types
- Floating point: `f32`, `f64`
- Complex numbers: `Complex32`, `Complex64`
- Integers: `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`
- Any type that implements `Clone + Zero`

## Examples

### Basic Usage

```rust
use rfkit::pts::Pointsx;
use ndarray::{Ix1, Ix2, Ix3};

// 1D array
let zeros_1d: Pointsx<f64, Ix1> = Pointsx::zeros(10);

// 2D array
let zeros_2d: Pointsx<f64, Ix2> = Pointsx::zeros((3, 4));

// 3D array
let zeros_3d: Pointsx<f64, Ix3> = Pointsx::zeros((2, 3, 4));
```

### Complex Numbers

```rust
use num::complex::Complex64;

let zeros_complex: Pointsx<Complex64, Ix2> = Pointsx::zeros((3, 3));
```

### Integer Types

```rust
let zeros_int: Pointsx<i32, Ix2> = Pointsx::zeros((5, 5));
let zeros_uint: Pointsx<u64, Ix3> = Pointsx::zeros((2, 2, 2));
```

## Testing

Comprehensive tests have been added in `tests/ptsx_zeros_test.rs` covering:
- Multiple dimensions (1D through 5D)
- Multiple types (f64, Complex64, i32, u64)
- Shape verification
- Value verification (all elements are zero)

All tests pass successfully.

## Demo

A demonstration program is available at `examples/ptsx_zeros_demo.rs` that shows the `zeros` method working with various dimensions and types.

Run it with:
```bash
cargo run --example ptsx_zeros_demo
```
