# rfkit

`rfkit` is a Rust workspace for RF circuit modeling, network parameter conversion, Touchstone file I/O, unit-aware numeric data, and numerical optimization.

The public package is the `rfkit` facade crate. Most users should start with:

```rust
use rfkit::prelude::*;
```

The facade re-exports the common types from the implementation crates while still leaving the split architecture visible when you want more explicit imports.

## Workspace Layout

- `rfkit`: facade crate for user-facing imports
- `rfkit-base`: numeric traits, `Points`, units, impedance helpers, RF math, comparison utilities, and macros
- `rfkit-network`: `Network`, RF parameter conversions, builders, and Touchstone I/O
- `rfkit-circuit`: circuits, elements, circuit solving, and conversion to networks
- `rfkit-minimize`: optimization traits and algorithms

## Features

- Build RF circuits from ports, lumped elements, microstrip elements, and transformers
- Solve circuits into `Network` objects and inspect S/Y/Z/ABCD/T/G/H style parameter views
- Read Touchstone files into typed RF networks
- Work with unit-aware scalar and array values
- Use `Points1`, `Points2`, and `Points3` wrappers around `ndarray`
- Run optimizers such as Nelder-Mead, Powell, Brent, quasi-Newton, CMA-ES, and simplex search

## Installation

Inside this workspace, use the facade package directly:

```bash
cargo check -p rfkit
```

From another local crate, depend on the facade crate by path:

```toml
[dependencies]
rfkit = { path = "../rfkit/rfkit" }
ndarray = "0.17"
num-complex = "0.4"
```

The examples below use `ndarray::array` and `num_complex::c64`, so those crates are shown explicitly.

## Imports

Use the full facade prelude for convenience:

```rust
use rfkit::prelude::*;
```

Or import from specific facade namespaces:

```rust
use rfkit::base::prelude::*;
use rfkit::circuit::prelude::*;
use rfkit::minimize::prelude::*;
use rfkit::network::prelude::*;
```

Compatibility modules are also available:

```rust
use rfkit::pts::{Points, Points1};
use rfkit::units::{ArrayUnitValue, Scale, Unit};
use rfkit::element::{Port, Resistor};
```

## Example: Points And Units

`Points` is a thin wrapper around `ndarray::Array` with RF-oriented helpers and type aliases for common dimensions.

```rust
use ndarray::prelude::*;
use num_complex::Complex64;
use rfkit::prelude::*;

fn main() {
    let real: Points<f64, Ix3> = Points::zeros((2, 3, 4));
    println!("shape = {:?}", real.dim());

    let complex: Points<Complex64, Ix3> = Points::zeros((2, 3, 4));
    println!("complex shape = {:?}", complex.dim());

    let capacitance = ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad);
    println!("stored farads = {}", capacitance.val());
}
```

For a runnable version:

```bash
cargo run -p rfkit --example ptsx_zeros_demo
```

## Example: Read A Touchstone File

Touchstone files are parsed into `Network<T, ArrayUnitValue<T>>`.

```rust
use rfkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = "data/test.s2p".to_string();
    let network = read_touchstone::<f64>(&file)?;

    println!("ports: {}", network.nports());
    println!("points: {}", network.npts());
    println!("S-parameters in dB: {:?}", network.s_db());

    Ok(())
}
```

## Example: Build A Simple Two-Port Circuit

This creates a 50 ohm, two-port network with a 100 ohm series resistor between the ports.

```rust
use ndarray::array;
use num_complex::c64;
use rfkit::prelude::*;

fn main() {
    let freq = ArrayUnitValue::new_freq_scaled(&array![1.0, 2.0, 3.0], Scale::Giga);
    let z0 = c64(50.0, 0.0);

    let p1: Element<f64> = Port::new("p1", z0, [1]).into();
    let p2: Element<f64> = Port::new("p2", z0, [2]).into();
    let r1: Element<f64> = Resistor::new(
        "r1",
        ScalarUnitValue::new_scaled(&100.0, Scale::Base, Unit::Ohm),
        [1, 2],
        z0,
    )
    .into();

    let mut circuit = Circuit::new(&freq);
    circuit.add_elem(&p1, &freq);
    circuit.add_elem(&p2, &freq);
    circuit.add_elem(&r1, &freq);

    let network = circuit.net();
    println!("S-parameters in dB: {:?}", network.s_db());
}
```

## Example: Add A Transformer

`Element::Transformer` is a simple public two-node element. Internally, `rfkit-circuit` lowers it into private leakage, magnetizing, and ideal-transformer elements so callers do not need to manage extra internal nodes.

```rust
use ndarray::array;
use num_complex::c64;
use rfkit::prelude::*;

fn main() -> Result<(), String> {
    let freq = ArrayUnitValue::new_freq_scaled(&array![1.0], Scale::Giga);
    let z0 = c64(50.0, 0.0);

    let transformer: Element<f64> = TransformerBuilder::new()
        .id("t1")
        .km(0.9)
        .n(2.0)
        .l1_val_scaled(10.0, Scale::Nano)
        .nodes([1, 2])
        .z0(z0)
        .build()?
        .into();

    let mut circuit = Circuit::new(&freq);
    circuit.add_elem(&Port::new("p1", z0, [1]).into(), &freq);
    circuit.add_elem(&Port::new("p2", z0, [2]).into(), &freq);
    circuit.add_elem(&transformer, &freq);

    assert!(circuit.elements().contains_key("t1"));
    println!("{:?}", circuit.net().s_db());

    Ok(())
}
```

## Example: Minimize A Function

The optimization API uses objective wrappers plus an options object. This example minimizes a simple sphere function.

```rust
use ndarray::array;
use rfkit::prelude::*;

fn sphere(x: &Points1<f64>) -> f64 {
    x.iter().map(|value| value * value).sum()
}

fn main() {
    let x0: Points1<f64> = array![5.0, -3.0].into();
    let scale = Points1::ones(2);

    let options = NelderMeadOptions::new(
        &x0,
        Some(&scale),
        None,
        None,
        Some(500),
        Some(1e-8),
        None,
        None,
        None,
        None,
        None,
        Some(0),
    );

    let objective = MultiDimFn::new(sphere);
    let mut optimizer = NelderMead::new(objective);
    let result = optimizer.minimize(&options).unwrap();

    println!("xmin = {:?}", result.xmin());
    println!("fmin = {}", result.fmin());
}
```

For random restarts and adaptive simplex behavior, see:

```bash
rfkit/examples/nelder_mead_multimodal.md
```

## Common Types

- `Points<T, D>`: generic ndarray-backed data container
- `Points1<T>`, `Points2<T>`, `Points3<T>`: 1D, 2D, and 3D aliases
- `ScalarUnitValue<T>`: a single scaled unit value
- `ArrayUnitValue<T>`: an array of scaled unit values, commonly used for frequency sweeps
- `Network<T, U>`: RF network data and parameter views
- `Circuit<T, U>`: circuit assembly and solve state
- `Element<T>`: enum over supported circuit elements
- `NelderMead`, `Powell`, `Brent`, `CmaEs`, and related option/result types

## Development

Check all crates and targets:

```bash
cargo check --workspace --all-targets
```

Run the full test suite:

```bash
cargo test --workspace --no-fail-fast
```

Build benchmarks without running them:

```bash
cargo bench -p rfkit --no-run
```

Run the Nelder-Mead benchmarks:

```bash
cargo bench -p rfkit --bench nelder_mead_benchmarks
```

The workspace uses `ndarray-linalg` with system OpenBLAS. If linking fails, make sure OpenBLAS is installed and visible to your linker.

## Architecture Notes

The root `Cargo.toml` is a virtual workspace. The old monolithic root `src/` layout has been replaced by the split crates listed above. New implementation work should go into the crate that owns the behavior, and `rfkit/src/lib.rs` should only expose stable facade paths for users.
