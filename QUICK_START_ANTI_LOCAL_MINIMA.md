# Quick Start: Avoiding Local Minima

## TL;DR

```rust
use ndarray::array;
use rfkit::prelude::*;

let x0: Points1<f64> = array![1.0, 1.0, 1.0].into();
let scale = Points1::ones(3);
let lb: Points1<f64> = array![0.0, 0.0, 0.0].into();
let ub: Points1<f64> = array![10.0, 10.0, 10.0].into();

let objective = MultiDimFn::new(|x: &Points1<f64>| x[0].powi(2) + x[1].powi(2) + x[2].powi(2));
let mut optimizer = NelderMead::new(objective);

let mut options = NelderMeadOptions::new(
    &x0,
    Some(&scale),
    Some(&lb),
    Some(&ub),
    Some(1000),
    Some(1e-8),
    None,
    None,
    None,
    None,
    None,
    Some(0),
);

options.set_random_restart(true);
options.set_adaptive_simplex(true);

let result = optimizer.minimize(&options).unwrap();
```

The optimizer will detect stagnation, perturb or restart the simplex, keep the best solution found, and return that best result.

## Common Settings

```rust
options.set_random_restart(true);
options.set_adaptive_simplex(true);
options.set_stagnation_threshold(50);
```

For more aggressive exploration, lower `stagnation_threshold` and raise `max_iterations`. For expensive objectives, raise `stagnation_threshold` so restarts happen less often.

## Imports

Use `rfkit::prelude::*` for the facade crate, or import from the split crates explicitly:

```rust
use rfkit::base::prelude::{Points1, Points};
use rfkit::minimize::prelude::{Minimizer, MultiDimFn, NelderMead, NelderMeadOptions};
```

## Full Example

See [rfkit/examples/nelder_mead_multimodal.md](rfkit/examples/nelder_mead_multimodal.md).
