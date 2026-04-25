# Example: Avoiding Local Minima with Random Restarts

This example demonstrates how Nelder-Mead random restarts can escape local minima on a multimodal objective.

## Test Function

The Rastrigin function has a global minimum at `(0, 0)` and many local minima:

```text
f(x, y) = 20 + x^2 + y^2 - 10 * (cos(2*pi*x) + cos(2*pi*y))
```

## Example Code

```rust
use ndarray::array;
use rfkit::prelude::*;
use std::f64::consts::PI;

fn rastrigin_2d(x: &Points1<f64>) -> f64 {
    20.0 + x[0].powi(2) + x[1].powi(2)
        - 10.0 * (2.0 * PI * x[0]).cos()
        - 10.0 * (2.0 * PI * x[1]).cos()
}

fn options(
    x0: &Points1<f64>,
    scale: &Points1<f64>,
    lb: &Points1<f64>,
    ub: &Points1<f64>,
) -> NelderMeadOptions<f64> {
    NelderMeadOptions::new(
        x0,
        Some(scale),
        Some(lb),
        Some(ub),
        Some(500),
        Some(1e-8),
        None,
        None,
        None,
        None,
        None,
        Some(0),
    )
}

fn main() {
    let x0: Points1<f64> = array![4.5, 4.5].into();
    let scale = Points1::ones(2);
    let lb: Points1<f64> = array![-5.0, -5.0].into();
    let ub: Points1<f64> = array![5.0, 5.0].into();

    println!("Testing without random restarts:");
    let objective = MultiDimFn::new(rastrigin_2d);
    let mut optimizer = NelderMead::new(objective);
    let result = optimizer.minimize(&options(&x0, &scale, &lb, &ub)).unwrap();
    println!("Final solution: x = {:?}", result.xmin());
    println!("Final value:    f = {:.6}", result.fmin());
    println!("Iterations:     {}", result.iters());
    println!("Function evals: {}\n", result.fn_evals());

    println!("Testing with random restarts and adaptive simplex:");
    let objective = MultiDimFn::new(rastrigin_2d);
    let mut optimizer = NelderMead::new(objective);
    let mut restart_options = options(&x0, &scale, &lb, &ub);
    restart_options.enable_anti_stagnation(Some(3), Some(30), Some(0.1));
    restart_options.set_verbosity(1);

    let result = optimizer.minimize(&restart_options).unwrap();
    println!("\nFinal solution: x = {:?}", result.xmin());
    println!("Final value:    f = {:.6}", result.fmin());
    println!("Iterations:     {}", result.iters());
    println!("Function evals: {}", result.fn_evals());
    println!("Restarts:       {}", result.restarts);
}
```

## Notes

Random restarts are most useful for multimodal functions, unknown objective landscapes, poor initial guesses, and optimization problems where solution quality matters more than the extra function evaluations.
