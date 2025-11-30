# Example: Avoiding Local Minima with Random Restarts

This example demonstrates how the random restart feature helps escape local minima on a multimodal function.

## Test Function: Rastrigin-like Function

The Rastrigin function is a classic multimodal benchmark with many local minima:

```
f(x, y) = 20 + x² + y² - 10*(cos(2πx) + cos(2πy))
```

This function has:
- Global minimum at (0, 0) with f(0,0) = 0
- Many local minima in a grid pattern
- Difficult for gradient-free methods

## Example Code

```rust
use ndarray::array;
use rfkit::minimize::f64::nelder_mead_bounded::NelderMeadBounded;
use rfkit::minimize::Minimizer;
use std::f64::consts::PI;

fn rastrigin_2d(x: &Array1<f64>) -> f64 {
    let a = 10.0;
    20.0 + x[0].powi(2) + x[1].powi(2)
        - a * (2.0 * PI * x[0]).cos()
        - a * (2.0 * PI * x[1]).cos()
}

fn main() {
    // Poor starting point far from global minimum
    let x0 = array![4.5, 4.5];
    let scale = array![1.0, 1.0];
    let lb = array![-5.0, -5.0];
    let ub = array![5.0, 5.0];

    println!("Testing WITHOUT random restarts:");
    let mut opt1 = NelderMeadBounded::new(x0.clone(), scale.clone(), lb.clone(), ub.clone(), rastrigin_2d);
    opt1.set_verbosity(0);
    let result1 = opt1.minimize(Some(500));
    println!("Final solution: x = {:?}", result1.xmin());
    println!("Final value:    f = {:.6}", result1.fmin());
    println!("Iterations:     {}", result1.iters());
    println!("Function evals: {}\n", result1.fn_evals());

    println!("Testing WITH random restarts and adaptive simplex:");
    let mut opt2 = NelderMeadBounded::new(x0.clone(), scale.clone(), lb.clone(), ub.clone(), rastrigin_2d);
    opt2.set_random_restart(true);
    opt2.set_adaptive_simplex(true);
    opt2.set_stagnation_threshold(30);
    opt2.set_verbosity(1);
    let result2 = opt2.minimize(Some(500));
    println!("\nFinal solution: x = {:?}", result2.xmin());
    println!("Final value:    f = {:.6}", result2.fmin());
    println!("Iterations:     {}", result2.iters());
    println!("Function evals: {}", result2.fn_evals());

    println!("\n--- Comparison ---");
    println!("Improvement: {:.2}%",
             (result1.fmin() - result2.fmin()) / result1.fmin() * 100.0);
    println!("Global minimum (ideal): f(0, 0) = 0.0");
}
```

## Expected Results

### Without Random Restarts
- Often gets stuck in a local minimum
- Final value typically: f ≈ 4.0 - 20.0 (depending on starting point)
- May converge to points like (1.0, 1.0) or (2.0, 2.0)
- Fast convergence but poor solution quality

### With Random Restarts + Adaptive Simplex
- Much better chance of finding global minimum
- Final value typically: f ≈ 0.0 - 2.0
- May find solution very close to (0, 0)
- More function evaluations but better solution quality

## Key Observations

1. **Restart Activity**: With verbosity enabled, you'll see messages like:
   ```
   Stagnation detected at iteration 145. Performing random restart. Best global: 2.3456
   ```

2. **History Analysis**: The optimization history shows "jumps" when restarts occur, but the global best steadily improves.

3. **Trade-offs**:
   - More function evaluations (typically 2-5x more)
   - Better final solution quality (10-90% improvement typical)
   - Automatic exploration of parameter space

## Tuning for Your Problem

### For expensive function evaluations:
```rust
opt.set_stagnation_threshold(100);  // Fewer restarts
opt.set_random_restart(true);
opt.minimize(Some(300));            // Fewer total iterations
```

### For cheap functions / need global optimum:
```rust
opt.set_stagnation_threshold(20);   // More aggressive restarts
opt.set_adaptive_simplex(true);     // Better initial exploration
opt.set_random_restart(true);
opt.minimize(Some(1000));           // More iterations for multiple restarts
```

### For well-behaved unimodal functions:
```rust
// Don't enable random restarts - wastes function evaluations
opt.set_random_restart(false);
opt.set_adaptive_simplex(false);
opt.minimize(Some(200));
```

## When This Helps Most

Random restarts are especially effective for:

1. **Multimodal functions** - Multiple local minima like Rastrigin, Ackley
2. **Unknown landscapes** - When you don't know the function topology
3. **Poor initial guesses** - Starting point far from any good minimum
4. **High-dimensional problems** - More likely to have multiple minima
5. **Rugged surfaces** - Many small local features

## Performance Tips

1. **Set verbosity=1** initially to understand restart behavior
2. **Adjust stagnation threshold** based on convergence patterns
3. **Increase max_iters** when enabling restarts (at least 3-5x stagnation_threshold)
4. **Monitor the history** to see if restarts are helping or just wasting evaluations
5. **Try adaptive simplex alone first** - it may be enough for some problems
