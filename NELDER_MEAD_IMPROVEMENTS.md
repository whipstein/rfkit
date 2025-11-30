# Nelder-Mead Bounded Optimization Improvements

## Summary of Changes

I've enhanced the Nelder-Mead bounded minimization algorithm to significantly reduce the likelihood of getting stuck in local minima. The improvements include:

### 1. **Random Restart Mechanism** (Main Feature)
When the algorithm detects stagnation (no improvement over many iterations), it automatically restarts from a new random location in the search space while keeping track of the best solution found globally.

### 2. **Stagnation Detection**
The algorithm monitors convergence progress and counts consecutive iterations without significant improvement. When stagnation exceeds a threshold, a restart is triggered.

### 3. **Global Best Tracking**
Across all restarts, the algorithm maintains the best solution ever found and returns it at the end, ensuring you never lose a good solution due to a restart.

### 4. **Adaptive Initial Simplex**
The initial simplex is now scaled based on the parameter bounds (10% of each parameter's range), providing better initial exploration of the search space.

### 5. **Improved Random Sampling**
Uses a deterministic pseudo-random number generator (xorshift) that provides good coverage of the search space while remaining reproducible.

## Key Algorithm Features

### Stagnation Detection
- Monitors if the best objective function value hasn't improved by more than `1e-10` over consecutive iterations
- Default threshold: 50 iterations without improvement triggers a restart
- Configurable via `set_stagnation_threshold()`

### Random Restart Strategy
- When stagnation is detected, generates a completely new random starting point within bounds
- Creates a new randomized simplex around that point (10% of parameter range)
- Continues optimization from the new location
- Can perform multiple restarts during a single `minimize()` call

### Global Best Retention
- Always tracks the best solution found across all restarts
- Returns the global best even if the final simplex converged to a worse local minimum
- Ensures monotonic improvement in solution quality with more iterations

## New API Methods

```rust
// Enable/disable random restart (disabled by default for backward compatibility)
optimizer.set_random_restart(true);

// Enable/disable adaptive simplex sizing (disabled by default for backward compatibility)
optimizer.set_adaptive_simplex(true);

// Set how many stagnant iterations before restart (default: 50)
optimizer.set_stagnation_threshold(100);
```

## Usage Example

```rust
use ndarray::array;
use rfkit::minimize::f64::nelder_mead_bounded::NelderMeadBounded;
use rfkit::minimize::Minimizer;

// Your objective function
let f = |x: &Array1<f64>| {
    // Your complex function here
    x[0].powi(2) + x[1].powi(2)
};

// Initial guess and bounds
let x0 = array![1.0, 1.0];
let scale = array![1.0, 1.0];
let lb = array![0.0, 0.0];
let ub = array![10.0, 10.0];

let mut optimizer = NelderMeadBounded::new(x0, scale, lb, ub, f);

// Configure anti-local-minima features
optimizer.set_random_restart(true);          // Enable random restarts
optimizer.set_adaptive_simplex(true);        // Enable adaptive initial simplex
optimizer.set_stagnation_threshold(50);      // Restart after 50 stagnant iterations
optimizer.set_verbosity(1);                  // See restart messages

// Run optimization
let result = optimizer.minimize(Some(1000));

println!("Best solution: {:?}", result.xmin());
println!("Best value: {}", result.fmin());
```

## When Random Restarts Help

Random restarts are particularly effective for:
- **Multimodal functions** - Functions with multiple local minima
- **Rugged landscapes** - Objective functions with many small bumps
- **High-dimensional problems** - Where the search space is large
- **Poor initial guesses** - When starting point is far from global optimum

## When to Disable Random Restarts

You might want to disable random restarts (`set_random_restart(false)`) when:
- The objective function is known to be unimodal (single minimum)
- You have a very good initial guess near the optimum
- Function evaluations are extremely expensive and you want to minimize calls
- You want the algorithm to fully converge to the nearest local minimum

## Tuning Parameters

### Stagnation Threshold
- **Lower values (20-30)**: More aggressive restarts, better global exploration, more function evaluations
- **Higher values (100-200)**: Fewer restarts, more local refinement, fewer function evaluations
- **Default (50)**: Good balance for most problems

### Maximum Iterations
- Should be increased when using random restarts to allow multiple restart cycles
- Recommended: At least `stagnation_threshold * 5` for multi-restart optimization
- Example: If threshold is 50, use at least 250 iterations

## Performance Characteristics

### Computational Cost
- Random restarts increase function evaluations but improve solution quality
- Each restart costs `n+1` function evaluations (where n = number of parameters)
- Trade-off between computational cost and solution quality is controllable

### Convergence Behavior
- With restarts: May see objective function "jump" in history when restarts occur
- Without restarts: Monotonic convergence to nearest local minimum
- Global best is always monotonically improving

## Technical Details

### Barrier Method
The algorithm still uses logarithmic barrier functions to handle bounds:
```
f_augmented(x) = f(x) - μ * Σ[ln(ub[i] - x[i]) + ln(x[i] - lb[i])]
```

### Random Number Generation
- Uses xorshift algorithm for pseudo-random numbers
- Deterministic based on iteration count
- Good statistical properties without external dependencies

### Simplex Generation
- Initial simplex scaled by parameter ranges
- Restart simplices are randomized around new center points
- All points checked against bounds

## Backward Compatibility

All existing code will continue to work without modification:
- Random restart is **disabled by default** (opt-in for new behavior)
- Adaptive simplex is **disabled by default** (opt-in for new behavior)
- Default stagnation threshold is 50 iterations
- Existing tests pass unchanged

To enable the anti-local-minima features:
```rust
optimizer.set_random_restart(true);
optimizer.set_adaptive_simplex(true);
```

## Testing

The existing test suite has been preserved. You can verify the improvements work by:
1. Running existing tests: `cargo test nelder_mead_bounded`
2. Comparing convergence with/without restarts on your specific problems
3. Monitoring verbosity output to see when restarts occur

## Future Enhancements (Not Implemented)

Possible further improvements:
- Adaptive stagnation threshold based on problem difficulty
- Multi-start from Latin Hypercube Sampling of initial points
- Simulated annealing for accepting worse solutions
- Adaptive Nelder-Mead coefficients (α, β, γ, ρ)
- Gradient-based local refinement
- Population-based restarts (maintain multiple simplices)
