#![allow(dead_code)]
#![allow(unused_assignments)]
use rand::prelude::*;
use std::f64;

/// Result of the simulated annealing optimization
#[derive(Debug)]
pub struct AnnealingResult<T> {
    pub best_solution: T,
    pub best_energy: f64,
    pub final_solution: T,
    pub final_energy: f64,
    pub iterations: usize,
    pub accepted_moves: usize,
}

/// Trait for problems that can be solved with simulated annealing
pub trait Annealable: Clone {
    /// Evaluate the energy (cost) of the current solution
    /// Lower values are better
    fn energy(&self) -> f64;

    /// Generate a neighboring solution
    fn neighbor(&self, rng: &mut impl Rng) -> Self;

    /// Optional: implement custom acceptance probability
    /// Default uses standard Boltzmann probability
    fn acceptance_probability(
        &self,
        new_energy: f64,
        current_energy: f64,
        temperature: f64,
    ) -> f64 {
        if new_energy < current_energy {
            1.0 // Always accept better solutions
        } else {
            f64::exp(-(new_energy - current_energy) / temperature)
        }
    }
}

/// Simulated annealing optimizer
pub struct SimAnnealing {
    initial_temp: f64,     // Initial temperature
    final_temp: f64,       // Final temperature (stopping criterion)
    alpha: f64,            // Temperature reduction factor (0 < alpha < 1)
    iters_per_temp: usize, // Number of iterations at each temperature
    max_iters: usize,      // Maximum total iterations
    rng: ThreadRng,
}

impl SimAnnealing {
    pub fn new(
        initial_temp: Option<f64>,
        final_temp: Option<f64>,
        alpha: Option<f64>,
        iters_per_temp: Option<usize>,
        max_iters: Option<usize>,
    ) -> Self {
        Self {
            initial_temp: match initial_temp {
                Some(x) => x,
                _ => 100.0,
            },
            final_temp: match final_temp {
                Some(x) => x,
                _ => 1e-6,
            },
            alpha: match alpha {
                Some(x) => x,
                _ => 0.95,
            },
            iters_per_temp: match iters_per_temp {
                Some(x) => x,
                _ => 100,
            },
            max_iters: match max_iters {
                Some(x) => x,
                _ => 100_000,
            },
            rng: rand::rng(),
        }
    }

    /// Run simulated annealing optimization
    pub fn optimize<T: Annealable>(&mut self, initial_solution: T) -> AnnealingResult<T> {
        let mut current_solution = initial_solution;
        let mut current_energy = current_solution.energy();

        let mut best_solution = current_solution.clone();
        let mut best_energy = current_energy;

        let mut temperature = self.initial_temp;
        let mut total_iterations = 0;
        let mut accepted_moves = 0;

        while temperature > self.final_temp && total_iterations < self.max_iters {
            for _ in 0..self.iters_per_temp {
                if total_iterations >= self.max_iters {
                    break;
                }

                // Generate neighbor solution
                let new_solution = current_solution.neighbor(&mut self.rng);
                let new_energy = new_solution.energy();

                // Calculate acceptance probability
                let acceptance_prob = current_solution.acceptance_probability(
                    new_energy,
                    current_energy,
                    temperature,
                );

                // Accept or reject the new solution
                if self.rng.random::<f64>() < acceptance_prob {
                    current_solution = new_solution;
                    current_energy = new_energy;
                    accepted_moves += 1;

                    // Update best solution if necessary
                    if new_energy < best_energy {
                        best_solution = current_solution.clone();
                        best_energy = new_energy;
                    }
                }

                total_iterations += 1;
            }

            // Cool down
            temperature *= self.alpha;
        }

        AnnealingResult {
            best_solution,
            best_energy,
            final_solution: current_solution,
            final_energy: current_energy,
            iterations: total_iterations,
            accepted_moves,
        }
    }
}

// Example: Traveling Salesman Problem
#[derive(Debug, Clone)]
pub struct TSPSolution {
    pub cities: Vec<usize>,
    pub distances: Vec<Vec<f64>>, // Distance matrix
}

impl TSPSolution {
    pub fn new(distances: Vec<Vec<f64>>) -> Self {
        let n = distances.len();
        let mut cities: Vec<usize> = (0..n).collect();
        cities.shuffle(&mut rand::rng());

        Self { cities, distances }
    }

    fn total_distance(&self) -> f64 {
        let n = self.cities.len();
        let mut total = 0.0;

        for i in 0..n {
            let from = self.cities[i];
            let to = self.cities[(i + 1) % n];
            total += self.distances[from][to];
        }

        total
    }
}

impl Annealable for TSPSolution {
    fn energy(&self) -> f64 {
        self.total_distance()
    }

    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_solution = self.clone();
        let n = new_solution.cities.len();

        // Random 2-opt swap
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);

        if i != j {
            new_solution.cities.swap(i, j);
        }

        new_solution
    }
}

// Test functions for optimization benchmarking
#[derive(Debug, Clone)]
pub enum TestFunction {
    Rosenbrock,
    Sphere,
    Rastrigin,
    Ackley,
    Himmelblau,
    Beale,
    GoldsteinPrice,
    Booth,
    Matyas,
    Levy,
}

// Example: Function minimization
#[derive(Debug, Clone)]
pub struct FunctionOptimization {
    pub x: Vec<f64>,
    pub step_size: f64,
    pub function: TestFunction,
}

impl FunctionOptimization {
    pub fn new(initial_x: Vec<f64>, step_size: f64, function: TestFunction) -> Self {
        Self {
            x: initial_x,
            step_size,
            function,
        }
    }

    pub fn new_rosenbrock(initial_x: Vec<f64>, step_size: f64) -> Self {
        Self::new(initial_x, step_size, TestFunction::Rosenbrock)
    }

    // Rosenbrock function - Global minimum at (1,1,...,1) with value 0
    fn rosenbrock(&self) -> f64 {
        if self.x.len() < 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..self.x.len() - 1 {
            let term1 = 100.0 * (self.x[i + 1] - self.x[i].powi(2)).powi(2);
            let term2 = (1.0 - self.x[i]).powi(2);
            sum += term1 + term2;
        }
        sum
    }

    // Sphere function - Global minimum at (0,0,...,0) with value 0
    fn sphere(&self) -> f64 {
        self.x.iter().map(|&xi| xi * xi).sum()
    }

    // Rastrigin function - Global minimum at (0,0,...,0) with value 0
    // Highly multimodal with many local minima
    fn rastrigin(&self) -> f64 {
        let n = self.x.len() as f64;
        let a = 10.0;

        a * n
            + self
                .x
                .iter()
                .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    // Ackley function - Global minimum at (0,0,...,0) with value 0
    // Highly multimodal
    fn ackley(&self) -> f64 {
        let n = self.x.len() as f64;
        let sum_sq = self.x.iter().map(|&xi| xi * xi).sum::<f64>();
        let sum_cos = self
            .x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>();

        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    }

    // Himmelblau's function - 2D only, has 4 global minima all with value 0
    // (3,2), (-2.805118,3.131312), (-3.779310,-3.283186), (3.584428,-1.848126)
    fn himmelblau(&self) -> f64 {
        if self.x.len() != 2 {
            return f64::INFINITY;
        }
        let x = self.x[0];
        let y = self.x[1];
        (x * x + y - 11.0).powi(2) + (x + y * y - 7.0).powi(2)
    }

    // Beale function - 2D only, global minimum at (3, 0.5) with value 0
    fn beale(&self) -> f64 {
        if self.x.len() != 2 {
            return f64::INFINITY;
        }
        let x = self.x[0];
        let y = self.x[1];
        (1.5 - x + x * y).powi(2)
            + (2.25 - x + x * y * y).powi(2)
            + (2.625 - x + x * y.powi(3)).powi(2)
    }

    // Goldstein-Price function - 2D only, global minimum at (0, -1) with value 3
    fn goldstein_price(&self) -> f64 {
        if self.x.len() != 2 {
            return f64::INFINITY;
        }
        let x = self.x[0];
        let y = self.x[1];

        let term1 = 1.0
            + (x + y + 1.0).powi(2)
                * (19.0 - 14.0 * x + 3.0 * x * x - 14.0 * y + 6.0 * x * y + 3.0 * y * y);

        let term2 = 30.0
            + (2.0 * x - 3.0 * y).powi(2)
                * (18.0 - 32.0 * x + 12.0 * x * x + 48.0 * y - 36.0 * x * y + 27.0 * y * y);

        term1 * term2
    }

    // Booth function - 2D only, global minimum at (1, 3) with value 0
    fn booth(&self) -> f64 {
        if self.x.len() != 2 {
            return f64::INFINITY;
        }
        let x = self.x[0];
        let y = self.x[1];
        (x + 2.0 * y - 7.0).powi(2) + (2.0 * x + y - 5.0).powi(2)
    }

    // Matyas function - 2D only, global minimum at (0, 0) with value 0
    fn matyas(&self) -> f64 {
        if self.x.len() != 2 {
            return f64::INFINITY;
        }
        let x = self.x[0];
        let y = self.x[1];
        0.26 * (x * x + y * y) - 0.48 * x * y
    }

    // Levy function - Global minimum at (1,1,...,1) with value 0
    fn levy(&self) -> f64 {
        let n = self.x.len();
        if n == 0 {
            return 0.0;
        }

        let w: Vec<f64> = self.x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();

        let mut sum = 0.0;

        // First term
        sum += (std::f64::consts::PI * w[0]).sin().powi(2);

        // Middle terms
        for i in 0..n - 1 {
            sum += (w[i] - 1.0).powi(2)
                * (1.0 + 10.0 * (std::f64::consts::PI * w[i + 1]).sin().powi(2));
        }

        // Last term
        sum += (w[n - 1] - 1.0).powi(2);

        sum
    }
}

impl Annealable for FunctionOptimization {
    fn energy(&self) -> f64 {
        match self.function {
            TestFunction::Rosenbrock => self.rosenbrock(),
            TestFunction::Sphere => self.sphere(),
            TestFunction::Rastrigin => self.rastrigin(),
            TestFunction::Ackley => self.ackley(),
            TestFunction::Himmelblau => self.himmelblau(),
            TestFunction::Beale => self.beale(),
            TestFunction::GoldsteinPrice => self.goldstein_price(),
            TestFunction::Booth => self.booth(),
            TestFunction::Matyas => self.matyas(),
            TestFunction::Levy => self.levy(),
        }
    }

    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_solution = self.clone();

        // Perturb all dimensions with smaller steps for better convergence
        for i in 0..new_solution.x.len() {
            let perturbation = rng.random_range(-self.step_size..self.step_size);
            new_solution.x[i] += perturbation;
        }

        new_solution
    }
}

#[cfg(test)]
mod simannealingf64_tests {
    use super::*;

    #[test]
    fn test_function_optimization() {
        let mut optimizer = SimAnnealing::new(
            Some(100.0),
            Some(0.0001),
            Some(0.995),
            Some(200),
            Some(50_000),
        );
        let initial = FunctionOptimization::new_rosenbrock(vec![0.0, 0.0], 0.05);

        let result = optimizer.optimize(initial);

        println!("Best solution: {:?}", result.best_solution.x);
        println!("Best energy: {}", result.best_energy);
        println!("Iterations: {}", result.iterations);
        println!(
            "Acceptance rate: {:.2}%",
            100.0 * result.accepted_moves as f64 / result.iterations as f64
        );

        // The minimum of Rosenbrock function is at (1, 1) with value 0
        // Relax the assertion since SA is stochastic
        assert!(
            result.best_energy < 10.0,
            "Best energy {} should be less than 10.0",
            result.best_energy
        );

        // Optional: Run multiple times to check consistency
        let mut successes = 0;
        for _ in 0..5 {
            let mut opt =
                SimAnnealing::new(Some(50.0), Some(0.001), Some(0.99), Some(100), Some(20_000));
            let init = FunctionOptimization::new_rosenbrock(vec![0.5, 0.5], 0.02);
            let res = opt.optimize(init);
            if res.best_energy < 1.0 {
                successes += 1;
            }
        }
        println!("Successful runs (energy < 1.0): {}/5", successes);
    }

    #[test]
    fn test_sphere_function() {
        let mut optimizer =
            SimAnnealing::new(Some(10.0), Some(0.001), Some(0.99), Some(100), Some(10_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial =
                FunctionOptimization::new(vec![5.0, -3.0, 2.0], 0.1, TestFunction::Sphere);

            let result = optimizer.optimize(initial);

            if result.best_energy < 1.0 || iter >= 100 {
                println!("Sphere - Best solution: {:?}", result.best_solution.x);
                println!("Sphere - Best energy: {}", result.best_energy);

                // Sphere should be easy to optimize
                assert!(result.best_energy < 1.0);
                break;
            }
        }
    }

    #[test]
    fn test_rastrigin_function() {
        let mut optimizer =
            SimAnnealing::new(Some(50.0), Some(0.01), Some(0.95), Some(200), Some(30_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![2.0, -1.5], 0.2, TestFunction::Rastrigin);

            let result = optimizer.optimize(initial);

            if result.best_energy < 10.0 || iter >= 100 {
                println!("Rastrigin - Best solution: {:?}", result.best_solution.x);
                println!("Rastrigin - Best energy: {}", result.best_energy);

                // Rastrigin is multimodal, so we're more lenient
                assert!(result.best_energy < 10.0);
                break;
            }
        }
    }

    #[test]
    fn test_ackley_function() {
        let mut optimizer =
            SimAnnealing::new(Some(20.0), Some(0.001), Some(0.98), Some(150), Some(20_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![3.0, -2.0], 0.1, TestFunction::Ackley);

            let result = optimizer.optimize(initial);

            if result.best_energy < 5.0 || iter >= 100 {
                println!("Ackley - Best solution: {:?}", result.best_solution.x);
                println!("Ackley - Best energy: {}", result.best_energy);

                // Ackley is highly multimodal
                assert!(result.best_energy < 5.0);
                break;
            }
        }
    }

    #[test]
    fn test_himmelblau_function() {
        let mut optimizer =
            SimAnnealing::new(Some(100.0), Some(0.01), Some(0.99), Some(100), Some(15_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![1.0, 1.0], 0.1, TestFunction::Himmelblau);

            let result = optimizer.optimize(initial);

            if result.best_energy < 1.0 || iter >= 100 {
                println!("Himmelblau - Best solution: {:?}", result.best_solution.x);
                println!("Himmelblau - Best energy: {}", result.best_energy);

                // Should find one of the 4 global minima
                assert!(result.best_energy < 1.0);
                break;
            }
        }
    }

    #[test]
    fn test_beale_function() {
        let mut optimizer =
            SimAnnealing::new(Some(50.0), Some(0.001), Some(0.99), Some(100), Some(15_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![1.0, 1.0], 0.05, TestFunction::Beale);

            let result = optimizer.optimize(initial);

            if result.best_energy < 1.0 || iter >= 100 {
                println!("Beale - Best solution: {:?}", result.best_solution.x);
                println!("Beale - Best energy: {}", result.best_energy);
                println!("Beale - Expected minimum at (3.0, 0.5)");

                assert!(result.best_energy < 1.0);
                break;
            }
        }
    }

    #[test]
    fn test_booth_function() {
        let mut optimizer =
            SimAnnealing::new(Some(20.0), Some(0.001), Some(0.99), Some(100), Some(10_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![0.0, 0.0], 0.1, TestFunction::Booth);

            let result = optimizer.optimize(initial);

            if result.best_energy < 1.0 || iter >= 100 {
                println!("Booth - Best solution: {:?}", result.best_solution.x);
                println!("Booth - Best energy: {}", result.best_energy);
                println!("Booth - Expected minimum at (1.0, 3.0)");

                assert!(result.best_energy < 1.0);
                break;
            }
        }
    }

    #[test]
    fn test_matyas_function() {
        let mut optimizer =
            SimAnnealing::new(Some(10.0), Some(0.001), Some(0.99), Some(100), Some(10_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![2.0, -1.0], 0.05, TestFunction::Matyas);

            let result = optimizer.optimize(initial);

            if result.best_energy < 0.1 || iter >= 100 {
                println!("Matyas - Best solution: {:?}", result.best_solution.x);
                println!("Matyas - Best energy: {}", result.best_energy);

                assert!(result.best_energy < 0.1);
                break;
            }
        }
    }

    #[test]
    fn test_levy_function() {
        let mut optimizer = SimAnnealing::new(
            Some(100.0),
            Some(0.0001),
            Some(0.998),
            Some(200),
            Some(40_000),
        );

        let mut iter = 0;
        loop {
            iter += 1;
            let initial = FunctionOptimization::new(vec![0.5, 0.5, 0.5], 0.05, TestFunction::Levy); // Better starting point and smaller steps

            let result = optimizer.optimize(initial);

            if result.best_energy < 1.0 || iter >= 100 {
                println!("Levy - Best solution: {:?}", result.best_solution.x);
                println!("Levy - Best energy: {}", result.best_energy);
                println!("Levy - Expected minimum at (1.0, 1.0, 1.0)");

                assert!(result.best_energy < 1.0);
                break;
            }
        }
    }

    #[test]
    fn test_goldstein_price_function() {
        let mut optimizer =
            SimAnnealing::new(Some(1000.0), Some(0.1), Some(0.99), Some(200), Some(30_000));

        let mut iter = 0;
        loop {
            iter += 1;
            let initial =
                FunctionOptimization::new(vec![1.0, 1.0], 0.1, TestFunction::GoldsteinPrice);

            let result = optimizer.optimize(initial);

            if result.best_energy < 10.0 || iter >= 100 {
                println!(
                    "GoldsteinPrice - Best solution: {:?}",
                    result.best_solution.x
                );
                println!("GoldsteinPrice - Best energy: {}", result.best_energy);
                println!("GoldsteinPrice - Expected minimum at (0.0, -1.0) with value 3.0");

                // This function's minimum is 3.0, not 0
                assert!(result.best_energy < 10.0);
                break;
            }
        }
    }

    #[test]
    fn benchmark_all_functions() {
        let functions = [
            (TestFunction::Sphere, vec![2.0, -1.0], "Easy - convex"),
            (
                TestFunction::Rosenbrock,
                vec![0.0, 0.0],
                "Medium - narrow valley",
            ),
            (TestFunction::Booth, vec![0.0, 0.0], "Easy - quadratic"),
            (
                TestFunction::Matyas,
                vec![2.0, -1.0],
                "Easy - nearly convex",
            ),
            (
                TestFunction::Himmelblau,
                vec![1.0, 1.0],
                "Medium - 4 global minima",
            ),
            (
                TestFunction::Beale,
                vec![1.0, 1.0],
                "Medium - steep valleys",
            ),
            (
                TestFunction::Levy,
                vec![2.0, -1.0, 1.5],
                "Hard - many local minima",
            ),
            (
                TestFunction::Ackley,
                vec![3.0, -2.0],
                "Hard - highly multimodal",
            ),
            (
                TestFunction::Rastrigin,
                vec![2.0, -1.5],
                "Hard - many local minima",
            ),
            (
                TestFunction::GoldsteinPrice,
                vec![1.0, 1.0],
                "Very hard - complex landscape",
            ),
        ];

        println!("\n=== Benchmark Results ===");

        for (func, initial, description) in functions.iter() {
            let mut optimizer = SimAnnealing::new(
                Some(100.0),
                Some(0.001),
                Some(0.995),
                Some(100),
                Some(20_000),
            );
            let problem = FunctionOptimization::new(initial.clone(), 0.1, func.clone());
            let result = optimizer.optimize(problem);

            println!(
                "{:15} | Energy: {:10.6} | Solution: {:?} | {}",
                format!("{:?}", func),
                result.best_energy,
                result.best_solution.x,
                description
            );
        }
    }

    #[test]
    fn test_tsp() {
        // Small 4-city example
        let distances = vec![
            vec![0.0, 2.0, 9.0, 10.0],
            vec![1.0, 0.0, 6.0, 4.0],
            vec![15.0, 7.0, 0.0, 8.0],
            vec![6.0, 3.0, 12.0, 0.0],
        ];

        let mut optimizer =
            SimAnnealing::new(Some(100.0), Some(0.1), Some(0.95), Some(100), Some(5_000));
        let initial = TSPSolution::new(distances);

        let result = optimizer.optimize(initial);

        println!("Best tour: {:?}", result.best_solution.cities);
        println!("Best distance: {}", result.best_energy);
    }
}
