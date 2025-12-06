use rfkit::float::RFComplex;
use num::complex::Complex64;
use rfkit::mycomplex::MyComplex;

fn complex_operations<T: RFComplex>(z1: &T, z2: &T) -> (T, T::Real, T::Real) {
    // Demonstrate various operations
    let sum = z1.clone() + z2.clone();
    let magnitude = z1.abs();
    let phase = z1.arg();

    println!("z1 = {}", z1);
    println!("z2 = {}", z2);
    println!("z1 + z2 = {}", sum);
    println!("|z1| = {}", magnitude);
    println!("arg(z1) = {}", phase);
    println!("z1* = {}", z1.conj());
    println!("exp(z1) = {}", z1.exp());
    println!("ln(z1) = {}", z1.ln());
    println!("sqrt(z1) = {}", z1.sqrt());
    println!("sin(z1) = {}", z1.sin());

    (sum, magnitude, phase)
}

fn main() {
    println!("=== Using Complex<f64> ===");
    let c1 = Complex64::from_f64(3.0, 4.0);
    let c2 = Complex64::from_f64(1.0, 2.0);
    complex_operations(&c1, &c2);

    println!("\n=== Using MyComplex ===");
    let m1 = MyComplex::from_f64(3.0, 4.0);
    let m2 = MyComplex::from_f64(1.0, 2.0);
    complex_operations(&m1, &m2);

    println!("\n=== Trigonometric and Hyperbolic Functions ===");
    let z = Complex64::from_f64(1.0, 0.5);
    println!("z = {}", z);
    println!("sin(z) = {}", z.sin());
    println!("cos(z) = {}", z.cos());
    println!("tan(z) = {}", z.tan());
    println!("sinh(z) = {}", z.sinh());
    println!("cosh(z) = {}", z.cosh());
    println!("tanh(z) = {}", z.tanh());
}
