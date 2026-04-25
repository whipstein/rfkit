use ndarray::prelude::*;
use num_complex::Complex64;
use rfkit::prelude::*;

fn main() {
    let zeros_3d: Points<f64, Ix3> = Points::zeros((2, 3, 4));
    println!("3D zeros shape: {:?}", zeros_3d.shape());
    println!("3D zeros:\n{:?}\n", zeros_3d.inner());

    let zeros_complex: Points<Complex64, Ix3> = Points::zeros((2, 3, 4));
    println!("Complex 3D zeros shape: {:?}", zeros_complex.shape());
    println!("Complex 3D zeros:\n{:?}", zeros_complex.inner());
}
