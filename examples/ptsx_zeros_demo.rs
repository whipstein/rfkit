use ndarray::prelude::*;
use rfkit::pts::{Points, Pts};

fn main() {
    // 1D array (vector)
    // let zeros_1d: Points<f64, Ix1> = Points::zeros(10);
    // println!("1D zeros shape: {:?}", zeros_1d.0.shape());
    // println!("1D zeros: {:?}\n", zeros_1d.0);

    // 2D array (matrix)
    // let zeros_2d: Points<f64, Ix2> = Points::zeros((3, 4));
    // println!("2D zeros shape: {:?}", zeros_2d.0.shape());
    // println!("2D zeros:\n{:?}\n", zeros_2d.0);

    // 3D array
    let zeros_3d: Points<f64, Ix3> = Points::zeros((2, 3, 4));
    println!("3D zeros shape: {:?}", zeros_3d.0.shape());
    println!("3D zeros:\n{:?}\n", zeros_3d.0);

    // 4D array
    // let zeros_4d: Points<f64, Ix4> = Points::zeros((2, 2, 2, 2));
    // println!("4D zeros shape: {:?}", zeros_4d.0.shape());
    // println!("4D zeros:\n{:?}\n", zeros_4d.0);

    // Also works with Complex64
    use num::complex::Complex64;
    let zeros_complex: Points<Complex64, Ix3> = Points::zeros((2, 3, 4));
    println!("Complex 3D zeros shape: {:?}", zeros_complex.0.shape());
    println!("Complex 3D zeros:\n{:?}", zeros_complex.0);
}
