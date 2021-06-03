use ndarray::{Array2, Axis};
use ndarray_linalg::UVTFlag;
use ndarray_linalg::*;
use rand::Rng;

pub fn main() {
    let nrows = 10;
    let ncols = 5;

    let flag = UVTFlag::Some;

    let mut rng = rand::thread_rng();
    let mut mat = Array2::<f64>::zeros((nrows, ncols));
    mat.map_inplace(|item| *item = rng.gen::<f64>());

    let (u_mat, sigma, vt_mat) = mat.svddc(flag).unwrap();

    let u_mat = u_mat.unwrap();
    let vt_mat = vt_mat.unwrap();

    println!(
        "Shape: {}x{}",
        vt_mat.len_of(Axis(0)),
        vt_mat.len_of(Axis(1))
    );

    println!("Success.")
}

