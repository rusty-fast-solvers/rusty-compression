use rusty_compression::prelude::RandomMatrix;
use ndarray::Axis;

pub fn main() {

//    let m: usize = 100;
//    let n: usize = 50;
//
//    let tol = 1E-5;
//
//    let sigma_max = 1.0;
//    let sigma_min = 1E-15;
//
//
//    let mut rng = rand::thread_rng();
//    let mat = f64::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);
//
//    let (a, bt) = mat.compress_svd(CompressionType::ADAPTIVE(tol)).unwrap();
//
//    let rel_diff = a.dot(&bt).rel_diff(&mat);
//
//
//    println!("Compressiong a matrix with rank {}", n);
//    println!("The rank of the compressed matrix is {}.", a.len_of(Axis(1)));
//    println!("The estimated compression error is {}.", rel_diff);
//
}

