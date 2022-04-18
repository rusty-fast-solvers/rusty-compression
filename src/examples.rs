//! # Library examples
//! 
//! We provide some examples in the `examples` subdirectory. To run them
//! use `cargo run --example <example_filename>`.
//! 
//! ### Computing a rank k interpolative decomposition of an $m\times n$ matrix.
//!
//! This example computes the rank $k$ two-sided interpolative decomposition of a 
//! given matrix and prints the relative distance between the compressed
//! interpolative decmposition and the original matrix. 
//! The corresponding code is in the file `interpolative_decomposition.rs`.
//! 
//! ### Adaptive range sampling of a matrix.
//! 
//! This code adaptively samples the range of a matrix up to a given error tolerance
//! and uses the range estimate to compute an approximate interpolative decomposition.
//! It generates a residual curve comparing the probabilistic erro bound from the adaptive
//! sampling and the exact relative error. The convergence curve is saved in the file
//! `residuals.png`.