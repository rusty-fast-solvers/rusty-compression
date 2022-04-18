//! Computing the interpolative decomposition of a matrix.
//!
//! This example computes the rank k interpolative decomposition of an $m\times n$ matrix.
//!

use rusty_compression::*;

pub fn main() {

    // We initialize a random number generator.
    let mut rng = rand::thread_rng();

    // The dimension of the matrix for which we want to
    // compute a low-rank approximation.
    let dimension = (500, 100);

    // The compression rank.
    let k = 20;

    // Generate a random matrix with singular values logarithmically
    // distributed between 1 and 1E-10.
    let mat = f64::random_approximate_low_rank_matrix(dimension, 1.0, 1E-10, &mut rng);

    // Compute the pivoted QR decomposition of the matrix.
    let qr = QR::<f64>::compute_from(mat.view()).expect("Could not compress the matrix.");

    // We now compress the pivoted QR decomposition to only include the k most significant
    // basis vectors of the range.
    let qr_compressed = qr.compress(CompressionType::RANK(k)).unwrap();

    // From the compressed representation we now compute the column interpolative decomposition
    let col_int_decomp = qr_compressed.column_id().unwrap();

    // We can also compute a two sided interpolative decomposition.
    let two_sided_int_decomp = col_int_decomp.two_sided_id().unwrap();

    // Let us compare the relative difference of the compressed operator to the original matrix.

    // First we multiply the factors of the two-sided decomposition to get back a matrix.
    // This should only be done for debugging purposes and non-probabilistic relative error
    // computations.
    let mat_approx = two_sided_int_decomp.to_mat();

    // We now compue the relative difference.
    let rel_diff = f64::rel_diff_fro(mat.view(), mat_approx.view());

    // Print the result.
    println!("The relative difference of the compressed and original matrix is {:1.2E}", rel_diff);


}
