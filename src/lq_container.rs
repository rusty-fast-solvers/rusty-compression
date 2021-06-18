//! A container for LQ Decompositions.

use crate::permutation::*;
use crate::prelude::CompressionType;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{s, Array1, Array2};
use num::ToPrimitive;

pub struct LQContainer<A: ScalarType> {
    /// The L matrix from the QR Decomposition
    pub l: Array2<A>,
    /// The Q matrix from the LQ Decomposition
    pub q: Array2<A>,
    /// An index array. If ind[j] = k then the
    /// jth row of L * Q is identical to the
    /// kth row of the original matrix A.
    pub ind: Array1<usize>,
}

impl<A: ScalarType> LQContainer<A> {
    pub fn nrows(&self) -> usize {
        self.l.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.q.ncols()
    }

    pub fn rank(&self) -> usize {
        self.q.nrows()
    }

    /// If P A  = LQ, multiply P^T * L * Q to obtain the matrix A.
    pub fn to_mat(&self) -> Array2<A> {
        self.l.apply_permutation(self.ind.view(), MatrixPermutationMode::ROWINV).dot(
            &self
                .q
        )
    }

    pub fn compress(self, compression_type: CompressionType) -> Result<LQContainer<A>> {
        compress_lq(self, compression_type)
    }
}

fn compress_lq<A: ScalarType>(
    lq_container: LQContainer<A>,
    compression_type: CompressionType,
) -> Result<LQContainer<A>> {
    match compression_type {
        CompressionType::ADAPTIVE(tol) => compress_lq_tolerance(lq_container, tol),
        CompressionType::RANK(rank) => compress_lq_rank(lq_container, rank),
    }
}

fn compress_lq_rank<A: ScalarType>(
    lq_container: LQContainer<A>,
    mut max_rank: usize,
) -> Result<LQContainer<A>> {
    let (l, q, ind) = (lq_container.l, lq_container.q, lq_container.ind);

    if max_rank > q.nrows() {
        max_rank = q.nrows()
    }

    let q = q.slice_move(s![0..max_rank, ..]);
    let l = l.slice_move(s![.., 0..max_rank]);

    Ok(LQContainer { l, q, ind })
}

fn compress_lq_tolerance<A: ScalarType>(
    lq_container: LQContainer<A>,
    tol: f64,
) -> Result<LQContainer<A>> {
    assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

    let pos = lq_container
        .l
        .diag()
        .iter()
        .position(|&item| ((item / lq_container.l[[0, 0]]).abs()).to_f64().unwrap() < tol);

    match pos {
        Some(index) => compress_lq_rank(lq_container, index),
        None => Err("Could not compress operator to desired tolerance."),
    }
}

//#[cfg(test)]
//mod tests {

    //use super::*;
    //use crate::prelude::PivotedQR;
    //use crate::prelude::Random;
    //use crate::prelude::RelDiff;
    //use ndarray::Axis;

    //macro_rules! qr_compression_by_rank_tests {

        //($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

            //$(

        //#[test]
        //fn $name() {
            //let m = $dim.0;
            //let n = $dim.1;
            //let rank: usize = 30;

            //let sigma_max = 1.0;
            //let sigma_min = 1E-10;
            //let mut rng = rand::thread_rng();
            //let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

            //let qr = mat.pivoted_qr().unwrap().compress(CompressionType::RANK(rank)).unwrap();

            //// Compare with original matrix

            //assert!(qr.q.len_of(Axis(1)) == rank);
            //assert!(qr.r.len_of(Axis(0)) == rank);

            //assert!(qr.to_mat().rel_diff(&mat) < $tol);
        //}


            //)*

        //}
    //}

    //macro_rules! qr_compression_by_tol_tests {

        //($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

            //$(

        //#[test]
        //fn $name() {
            //let m = $dim.0;
            //let n = $dim.1;

            //let sigma_max = 1.0;
            //let sigma_min = 1E-10;
            //let mut rng = rand::thread_rng();
            //let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

            //let qr = mat.pivoted_qr().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();

            //// Compare with original matrix

            //assert!(qr.to_mat().rel_diff(&mat) < 5.0 * $tol);

            //// Make sure new rank is smaller than original rank

            //assert!(qr.q.ncols() < m.min(n));
        //}


            //)*

        //}
    //}

    //qr_compression_by_rank_tests! {
        //test_qr_compression_by_rank_f32_thin: f32, (100, 50), 1E-4,
        //test_qr_compression_by_rank_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        //test_qr_compression_by_rank_f64_thin: f64, (100, 50), 1E-4,
        //test_qr_compression_by_rank_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        //test_qr_compression_by_rank_f32_thick: f32, (50, 100), 1E-4,
        //test_qr_compression_by_rank_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        //test_qr_compression_by_rank_f64_thick: f64, (50, 100), 1E-4,
        //test_qr_compression_by_rank_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    //}

    //qr_compression_by_tol_tests! {
        //test_qr_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        //test_qr_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        //test_qr_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        //test_qr_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        //test_qr_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        //test_qr_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        //test_qr_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        //test_qr_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    //}
//}
