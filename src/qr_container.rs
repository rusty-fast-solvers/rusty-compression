//! A container for QR Decompositions.

use crate::permutation::*;
use crate::prelude::CompressionType;
use crate::prelude::HasPivotedQR;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{s, Array1, Array2};
use num::ToPrimitive;

pub struct QRContainer<A: HasPivotedQR> {
    /// The Q matrix from the QR Decomposition
    pub q: Array2<A>,
    /// The R matrix from the QR Decomposition
    pub r: Array2<A>,
    /// An index array. If ind[j] = k then the
    /// jth column of Q * R is identical to the
    /// kth column of the original matrix A.
    pub ind: Array1<usize>,
}

impl<A: ScalarType> QRContainer<A> {
    pub fn nrows(&self) -> usize {
        self.q.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.r.ncols()
    }

    pub fn rank(&self) -> usize {
        self.q.ncols()
    }

    /// If A P = QR, multiply Q * R * P^T to obtain the matrix A.
    pub fn to_mat(&self) -> Array2<A> {
        self.q.dot(
            &self
                .r
                .apply_permutation(self.ind.view(), MatrixPermutationMode::COLINV),
        )
    }

    pub fn compress(self, compression_type: CompressionType) -> Result<QRContainer<A>> {
        compress_qr(self, compression_type)
    }
}

fn compress_qr<A: ScalarType>(
    qr_container: QRContainer<A>,
    compression_type: CompressionType,
) -> Result<QRContainer<A>> {
    match compression_type {
        CompressionType::ADAPTIVE(tol) => compress_qr_tolerance(qr_container, tol),
        CompressionType::RANK(rank) => compress_qr_rank(qr_container, rank),
    }
}

fn compress_qr_rank<A: ScalarType>(
    qr_container: QRContainer<A>,
    mut max_rank: usize,
) -> Result<QRContainer<A>> {
    let (q, r, ind) = (qr_container.q, qr_container.r, qr_container.ind);

    if max_rank > q.ncols() {
        max_rank = q.ncols()
    }

    let q = q.slice_move(s![.., 0..max_rank]);
    let r = r.slice_move(s![0..max_rank, ..]);

    Ok(QRContainer { q, r, ind })
}

fn compress_qr_tolerance<A: ScalarType>(
    qr_container: QRContainer<A>,
    tol: f64,
) -> Result<QRContainer<A>> {
    assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

    let pos = qr_container
        .r
        .diag()
        .iter()
        .position(|&item| ((item / qr_container.r[[0, 0]]).abs()).to_f64().unwrap() < tol);

    match pos {
        Some(index) => compress_qr_rank(qr_container, index),
        None => Err("Could not compress operator to desired tolerance."),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::prelude::PivotedQR;
    use crate::prelude::RandomMatrix;
    use crate::prelude::RelDiff;
    use ndarray::Axis;

    macro_rules! qr_compression_by_rank_tests {

        ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

            $(

        #[test]
        fn $name() {
            let m = $dim.0;
            let n = $dim.1;
            let rank: usize = 30;

            let sigma_max = 1.0;
            let sigma_min = 1E-10;
            let mut rng = rand::thread_rng();
            let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

            let qr = mat.pivoted_qr().unwrap().compress(CompressionType::RANK(rank)).unwrap();

            // Compare with original matrix

            assert!(qr.q.len_of(Axis(1)) == rank);
            assert!(qr.r.len_of(Axis(0)) == rank);

            assert!(qr.to_mat().rel_diff(&mat) < $tol);
        }


            )*

        }
    }

    macro_rules! qr_compression_by_tol_tests {

        ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

            $(

        #[test]
        fn $name() {
            let m = $dim.0;
            let n = $dim.1;

            let sigma_max = 1.0;
            let sigma_min = 1E-10;
            let mut rng = rand::thread_rng();
            let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

            let qr = mat.pivoted_qr().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();

            // Compare with original matrix

            assert!(qr.to_mat().rel_diff(&mat) < 5.0 * $tol);

            // Make sure new rank is smaller than original rank

            assert!(qr.q.ncols() < m.min(n));
        }


            )*

        }
    }

    qr_compression_by_rank_tests! {
        test_qr_compression_by_rank_f32_thin: f32, (100, 50), 1E-4,
        test_qr_compression_by_rank_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_qr_compression_by_rank_f64_thin: f64, (100, 50), 1E-4,
        test_qr_compression_by_rank_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_qr_compression_by_rank_f32_thick: f32, (50, 100), 1E-4,
        test_qr_compression_by_rank_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_qr_compression_by_rank_f64_thick: f64, (50, 100), 1E-4,
        test_qr_compression_by_rank_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }

    qr_compression_by_tol_tests! {
        test_qr_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_qr_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_qr_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_qr_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_qr_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        test_qr_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_qr_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_qr_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }
}
