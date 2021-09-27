//! A container for QR Decompositions.

use crate::permutation::{ApplyPermutationToMatrix, MatrixPermutationMode};
use crate::pivoted_qr::PivotedQR;
use crate::CompressionType;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use num::ToPrimitive;
use rusty_base::types::{c32, c64, Result, Scalar};

pub struct QRData<A: Scalar> {
    /// The Q matrix from the QR Decomposition
    pub q: Array2<A>,
    /// The R matrix from the QR Decomposition
    pub r: Array2<A>,
    /// An index array. If ind[j] = k then the
    /// jth column of Q * R is identical to the
    /// kth column of the original matrix A.
    pub ind: Array1<usize>,
}

pub trait QR {
    type A: Scalar;

    fn nrows(&self) -> usize {
        self.get_q().nrows()
    }
    fn ncols(&self) -> usize {
        self.get_r().ncols()
    }
    fn rank(&self) -> usize {
        self.get_q().ncols()
    }
    fn to_mat(&self) -> Array2<Self::A> {
        self.get_q().dot(
            &self
                .get_r()
                .apply_permutation(self.get_ind(), MatrixPermutationMode::COLINV),
        )
    }

    fn compress_qr_rank(&self, mut max_rank: usize) -> Result<QRData<Self::A>> {
        let (q, r, ind) = (self.get_q(), self.get_r(), self.get_ind());

        if max_rank > q.ncols() {
            max_rank = q.ncols()
        }

        let q = q.slice_move(s![.., 0..max_rank]);
        let r = r.slice_move(s![0..max_rank, ..]);

        Ok(QRData {
            q: q.into_owned(),
            r: r.into_owned(),
            ind: ind.into_owned(),
        })
    }

    fn compress_qr_tolerance(&self, tol: f64) -> Result<QRData<Self::A>> {
        assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

        let pos = self
            .get_r()
            .diag()
            .iter()
            .position(|&item| ((item / self.get_r()[[0, 0]]).abs()).to_f64().unwrap() < tol);

        match pos {
            Some(index) => self.compress_qr_rank(index),
            None => Err("Could not compress operator to desired tolerance."),
        }
    }

    fn compress(&self, compression_type: CompressionType) -> Result<QRData<Self::A>> {
        match compression_type {
            CompressionType::ADAPTIVE(tol) => self.compress_qr_tolerance(tol),
            CompressionType::RANK(rank) => self.compress_qr_rank(rank),
        }
    }

    fn new(arr: ArrayView2<Self::A>) -> Result<QRData<Self::A>>;

    fn get_q(&self) -> ArrayView2<Self::A>;
    fn get_r(&self) -> ArrayView2<Self::A>;
    fn get_ind(&self) -> ArrayView1<usize>;

    fn get_q_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_ind_mut(&mut self) -> ArrayViewMut1<usize>;
}

macro_rules! qr_data_impl {
    ($scalar:ty) => {
        impl QR for QRData<$scalar> {
            type A = $scalar;
            fn get_q(&self) -> ArrayView2<Self::A> {
                self.q.view()
            }
            fn get_r(&self) -> ArrayView2<Self::A> {
                self.r.view()
            }

            fn new(arr: ArrayView2<Self::A>) -> Result<QRData<Self::A>> {
                <$scalar>::pivoted_qr(arr)
            }

            fn get_ind(&self) -> ArrayView1<usize> {
                self.ind.view()
            }

            fn get_q_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.q.view_mut()
            }
            fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.r.view_mut()
            }

            fn get_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.ind.view_mut()
            }
        }
    };
}

qr_data_impl!(f32);
qr_data_impl!(f64);
qr_data_impl!(c32);
qr_data_impl!(c64);

#[cfg(test)]
mod tests {

    use super::*;
    use crate::helpers::RelDiff;
    use crate::pivoted_qr::PivotedQR;
    use crate::random_matrix::RandomMatrix;
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

            let qr = <$scalar>::pivoted_qr(mat.view()).unwrap().compress(CompressionType::RANK(rank)).unwrap();

            // Compare with original matrix

            assert!(qr.q.len_of(Axis(1)) == rank);
            assert!(qr.r.len_of(Axis(0)) == rank);
            assert!(<$scalar>::rel_diff_fro(qr.to_mat().view(), mat.view()) < $tol);

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

            let qr = <$scalar>::pivoted_qr(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();

            // Compare with original matrix

            assert!(<$scalar>::rel_diff_fro(qr.to_mat().view(), mat.view()) < 5.0 * $tol);

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
