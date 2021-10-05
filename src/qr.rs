//! A container for QR Decompositions.

use crate::col_interp_decomp::{ColumnID, ColumnIDData};
use crate::permutation::{ApplyPermutationToMatrix, MatrixPermutationMode};
use crate::pivoted_qr::PivotedQR;
use crate::row_interp_decomp::{RowID, RowIDData};
use crate::CompressionType;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};
use ndarray_linalg::{Diag, SolveTriangular, UPLO};
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

pub struct LQData<A: Scalar> {
    /// The Q matrix from the LQ Decomposition
    pub l: Array2<A>,
    /// The Q matrix from the LQ Decomposition
    pub q: Array2<A>,
    /// An index array. If ind[j] = k then the
    /// jth row of L * Q is identical to the
    /// kth row of the original matrix A.
    pub ind: Array1<usize>,
}

pub trait LQ {
    type A: Scalar;

    fn nrows(&self) -> usize {
        self.get_l().nrows()
    }
    fn ncols(&self) -> usize {
        self.get_q().ncols()
    }
    fn rank(&self) -> usize {
        self.get_q().nrows()
    }
    fn to_mat(&self) -> Array2<Self::A> {
        self.get_l()
            .apply_permutation(self.get_ind(), MatrixPermutationMode::ROWINV)
            .dot(&self.get_q())
    }

    fn compress_lq_rank(&self, mut max_rank: usize) -> Result<LQData<Self::A>> {
        let (l, q, ind) = (self.get_l(), self.get_q(), self.get_ind());

        if max_rank > q.nrows() {
            max_rank = q.nrows()
        }

        let q = q.slice(s![0..max_rank, ..]);
        let l = l.slice(s![.., 0..max_rank]);

        Ok(LQData {
            l: l.into_owned(),
            q: q.into_owned(),
            ind: ind.into_owned(),
        })
    }

    fn compress_lq_tolerance(&self, tol: f64) -> Result<LQData<Self::A>> {
        assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

        let pos = self
            .get_l()
            .diag()
            .iter()
            .position(|&item| ((item / self.get_l()[[0, 0]]).abs()).to_f64().unwrap() < tol);

        match pos {
            Some(index) => self.compress_lq_rank(index),
            None => Err("Could not compress operator to desired tolerance."),
        }
    }

    fn compress(&self, compression_type: CompressionType) -> Result<LQData<Self::A>> {
        match compression_type {
            CompressionType::ADAPTIVE(tol) => self.compress_lq_tolerance(tol),
            CompressionType::RANK(rank) => self.compress_lq_rank(rank),
        }
    }

    fn get_q(&self) -> ArrayView2<Self::A>;
    fn get_l(&self) -> ArrayView2<Self::A>;
    fn get_ind(&self) -> ArrayView1<usize>;

    fn get_q_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_l_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    fn new(arr: ArrayView2<Self::A>) -> Result<LQData<Self::A>>;
    fn row_id(&self) -> Result<RowIDData<Self::A>>;
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

        let q = q.slice(s![.., 0..max_rank]);
        let r = r.slice(s![0..max_rank, ..]);

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

    fn column_id(&self) -> Result<ColumnIDData<Self::A>>;

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

            fn column_id(&self) -> Result<ColumnIDData<Self::A>> {
                let rank = self.rank();
                let nrcols = self.ncols();

                if rank == nrcols {
                    // Matrix not rank deficient.
                    Ok(ColumnIDData::<Self::A>::new(
                        self.get_q().dot(&self.get_r()),
                        Array2::<Self::A>::eye(rank)
                            .apply_permutation(self.get_ind(), MatrixPermutationMode::COLINV),
                        self.get_ind().into_owned(),
                    ))
                } else {
                    // Matrix is rank deficient.

                    let mut z = Array2::<Self::A>::zeros((rank, self.get_r().ncols()));
                    z.slice_mut(s![.., 0..rank]).diag_mut().fill(num::one());
                    let first_part = self.get_r().slice(s![.., 0..rank]).to_owned();
                    let c = self.get_q().dot(&first_part);

                    for (index, col) in self
                        .get_r()
                        .slice(s![.., rank..nrcols])
                        .axis_iter(Axis(1))
                        .enumerate()
                    {
                        z.index_axis_mut(Axis(1), rank + index).assign(
                            &first_part
                                .solve_triangular(UPLO::Upper, Diag::NonUnit, &col.to_owned())
                                .unwrap(),
                        );
                    }

                    Ok(ColumnIDData::<Self::A>::new(
                        c,
                        z.apply_permutation(self.get_ind(), MatrixPermutationMode::COLINV),
                        self.get_ind().into_owned(),
                    ))
                }
            }
        }
    };
}

macro_rules! lq_data_impl {
    ($scalar:ty) => {
        impl LQ for LQData<$scalar> {
            type A = $scalar;

            fn get_q(&self) -> ArrayView2<Self::A> {
                self.q.view()
            }

            fn get_l(&self) -> ArrayView2<Self::A> {
                self.l.view()
            }
            fn get_ind(&self) -> ArrayView1<usize> {
                self.ind.view()
            }

            fn get_q_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.q.view_mut()
            }
            fn get_l_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.l.view_mut()
            }
            fn get_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.ind.view_mut()
            }

            fn new(arr: ArrayView2<Self::A>) -> Result<LQData<Self::A>> {
                let arr_trans = arr.t().map(|val| val.conj());
                let qr = QRData::<$scalar>::new(arr_trans.view())?;
                Ok(LQData {
                    l: qr.r.t().map(|item| item.conj()),
                    q: qr.q.t().map(|item| item.conj()),
                    ind: qr.ind,
                })
            }
            fn row_id(&self) -> Result<RowIDData<Self::A>> {
                let rank = self.rank();
                let nrows = self.nrows();

                if rank == nrows {
                    // Matrix not rank deficient.
                    Ok(RowIDData::<Self::A>::new(
                        Array2::<Self::A>::eye(rank)
                            .apply_permutation(self.ind.view(), MatrixPermutationMode::ROWINV),
                        self.l.dot(&self.q),
                        self.ind.clone(),
                    ))
                } else {
                    // Matrix is rank deficient.

                    let mut x = Array2::<Self::A>::zeros((self.nrows(), rank));
                    x.slice_mut(s![0..rank, ..]).diag_mut().fill(num::one());
                    let first_part = self.l.slice(s![0..rank, ..]).to_owned();
                    let r = first_part.dot(&self.q);
                    let first_part_transposed = first_part.t().to_owned();

                    for (index, row) in self
                        .l
                        .slice(s![rank..nrows, ..])
                        .axis_iter(Axis(0))
                        .enumerate()
                    {
                        x.index_axis_mut(Axis(0), rank + index).assign(
                            &first_part_transposed
                                .solve_triangular(UPLO::Upper, Diag::NonUnit, &row.to_owned())
                                .unwrap(),
                        );
                    }

                    Ok(RowIDData::<Self::A>::new(
                        x.apply_permutation(self.ind.view(), MatrixPermutationMode::ROWINV),
                        r,
                        self.ind.clone(),
                    ))
                }
            }
        }
    };
}

qr_data_impl!(f32);
qr_data_impl!(f64);
qr_data_impl!(c32);
qr_data_impl!(c64);

lq_data_impl!(f32);
lq_data_impl!(f64);
lq_data_impl!(c32);
lq_data_impl!(c64);

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

    macro_rules! col_id_compression_tests {

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

            let qr = QRData::<$scalar>::new(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
            let rank = qr.rank();
            let column_id = qr.column_id().unwrap();

            // Compare with original matrix

            assert!(<$scalar>::rel_diff_fro(column_id.to_mat().view(), mat.view()) < 5.0 * $tol);

            // Now compare the individual columns to make sure that the id basis columns
            // agree with the corresponding matrix columns.

            let mat_permuted = mat.apply_permutation(column_id.get_col_ind(), MatrixPermutationMode::COL);

            for index in 0..rank {
                assert!(
                    <$scalar>::rel_diff_l2(mat_permuted.index_axis(Axis(1), index), column_id.get_c().index_axis(Axis(1), index)) < $tol);

            }

        }

            )*

        }
    }
    macro_rules! row_id_compression_tests {

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

            let lq = LQData::<$scalar>::new(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
            let rank = lq.rank();
            let row_id = lq.row_id().unwrap();

            // Compare with original matrix

            assert!(<$scalar>::rel_diff_fro(row_id.to_mat().view(), mat.view()) < 5.0 * $tol);

            // Now compare the individual columns to make sure that the id basis columns
            // agree with the corresponding matrix columns.

            let mat_permuted = mat.apply_permutation(row_id.get_row_ind(), MatrixPermutationMode::ROW);

            for index in 0..rank {
                assert!(<$scalar>::rel_diff_l2(mat_permuted.index_axis(Axis(0), index), row_id.get_r().index_axis(Axis(0), index)) < $tol);

            }

        }

            )*

        }
    }

    row_id_compression_tests! {
        test_row_id_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_row_id_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_row_id_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_row_id_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_row_id_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        test_row_id_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_row_id_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_row_id_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }

    col_id_compression_tests! {
        test_col_id_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_col_id_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_col_id_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_col_id_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_col_id_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        test_col_id_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_col_id_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_col_id_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }

    qr_compression_by_rank_tests! {
        test_qr_compression_by_rank_f32_thin: f32, (100, 50), 1E-4,

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
