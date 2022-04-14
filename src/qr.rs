//! Data Structures and traits for QR Decompositions
//!
//! The pivoted QR Decomposition of a matrix $A\in\mathbb{C}^{m\times n}$ is
//! defined as $AP = QR$, where $P$ is a permutation matrix, $Q\in\mathbb{C}^{m\times k}$
//! is a matrix with orthogonal columns, satisfying $Q^HQ = I$, and $R\in\mathbb{C}^{k\times n}$
//! is an upper triangular matrix with diagonal elements $r_{ii}$ satisfying $|r_{11}|\geq |r_{22}|\geq \dots$.
//! Here $k=\min{m, n}$. The matrix $P$ is defined by an index vector `ind` in such a way that if ind\[j\] = k then
//! the jth column of $P$ is 1 at the position P\[k, j\] and 0 otherwise. In other words the matrix $P$ permutes the
//! $k$th column of $A$ to the $j$th column.
//!
//! This module also defines the LQ Decomposition defined as $PA = LQ$ with $L$ a lower triangular matrix. If
//! $A^H\tilde{P}=\tilde{Q}R$ is the QR decomposition as defined above, then $P = P^T$, $L=R^H$, $Q=\tilde{Q}^H$.
//!
//! Both, the QR and the LQ Decomposition of a matrix can be compressed further, either by specifying a rank or
//! by specifying a relative tolerance. Let $AP=QR$. We can compress the QR Decomposition by only keeping the first
//! $\ell$ columns ($\ell \leq k$) of $Q$ and correspondingly only keeping the first $\ell$ rows of $R$.
//! We can alternatively determine the $\ell$ by a tolerance tol such that only the first $\ell$ rows of $R$
//! are kept that satisfy $|r_{\ell, \ell}| / |r_{1, 1}| \geq tol$.

use crate::col_interp_decomp::{ColumnID, ColumnIDTraits};
use crate::permutation::{ApplyPermutationToMatrix, MatrixPermutationMode};
use crate::pivoted_qr::PivotedQR;
use crate::row_interp_decomp::{RowID, RowIDTraits};
use crate::CompressionType;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};
use ndarray_linalg::{Diag, SolveTriangular, UPLO};
use num::ToPrimitive;
use crate::types::{c32, c64, Result, Scalar};
use crate::types::{ConjMatMat, RustyCompressionError};

pub struct QR<A: Scalar> {
    /// The Q matrix from the QR Decomposition
    pub q: Array2<A>,
    /// The R matrix from the QR Decomposition
    pub r: Array2<A>,
    /// An index array. If ind\[j\] = k then the
    /// jth column of Q * R is identical to the
    /// kth column of the original matrix A.
    pub ind: Array1<usize>,
}

pub struct LQ<A: Scalar> {
    /// The Q matrix from the LQ Decomposition
    pub l: Array2<A>,
    /// The Q matrix from the LQ Decomposition
    pub q: Array2<A>,
    /// An index array. If ind\[j\] = k then the
    /// jth row of L * Q is identical to the
    /// kth row of the original matrix A.
    pub ind: Array1<usize>,
}

/// Traits for the LQ Decomposition
pub trait LQTraits {
    type A: Scalar;

    /// Number of rows
    fn nrows(&self) -> usize {
        self.get_l().nrows()
    }

    /// Number of columns
    fn ncols(&self) -> usize {
        self.get_q().ncols()
    }

    /// Rank of the LQ decomposition
    fn rank(&self) -> usize {
        self.get_q().nrows()
    }

    /// Convert the LQ decomposition to a matrix
    fn to_mat(&self) -> Array2<Self::A> {
        self.get_l()
            .apply_permutation(self.get_ind(), MatrixPermutationMode::ROWINV)
            .dot(&self.get_q())
    }

    /// Compress by giving a target rank
    fn compress_lq_rank(&self, mut max_rank: usize) -> Result<LQ<Self::A>> {
        let (l, q, ind) = (self.get_l(), self.get_q(), self.get_ind());

        if max_rank > q.nrows() {
            max_rank = q.nrows()
        }

        let q = q.slice(s![0..max_rank, ..]);
        let l = l.slice(s![.., 0..max_rank]);

        Ok(LQ {
            l: l.into_owned(),
            q: q.into_owned(),
            ind: ind.into_owned(),
        })
    }

    /// Compress by specifying a relative tolerance
    fn compress_lq_tolerance(&self, tol: f64) -> Result<LQ<Self::A>> {
        assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

        let pos = self
            .get_l()
            .diag()
            .iter()
            .position(|&item| ((item / self.get_l()[[0, 0]]).abs()).to_f64().unwrap() < tol);

        match pos {
            Some(index) => self.compress_lq_rank(index),
            None => Err(RustyCompressionError::CompressionError),
        }
    }

    /// Compress the LQ Decomposition by rank or tolerance
    fn compress(&self, compression_type: CompressionType) -> Result<LQ<Self::A>> {
        match compression_type {
            CompressionType::ADAPTIVE(tol) => self.compress_lq_tolerance(tol),
            CompressionType::RANK(rank) => self.compress_lq_rank(rank),
        }
    }

    /// Return the Q matrix
    fn get_q(&self) -> ArrayView2<Self::A>;

    /// Return the L matrix
    fn get_l(&self) -> ArrayView2<Self::A>;

    /// Return the index vector
    fn get_ind(&self) -> ArrayView1<usize>;

    fn get_q_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_l_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    /// Compute the LQ decomposition from a given array
    fn compute_from(arr: ArrayView2<Self::A>) -> Result<LQ<Self::A>>;

    /// Compute a row interpolative decomposition from the LQ decomposition
    fn row_id(&self) -> Result<RowID<Self::A>>;
}

pub trait QRTraits {
    type A: Scalar;

    /// Number of rows
    fn nrows(&self) -> usize {
        self.get_q().nrows()
    }

    /// Number of columns
    fn ncols(&self) -> usize {
        self.get_r().ncols()
    }

    /// Rank of the QR Decomposition
    fn rank(&self) -> usize {
        self.get_q().ncols()
    }

    /// Convert the QR decomposition to a matrix
    fn to_mat(&self) -> Array2<Self::A> {
        self.get_q().dot(
            &self
                .get_r()
                .apply_permutation(self.get_ind(), MatrixPermutationMode::COLINV),
        )
    }

    /// Compress by giving a target rank
    fn compress_qr_rank(&self, mut max_rank: usize) -> Result<QR<Self::A>> {
        let (q, r, ind) = (self.get_q(), self.get_r(), self.get_ind());

        if max_rank > q.ncols() {
            max_rank = q.ncols()
        }

        let q = q.slice(s![.., 0..max_rank]);
        let r = r.slice(s![0..max_rank, ..]);

        Ok(QR {
            q: q.into_owned(),
            r: r.into_owned(),
            ind: ind.into_owned(),
        })
    }

    /// Compress by specifying a relative tolerance
    fn compress_qr_tolerance(&self, tol: f64) -> Result<QR<Self::A>> {
        assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

        let pos = self
            .get_r()
            .diag()
            .iter()
            .position(|&item| ((item / self.get_r()[[0, 0]]).abs()).to_f64().unwrap() < tol);

        match pos {
            Some(index) => self.compress_qr_rank(index),
            None => Err(RustyCompressionError::CompressionError),
        }
    }

    /// Compress the QR decomposition by rank or tolerance
    fn compress(&self, compression_type: CompressionType) -> Result<QR<Self::A>> {
        match compression_type {
            CompressionType::ADAPTIVE(tol) => self.compress_qr_tolerance(tol),
            CompressionType::RANK(rank) => self.compress_qr_rank(rank),
        }
    }

    /// Compute a column interpolative decomposition from the QR decomposition
    fn column_id(&self) -> Result<ColumnID<Self::A>>;

    /// Compute the QR decomposition from a given array
    fn compute_from(arr: ArrayView2<Self::A>) -> Result<QR<Self::A>>;

    /// Compute a QR decomposition from a range estimate
    /// # Arguments
    /// * `range`: A matrix with orthogonal columns that approximates the range
    ///            of the operator.
    /// * `op`: The underlying operator.
    fn compute_from_range_estimate<Op: ConjMatMat<A = Self::A>>(
        range: ArrayView2<Self::A>,
        op: &Op,
    ) -> Result<QR<Self::A>>;

    /// Return the Q matrix
    fn get_q(&self) -> ArrayView2<Self::A>;

    /// Return the R matrix
    fn get_r(&self) -> ArrayView2<Self::A>;

    /// Return the index vector
    fn get_ind(&self) -> ArrayView1<usize>;

    fn get_q_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_ind_mut(&mut self) -> ArrayViewMut1<usize>;
}

macro_rules! qr_data_impl {
    ($scalar:ty) => {
        impl QRTraits for QR<$scalar> {
            type A = $scalar;
            fn get_q(&self) -> ArrayView2<Self::A> {
                self.q.view()
            }
            fn get_r(&self) -> ArrayView2<Self::A> {
                self.r.view()
            }

            fn compute_from(arr: ArrayView2<Self::A>) -> Result<QR<Self::A>> {
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

            fn column_id(&self) -> Result<ColumnID<Self::A>> {
                let rank = self.rank();
                let nrcols = self.ncols();

                if rank == nrcols {
                    // Matrix not rank deficient.
                    Ok(ColumnID::<Self::A>::new(
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

                    Ok(ColumnID::<Self::A>::new(
                        c,
                        z.apply_permutation(self.get_ind(), MatrixPermutationMode::COLINV),
                        self.get_ind().into_owned(),
                    ))
                }
            }

            fn compute_from_range_estimate<Op: ConjMatMat<A = Self::A>>(
                range: ArrayView2<Self::A>,
                op: &Op,
            ) -> Result<QR<Self::A>> {
                let b = op.conj_matmat(range.t().map(|item| item.conj()).view());
                let qr = QR::<$scalar>::compute_from(b.view())?;

                Ok(QR {
                    q: b.dot(&qr.get_q()),
                    r: qr.get_r().into_owned(),
                    ind: qr.get_ind().into_owned(),
                })
            }
        }
    };
}

macro_rules! lq_data_impl {
    ($scalar:ty) => {
        impl LQTraits for LQ<$scalar> {
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

            fn compute_from(arr: ArrayView2<Self::A>) -> Result<LQ<Self::A>> {
                let arr_trans = arr.t().map(|val| val.conj());
                let qr = QR::<$scalar>::compute_from(arr_trans.view())?;
                Ok(LQ {
                    l: qr.r.t().map(|item| item.conj()),
                    q: qr.q.t().map(|item| item.conj()),
                    ind: qr.ind,
                })
            }
            fn row_id(&self) -> Result<RowID<Self::A>> {
                let rank = self.rank();
                let nrows = self.nrows();

                if rank == nrows {
                    // Matrix not rank deficient.
                    Ok(RowID::<Self::A>::new(
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

                    Ok(RowID::<Self::A>::new(
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
    use crate::types::RelDiff;
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

            let qr = QR::<$scalar>::compute_from(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
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

            let lq = LQ::<$scalar>::compute_from(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
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
