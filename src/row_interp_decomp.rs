//! Data structure for Row Interpolative Decomposition

use crate::helpers::Apply;
use crate::two_sided_interp_decomp::TwoSidedID;
use crate::qr::{QR, QRTraits};
use crate::col_interp_decomp::ColumnIDTraits;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Data, Ix1, Ix2,
};
use rusty_base::types::{c32, c64, Scalar, Result};

pub struct RowID<A: Scalar> {
    x: Array2<A>,
    r: Array2<A>,
    row_ind: Array1<usize>,
}

pub trait RowIDTraits {
    type A: Scalar;

    fn nrows(&self) -> usize {
        self.get_x().nrows()
    }

    fn ncols(&self) -> usize {
        self.get_r().ncols()
    }

    fn rank(&self) -> usize {
        self.get_r().nrows()
    }

    fn to_mat(&self) -> Array2<Self::A> {
        self.get_x().dot(&self.get_r())
    }

    fn get_x(&self) -> ArrayView2<Self::A>;
    fn get_r(&self) -> ArrayView2<Self::A>;
    fn get_row_ind(&self) -> ArrayView1<usize>;

    fn get_x_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_row_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    fn new(x: Array2<Self::A>, r: Array2<Self::A>, row_ind: Array1<usize>) -> Self;

    fn two_sided_id(&self) -> Result<TwoSidedID<Self::A>>;

}

macro_rules! impl_row_id {
    ($scalar:ty) => {
        impl RowIDTraits for RowID<$scalar> {
            type A = $scalar;
            fn get_x(&self) -> ArrayView2<Self::A> {
                self.x.view()
            }
            fn get_r(&self) -> ArrayView2<Self::A> {
                self.r.view()
            }

            fn get_row_ind(&self) -> ArrayView1<usize> {
                self.row_ind.view()
            }

            fn get_x_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.x.view_mut()
            }
            fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.r.view_mut()
            }
            fn get_row_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.row_ind.view_mut()
            }

            fn new(x: Array2<Self::A>, r: Array2<Self::A>, row_ind: Array1<usize>) -> Self {
                RowID::<$scalar> { x, r, row_ind }
            }

            fn two_sided_id(&self) -> Result<TwoSidedID<Self::A>> {
                let col_id = QR::<$scalar>::compute_from(self.r.view())?.column_id()?;
                Ok(TwoSidedID {
                    c: self.x.to_owned(),
                    x: col_id.get_c().into_owned(),
                    r: col_id.get_z().into_owned(),
                    row_ind: self.row_ind.to_owned(),
                    col_ind: col_id.get_col_ind().into_owned(),

                })
            }

        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix1>> for RowID<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array1<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix1>) -> Self::Output {
                self.x.dot(&self.r.dot(rhs))
            }
        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix2>> for RowID<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array2<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix2>) -> Self::Output {
                self.x.dot(&self.r.dot(rhs))
            }
        }
    };
}

impl_row_id!(f32);
impl_row_id!(f64);
impl_row_id!(c32);
impl_row_id!(c64);

#[cfg(test)]
mod tests {

    use crate::permutation::ApplyPermutationToMatrix;
    use crate::CompressionType;
    use crate::permutation::MatrixPermutationMode;
    use crate::qr::{LQTraits, LQ};
    use crate::row_interp_decomp::RowIDTraits;
    use crate::two_sided_interp_decomp::TwoSidedIDTraits;
    use crate::random_matrix::RandomMatrix;
    use crate::helpers::RelDiff;
    use rusty_base::types::Scalar;

    macro_rules! id_compression_tests {

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
            let two_sided_id = lq.row_id().unwrap().two_sided_id().unwrap();

            // Compare with original matrix

            assert!(<$scalar>::rel_diff_fro(two_sided_id.to_mat().view(), mat.view()) < 5.0 * $tol);

            // Now compare the individual columns to make sure that the id basis columns
            // agree with the corresponding matrix columns.

            let mat_permuted = mat.apply_permutation(two_sided_id.row_ind.view(), MatrixPermutationMode::ROW).
                apply_permutation(two_sided_id.col_ind.view(), MatrixPermutationMode::COL);

            // Assert that the x matrix in the two sided id is squared with correct dimension.

            assert!(two_sided_id.x.nrows() == two_sided_id.x.ncols());
            assert!(two_sided_id.x.nrows() == rank);

            // Now compare with the original matrix.

            for row_index in 0..rank {
                for col_index in 0..rank {
                    assert!((two_sided_id.x[[row_index, col_index]] - mat_permuted[[row_index, col_index]]).abs()
                            < 10.0 * $tol * mat_permuted[[row_index, col_index]].abs())
                }
            }
        }

            )*

        }
    }

    id_compression_tests! {
        test_two_sided_from_row_id_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_two_sided_from_row_id_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_two_sided_from_row_id_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_two_sided_from_row_id_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_two_sided_from_row_id_compression_by_tol_f32_thick: f32, (50, 100), 5E-4,
        test_two_sided_from_row_id_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_two_sided_from_row_id_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_two_sided_from_row_id_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }
}
