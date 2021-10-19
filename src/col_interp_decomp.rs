//! Data structure for Column Interpolative Decomposition

use crate::helpers::Apply;
use crate::two_sided_interp_decomp::TwoSidedIDData;
use crate::qr::{LQ, LQTraits};
use crate::row_interp_decomp::RowID;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Data, Ix1, Ix2,
};
use rusty_base::types::{c32, c64, Scalar, Result};

pub struct ColumnIDData<A: Scalar> {
    c: Array2<A>,
    z: Array2<A>,
    col_ind: Array1<usize>,
}

pub trait ColumnID {
    type A: Scalar;

    fn nrows(&self) -> usize {
        self.get_c().nrows()
    }

    fn ncols(&self) -> usize {
        self.get_z().ncols()
    }

    fn rank(&self) -> usize {
        self.get_c().ncols()
    }

    fn to_mat(&self) -> Array2<Self::A> {
        self.get_c().dot(&self.get_z())
    }

    fn get_c(&self) -> ArrayView2<Self::A>;
    fn get_z(&self) -> ArrayView2<Self::A>;
    fn get_col_ind(&self) -> ArrayView1<usize>;

    fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_z_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    fn new(c: Array2<Self::A>, z: Array2<Self::A>, col_ind: Array1<usize>) -> Self;
    fn two_sided_id(&self) -> Result<TwoSidedIDData<Self::A>>;
}

macro_rules! impl_col_id {
    ($scalar:ty) => {
        impl ColumnID for ColumnIDData<$scalar> {
            type A = $scalar;
            fn get_c(&self) -> ArrayView2<Self::A> {
                self.c.view()
            }
            fn get_z(&self) -> ArrayView2<Self::A> {
                self.z.view()
            }

            fn get_col_ind(&self) -> ArrayView1<usize> {
                self.col_ind.view()
            }

            fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.c.view_mut()
            }
            fn get_z_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.z.view_mut()
            }
            fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.col_ind.view_mut()
            }

            fn new(c: Array2<Self::A>, z: Array2<Self::A>, col_ind: Array1<usize>) -> Self {
                ColumnIDData::<$scalar> { c, z, col_ind }
            }
            fn two_sided_id(&self) -> Result<TwoSidedIDData<Self::A>> {
                let row_id = LQ::<$scalar>::compute_from(self.c.view())?.row_id()?;
                Ok(TwoSidedIDData {
                    c: row_id.get_x().into_owned(),
                    x: row_id.get_r().into_owned(),
                    r: self.get_z().into_owned(),
                    row_ind: row_id.get_row_ind().into_owned(),
                    col_ind: self.col_ind.to_owned(),

                })




            }

        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix1>> for ColumnIDData<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array1<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix1>) -> Self::Output {
                self.c.dot(&self.z.dot(rhs))
            }
        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix2>> for ColumnIDData<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array2<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix2>) -> Self::Output {
                self.c.dot(&self.z.dot(rhs))
            }
        }
    };
}

impl_col_id!(f32);
impl_col_id!(f64);
impl_col_id!(c32);
impl_col_id!(c64);

#[cfg(test)]
mod tests {

    use crate::permutation::ApplyPermutationToMatrix;
    use crate::CompressionType;
    use crate::permutation::MatrixPermutationMode;
    use crate::qr::{QRTraits, QR};
    use crate::col_interp_decomp::ColumnID;
    use crate::two_sided_interp_decomp::TwoSidedID;
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

            let qr = QR::<$scalar>::compute_from(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
            let rank = qr.rank();
            let two_sided_id = qr.column_id().unwrap().two_sided_id().unwrap();

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
                    let tmp = (two_sided_id.x[[row_index, col_index]] - mat_permuted[[row_index, col_index]]).abs() / mat_permuted[[row_index, col_index]].abs();
                    println!("Rel Error {}", tmp);
                    //if tmp >= 5.0 * $tol {
                        //println!(" Rel Error {}", tmp);
                    //}

                    assert!((two_sided_id.x[[row_index, col_index]] - mat_permuted[[row_index, col_index]]).abs()
                            < 10.0 * $tol * mat_permuted[[row_index, col_index]].abs())
                }
            }
        }

            )*

        }
    }

    id_compression_tests! {
        test_two_sided_from_col_id_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_two_sided_from_col_id_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }
}
