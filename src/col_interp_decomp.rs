//! Data structures for Column Interpolative Decomposition.
//! 
//! A column interpolative decomposition of a matrix $A\in\mathbb{C}^{m\times n}$ is
//! defined as
//! $$
//! A\approx CZ
//! $$
//! with $C\in\mathbb{C}^{m\times k}$ being a matrix whose columns form a subset of the columns 
//! of $A$, and $Z\in\mathbb{R}^{k\times m}$. The columns of $C$ are obtained from the corresponding columns of
//! $A$ via an index vector col_ind. If col_ind\[i\] = j then the ith column of $C$ is identical to the jth column
//! of $A$.

use crate::types::Apply;
use crate::two_sided_interp_decomp::TwoSidedID;
use crate::qr::{LQ, LQTraits};
use crate::row_interp_decomp::RowIDTraits;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Data, Ix1, Ix2,
};
use crate::types::{c32, c64, Scalar, Result};

/// Store a Column Interpolative Decomposition
pub struct ColumnID<A: Scalar> {
    /// The C matrix of the column interpolative decomposition
    c: Array2<A>,
    /// The Z matrix of the column interpolative decomposition
    z: Array2<A>,
    /// An index vector. If col_ind\[i\] = j then the ith column of
    /// C is identical to the jth column of A.
    col_ind: Array1<usize>,
}

/// Traits defining a column interpolative decomposition
/// 
// A column interpolative decomposition of a matrix $A\in\mathbb{C}^{m\times n}$ is
// defined as
// $$
// A\approx CZ
// $$
// with $C\in\mathbb{C}^{m\times k}$ being a matrix whose columns form a subset of the columns 
// of $A$, and $Z\in\mathbb{R}^{k\times m}$. The columns of $C$ are obtained from the corresponding columns of
// $A$ via an index vector col_ind. If col_ind\[i\] = j then the ith column of $C$ is identical to the jth column
// of $A$.
pub trait ColumnIDTraits {
    type A: Scalar;

    /// Number of rows of the underlying operator
    fn nrows(&self) -> usize {
        self.get_c().nrows()
    }

    /// Number of columns of the underlying operator
    fn ncols(&self) -> usize {
        self.get_z().ncols()
    }

    /// Rank of the column interpolative decomposition
    fn rank(&self) -> usize {
        self.get_c().ncols()
    }

    /// Convert to a matrix
    fn to_mat(&self) -> Array2<Self::A> {
        self.get_c().dot(&self.get_z())
    }

    /// Return the C matrix
    fn get_c(&self) -> ArrayView2<Self::A>;

    /// Return the Z matrix
    fn get_z(&self) -> ArrayView2<Self::A>;

    /// Return the index vector
    fn get_col_ind(&self) -> ArrayView1<usize>;

    fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_z_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    /// Return a column interpolative decomposition from given component matrices $C$ and
    /// $Z$ and index array col_ind
    fn new(c: Array2<Self::A>, z: Array2<Self::A>, col_ind: Array1<usize>) -> Self;

    /// Convert the column interpolative decomposition into a two sided interpolative decomposition
    fn two_sided_id(&self) -> Result<TwoSidedID<Self::A>>;
}

macro_rules! impl_col_id {
    ($scalar:ty) => {
        impl ColumnIDTraits for ColumnID<$scalar> {
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
                ColumnID::<$scalar> { c, z, col_ind }
            }
            fn two_sided_id(&self) -> Result<TwoSidedID<Self::A>> {
                let row_id = LQ::<$scalar>::compute_from(self.c.view())?.row_id()?;
                Ok(TwoSidedID {
                    c: row_id.get_x().into_owned(),
                    x: row_id.get_r().into_owned(),
                    r: self.get_z().into_owned(),
                    row_ind: row_id.get_row_ind().into_owned(),
                    col_ind: self.col_ind.to_owned(),

                })




            }

        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix1>> for ColumnID<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array1<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix1>) -> Self::Output {
                self.c.dot(&self.z.dot(rhs))
            }
        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix2>> for ColumnID<$scalar>
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
    use crate::col_interp_decomp::ColumnIDTraits;
    use crate::two_sided_interp_decomp::TwoSidedIDTraits;
    use crate::random_matrix::RandomMatrix;
    use crate::types::RelDiff;
    use crate::types::Scalar;

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
