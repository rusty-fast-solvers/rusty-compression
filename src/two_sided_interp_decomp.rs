//! Implementation of the two sided interpolative decomposition
//! 
//! The two sided interpolative decomposition of a matrix $A\in\mathbb{C}&{m\times n} is
//! defined as
//! $$
//! A \approx CXR,
//! $$
//! where $C\in\mathbb{C}^{m\times k}$, $X\in\mathbb{C}^{k\times k}$, and $R\in\mathbb{C}^{k\times n}$.
//! The matrix $X$ contains a subset of the entries of $A$, such that A\[row_ind\[:\], col_ind\[:\]\] = X, where
//! row_ind and col_ind are index vectors.

use crate::types::Apply;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Data, Ix1, Ix2,
};
use crate::types::{c32, c64, Scalar};

/// Store a two sided interpolative decomposition
pub struct TwoSidedID<A: Scalar> {
    /// The C matrix of the two sided interpolative decomposition
    pub c: Array2<A>,
    /// The X matrix of the two sided interpolative decomposition
    pub x: Array2<A>,
    /// The R matrix of the two sided interpolative decomposition
    pub r: Array2<A>,
    /// The row index vector
    pub row_ind: Array1<usize>,
    /// The column index vector
    pub col_ind: Array1<usize>,
}

/// Traits defining a two sided interpolative decomposition
/// 
/// defined as
/// The two sided interpolative decomposition of a matrix $A\in\mathbb{C}&{m\times n} is
/// $$
/// A \approx CXR,
/// $$
/// where $C\in\mathbb{C}^{m\times k}$, $X\in\mathbb{C}^{k\times k}$, and $R\in\mathbb{C}^{k\times n}$.
/// The matrix $X$ contains a subset of the entries of $A$, such that A\[row_ind\[:\], col_ind\[:\]\] = X, where
/// row_ind and col_ind are index vectors.

pub trait TwoSidedIDTraits {
    type A: Scalar;

    /// Number of rows of the underlying operator
    fn nrows(&self) -> usize {
        self.get_c().nrows()
    }

    /// Number of columns of the underlying operator
    fn ncols(&self) -> usize {
        self.get_r().ncols()
    }

    /// Rank of the two sided interpolative decomposition
    fn rank(&self) -> usize {
        self.get_c().ncols()
    }

    /// Convert to a matrix
    fn to_mat(&self) -> Array2<Self::A> {
        self.get_c().dot(&self.get_x().dot(&self.get_r()))
    }

    /// Return the C matrix
    fn get_c(&self) -> ArrayView2<Self::A>;

    /// Return the X matrix
    fn get_x(&self) -> ArrayView2<Self::A>;

    /// Return the R matrix
    fn get_r(&self) -> ArrayView2<Self::A>;

    /// Return the column index vector
    fn get_col_ind(&self) -> ArrayView1<usize>;

    /// Return the row index vector
    fn get_row_ind(&self) -> ArrayView1<usize>;

    fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_x_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize>;
    fn get_row_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    /// Return a two sided interpolative decomposition from the component matrices
    /// X, R, C, and the column and row index vectors
    fn new(
        x: Array2<Self::A>,
        r: Array2<Self::A>,
        c: Array2<Self::A>,
        col_ind: Array1<usize>,
        row_ind: Array1<usize>,
    ) -> Self;
}

macro_rules! impl_two_sided_id {
    ($scalar:ty) => {
        impl TwoSidedIDTraits for TwoSidedID<$scalar> {
            type A = $scalar;

            fn get_c(&self) -> ArrayView2<Self::A> {
                self.c.view()
            }

            fn get_x(&self) -> ArrayView2<Self::A> {
                self.x.view()
            }

            fn get_r(&self) -> ArrayView2<Self::A> {
                self.r.view()
            }
            fn get_col_ind(&self) -> ArrayView1<usize> {
                self.col_ind.view()
            }
            fn get_row_ind(&self) -> ArrayView1<usize> {
                self.row_ind.view()
            }

            fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.c.view_mut()
            }

            fn get_x_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.x.view_mut()
            }

            fn get_r_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.r.view_mut()
            }
            fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.col_ind.view_mut()
            }
            fn get_row_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.row_ind.view_mut()
            }
            fn new(
                x: Array2<Self::A>,
                r: Array2<Self::A>,
                c: Array2<Self::A>,
                col_ind: Array1<usize>,
                row_ind: Array1<usize>,
            ) -> Self {
                TwoSidedID::<$scalar> {
                    x,
                    r,
                    c,
                    col_ind,
                    row_ind,
                }
            }
        }
        impl<S> Apply<$scalar, ArrayBase<S, Ix1>> for TwoSidedID<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array1<$scalar>;
            fn dot(&self, rhs: &ArrayBase<S, Ix1>) -> Self::Output {
                self.c.dot(&self.x.dot(&self.r.dot(rhs)))
            }
        }
        impl<S> Apply<$scalar, ArrayBase<S, Ix2>> for TwoSidedID<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array2<$scalar>;
            fn dot(&self, rhs: &ArrayBase<S, Ix2>) -> Self::Output {
                self.c.dot(&self.x.dot(&self.r.dot(rhs)))
            }
        }
    };
}

impl_two_sided_id!(f32);
impl_two_sided_id!(f64);
impl_two_sided_id!(c32);
impl_two_sided_id!(c64);

// impl<A: ScalarType> TwoSidedIDResult<A> {
//     pub fn nrows(&self) -> usize {
//         self.c.nrows()
//     }

//     pub fn ncols(&self) -> usize {
//         self.r.ncols()
//     }

//     pub fn rank(&self) -> usize {
//         self.x.nrows()
//     }

//     pub fn to_mat(&self) -> Array2<A> {
//         self.c.dot(&self.x.dot(&self.r))
//     }

//     pub fn apply_matrix<S: Data<Elem = A>>(
//         &self,
//         other: &ArrayBase<S, Ix2>,
//     ) -> ArrayBase<OwnedRepr<A>, Ix2> {
//         self.c.dot(&self.x.dot(&self.r.dot(other)))
//     }

//     pub fn apply_vector<S: Data<Elem = A>>(
//         &self,
//         other: &ArrayBase<S, Ix1>,
//     ) -> ArrayBase<OwnedRepr<A>, Ix1> {
//         self.c.dot(&self.x.dot(&self.r.dot(other)))
//     }

//     //}
// }

// impl<A: ScalarType> QRContainer<A> {
//     pub fn two_sided_id(&self) -> Result<TwoSidedIDResult<A>> {
//         let col_id = self.column_id()?;
//         let row_id = col_id.c.pivoted_lq()?.row_id()?;

//         Ok(TwoSidedIDResult {
//             c: row_id.x,
//             x: row_id.r,
//             r: col_id.z,
//             row_ind: row_id.row_ind,
//             col_ind: col_id.col_ind,
//         })
//     }
// }

// #[cfg(test)]
// mod tests {

//     use crate::prelude::ApplyPermutationToMatrix;
//     use crate::prelude::CompressionType;
//     use crate::prelude::MatrixPermutationMode;
//     use crate::prelude::PivotedQR;
//     use crate::prelude::RandomMatrix;
//     use crate::prelude::RelDiff;
//     use ndarray_linalg::Scalar;

//     macro_rules! id_compression_tests {

//         ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

//             $(

//         #[test]
//         fn $name() {
//             let m = $dim.0;
//             let n = $dim.1;

//             let sigma_max = 1.0;
//             let sigma_min = 1E-10;
//             let mut rng = rand::thread_rng();
//             let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

//             let qr = mat.pivoted_qr().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
//             let rank = qr.rank();
//             let two_sided_id = qr.two_sided_id().unwrap();

//             // Compare with original matrix

//             assert!(two_sided_id.to_mat().rel_diff(&mat) < 5.0 * $tol);

//             // Now compare the individual columns to make sure that the id basis columns
//             // agree with the corresponding matrix columns.

//             let mat_permuted = mat.apply_permutation(two_sided_id.row_ind.view(), MatrixPermutationMode::ROW).
//                 apply_permutation(two_sided_id.col_ind.view(), MatrixPermutationMode::COL);

//             // Assert that the x matrix in the two sided id is squared with correct dimension.

//             assert!(two_sided_id.x.nrows() == two_sided_id.x.ncols());
//             assert!(two_sided_id.x.nrows() == rank);

//             // Now compare with the original matrix.

//             for row_index in 0..rank {
//                 for col_index in 0..rank {
//                     let tmp = (two_sided_id.x[[row_index, col_index]] - mat_permuted[[row_index, col_index]]).abs() / mat_permuted[[row_index, col_index]].abs();
//                     println!("Rel Error {}", tmp);
//                     //if tmp >= 5.0 * $tol {
//                         //println!(" Rel Error {}", tmp);
//                     //}

//                     assert!((two_sided_id.x[[row_index, col_index]] - mat_permuted[[row_index, col_index]]).abs()
//                             < 10.0 * $tol * mat_permuted[[row_index, col_index]].abs())
//                 }
//             }
//         }

//             )*

//         }
//     }

//     id_compression_tests! {
//         test_id_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
//         test_id_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
//         test_id_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
//         test_id_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
//         test_id_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
//         test_id_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
//         test_id_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
//         test_id_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
//     }
// }
