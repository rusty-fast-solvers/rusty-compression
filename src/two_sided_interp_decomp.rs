//! Implementation of the interpolative decomposition.

use crate::pivoted_lq::PivotedLQ;
use crate::prelude::QRContainer;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2, OwnedRepr};

pub struct TwoSidedIDResult<A: ScalarType> {
    pub x: Array2<A>,
    pub r: Array2<A>,
    pub c: Array2<A>,
    pub row_ind: Array1<usize>,
    pub col_ind: Array1<usize>,
}

impl<A: ScalarType> TwoSidedIDResult<A> {
    pub fn nrows(&self) -> usize {
        self.c.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.r.ncols()
    }

    pub fn rank(&self) -> usize {
        self.x.nrows()
    }

    pub fn to_mat(&self) -> Array2<A> {
        self.c.dot(&self.x.dot(&self.r))
    }

    pub fn apply_matrix<S: Data<Elem = A>>(
        &self,
        other: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<A>, Ix2> {
        self.c.dot(&self.x.dot(&self.r.dot(other)))
    }

    pub fn apply_vector<S: Data<Elem = A>>(
        &self,
        other: &ArrayBase<S, Ix1>,
    ) -> ArrayBase<OwnedRepr<A>, Ix1> {
        self.c.dot(&self.x.dot(&self.r.dot(other)))
    }

    //}
}

impl<A: ScalarType> QRContainer<A> {
    pub fn two_sided_id(&self) -> Result<TwoSidedIDResult<A>> {
        let col_id = self.column_id()?;
        let row_id = col_id.c.pivoted_lq()?.row_id()?;

        Ok(TwoSidedIDResult {
            c: row_id.x,
            x: row_id.r,
            r: col_id.z,
            row_ind: row_id.row_ind,
            col_ind: col_id.col_ind,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::prelude::ApplyPermutationToMatrix;
    use crate::prelude::CompressionType;
    use crate::prelude::MatrixPermutationMode;
    use crate::prelude::PivotedQR;
    use crate::prelude::RandomMatrix;
    use crate::prelude::RelDiff;
    use ndarray_linalg::Scalar;

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

            let qr = mat.pivoted_qr().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
            let rank = qr.rank();
            let two_sided_id = qr.two_sided_id().unwrap();

            // Compare with original matrix

            assert!(two_sided_id.to_mat().rel_diff(&mat) < 5.0 * $tol);

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
        test_id_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_id_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_id_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_id_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_id_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        test_id_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_id_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_id_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }
}
