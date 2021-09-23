//! Implementation of the interpolative decomposition.

use crate::prelude::ApplyPermutationToMatrix;
use crate::prelude::LQContainer;
use crate::prelude::MatrixPermutationMode;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, OwnedRepr};
use ndarray_linalg::{Diag, SolveTriangular, UPLO};

pub struct RowIDResult<A: ScalarType> {
    pub x: Array2<A>,
    pub r: Array2<A>,
    pub row_ind: Array1<usize>,
}

impl<A: ScalarType> RowIDResult<A> {
    pub fn nrows(&self) -> usize {
        self.x.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.r.ncols()
    }

    pub fn rank(&self) -> usize {
        self.r.nrows()
    }

    pub fn to_mat(&self) -> Array2<A> {
        self.x.dot(&self.r)
    }

    pub fn apply_matrix<S: Data<Elem = A>>(
        &self,
        other: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<A>, Ix2> {
        self.x.dot(&self.r.dot(other))
    }

    pub fn apply_vector<S: Data<Elem = A>>(
        &self,
        other: &ArrayBase<S, Ix1>,
    ) -> ArrayBase<OwnedRepr<A>, Ix1> {
        self.x.dot(&self.r.dot(other))
    }

    //}
}

impl<A: ScalarType> LQContainer<A> {
    pub fn row_id(&self) -> Result<RowIDResult<A>> {
        let rank = self.rank();
        let nrows = self.nrows();

        if rank == nrows {
            // Matrix not rank deficient.
            Ok(RowIDResult::<A> {
                x: Array2::<A>::eye(rank)
                    .apply_permutation(self.ind.view(), MatrixPermutationMode::ROWINV),
                r: self.l.dot(&self.q),
                row_ind: self.ind.clone(),
            })
        } else {
            // Matrix is rank deficient.

            let mut x = Array2::<A>::zeros((self.nrows(), rank));
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

            Ok(RowIDResult::<A> {
                x: x.apply_permutation(self.ind.view(), MatrixPermutationMode::ROWINV),
                r,
                row_ind: self.ind.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::prelude::ApplyPermutationToMatrix;
    use crate::prelude::CompressionType;
    use crate::prelude::MatrixPermutationMode;
    use crate::prelude::PivotedLQ;
    use crate::prelude::RandomMatrix;
    use crate::prelude::RelDiff;
    use ndarray::Axis;

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

            let lq = mat.pivoted_lq().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
            let rank = lq.rank();
            let row_id = lq.row_id().unwrap();

            // Compare with original matrix

            assert!(row_id.to_mat().rel_diff(&mat) < 5.0 * $tol);

            // Now compare the individual columns to make sure that the id basis columns
            // agree with the corresponding matrix columns.

            let mat_permuted = mat.apply_permutation(row_id.row_ind.view(), MatrixPermutationMode::ROW);

            for index in 0..rank {
                assert!(mat_permuted.index_axis(Axis(0), index).rel_diff(&row_id.r.index_axis(Axis(0), index)) < $tol);

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
