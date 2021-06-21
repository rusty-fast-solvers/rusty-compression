//! Implementation of the interpolative decomposition.

use crate::prelude::ApplyPermutationToMatrix;
use crate::prelude::MatrixPermutationMode;
use crate::prelude::QRContainer;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2, Ix1, OwnedRepr};
use ndarray_linalg::{FactorizeInto, Solve};

pub struct ColumnIDResult<A: ScalarType> {
    pub c: Array2<A>,
    pub z: Array2<A>,
    pub col_ind: Array1<usize>,
}

pub struct RowIDResult<A: ScalarType> {
    pub x: Array2<A>,
    pub row_ind: Array1<usize>,
}

impl<A: ScalarType> ColumnIDResult<A> {
    pub fn nrows(&self) -> usize {
        self.c.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.z.ncols()
    }

    pub fn rank(&self) -> usize {
        self.c.ncols()
    }

    pub fn to_mat(&self) -> Array2<A> {
        self.c.dot(&self.z)
    }

    pub fn apply_matrix<S: Data<Elem = A>>(
        &self,
        other: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<A>, Ix2> {
        self.c.dot(&self.z.dot(other))
    }

    pub fn apply_vector<S: Data<Elem = A>>(
        &self,
        other: &ArrayBase<S, Ix1>,
        ) -> ArrayBase<OwnedRepr<A>, Ix1> {
        self.c.dot(&self.z.dot(other))
    }

    //}
}

pub struct TwoSidedIDResult<A: ScalarType> {
    pub x: Array2<A>,
    pub row_ind: Array1<usize>,
    pub col_ind: Array1<usize>,
}

impl<A: ScalarType> QRContainer<A> {
    pub fn column_id(&self) -> Result<ColumnIDResult<A>> {
        let rank = self.rank();
        let nrcols = self.ncols();

        if rank == nrcols {
            // Matrix not rank deficient.
            Ok(ColumnIDResult::<A> {
                c: self.q.dot(&self.r),
                z: Array2::<A>::eye(rank)
                    .apply_permutation(self.ind.view(), MatrixPermutationMode::COLINV),
                col_ind: self.ind.clone(),
            })
        } else {
            // Matrix is rank deficient.

            let mut z = Array2::<A>::zeros((rank, self.r.ncols()));
            z.slice_mut(s![.., 0..rank]).diag_mut().fill(num::one());
            let first_part = self.r.slice(s![.., 0..rank]).to_owned();
            let c = self.q.dot(&first_part);
            let factorized = first_part.factorize_into().unwrap();

            for (index, col) in self
                .r
                .slice(s![.., rank..nrcols])
                .axis_iter(Axis(1))
                .enumerate()
            {
                z.index_axis_mut(Axis(1), rank + index)
                    .assign(&factorized.solve(&col.to_owned()).unwrap());
            }

            Ok(ColumnIDResult::<A> {
                c,
                z: z.apply_permutation(self.ind.view(), MatrixPermutationMode::COLINV),
                col_ind: self.ind.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::prelude::ApplyPermutationToMatrix;
    use crate::prelude::CompressionType;
    use crate::prelude::MatrixPermutationMode;
    use crate::prelude::PivotedQR;
    use crate::prelude::Random;
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

            let qr = mat.pivoted_qr().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();
            let rank = qr.rank();
            let column_id = qr.column_id().unwrap();

            // Compare with original matrix

            assert!(column_id.to_mat().rel_diff(&mat) < 5.0 * $tol);

            // Now compare the individual columns to make sure that the id basis columns
            // agree with the corresponding matrix columns.

            let mat_permuted = mat.apply_permutation(column_id.col_ind.view(), MatrixPermutationMode::COL);

            for index in 0..rank {
                assert!(mat_permuted.index_axis(Axis(1), index).rel_diff(&column_id.c.index_axis(Axis(1), index)) < $tol);

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
