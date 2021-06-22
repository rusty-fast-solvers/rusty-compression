//! Random sampling of matrices

use crate::prelude::ScalarType;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Scalar;

pub trait MatVec {
    type A: Scalar;

    // Return the matrix vector product of an operator with a vector.
    fn matvec(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A>;
}

pub trait MatMat: MatVec {

    // Return the matrix-matrix product of an operator with a matrix.
    fn matmat(&self, vec: ArrayView2<Self::A>) -> Array2<Self::A>;
}

pub trait MatVecConjTrans: MatVec {

    // Return the matrix-vector product of the conjugate transpose of an operator
    // with a vector.
    fn matvec_conj_trans(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A>;
}

pub trait MatMatConjTrans: MatMat + MatVecConjTrans {

    // Return the matrix-matrix product of the conjugate transpose of an operator
    // with a matrix.
    fn matmat_conj_trans(&self, vec: ArrayView2<Self::A>) -> Array2<Self::A>;
}

pub trait SampleRangeByRank: MatMat {

    // Randomly sample the range of an operator.
    // Return an approximate orthogonal basis of the dominant range.
    // # Arguments
    // * `k`: The target rank of the basis for the range
    // * `p`: Oversampling parameter. `p` should be chosen small. A typical size
    //      is p=5.
    fn sample_range_by_rank(&self, k: usize, p: usize) -> Array2<Self::A>;
}

pub trait SampleRangeByRankPowerIteration: MatMat + MatMatConjTrans {

    // Randomly sample the range of an operator.
    // Refine by `n` steps of power iteration.
    // Return an approximate orthogonal basis of the dominant range
    // with `rank` basis vectors.
    fn sample_range_by_rank_power_iteration(&self, k: usize, p: usize) -> Array2<Self::A>;


}
