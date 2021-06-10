//! A container for QR Decompositions.

use crate::permutation::*;
use crate::prelude::ScalarType;
use ndarray::{Array1, Array2};

pub struct QRContainer<A: ScalarType> {
    /// The Q matrix from the QR Decomposition
    pub q: Array2<A>,
    /// The R matrix from the QR Decomposition
    pub r: Array2<A>,
    /// An index array. If ind[j] = k then the
    /// jth column of Q * R is identical to the
    /// kth column of the original matrix A.
    pub ind: Array1<usize>,
}

impl<A: ScalarType> QRContainer<A> {
    /// If A P = QR, multiply Q * R * P^T to obtain the matrix A.
    pub fn to_mat(&self) -> Array2<A> {
        self.q.dot(
            &self.r
                .apply_permutation(self.ind.view(), MatrixPermutationMode::COLTRANS),
        )
    }
}
