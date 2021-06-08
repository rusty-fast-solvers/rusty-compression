//! A container for QR Decompositions.

use crate::prelude::ScalarType;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Lapack, Scalar};

pub struct QRContainer<T: Scalar + Lapack> {
    /// The Q matrix from the QR Decomposition
    pub q: Array2<T>,
    /// The R matrix from the QR Decomposition
    pub r: Array2<T>,
    /// An index array. If ind[j] = k then the
    /// jth column of Q * R is identical to the
    /// kth column of the original matrix A.
    pub ind: Array1<usize>,
}

