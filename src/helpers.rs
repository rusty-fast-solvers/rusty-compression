//! This module collects the various traits definitions

use crate::prelude::ScalarType;
use ndarray::{ArrayBase, Data, Ix2};
use ndarray_linalg::Scalar;

pub trait RelDiff {
    type A: ScalarType;

    fn rel_diff(&self, other: Self) -> <Self::A as Scalar>::Real;
}

impl<A, S> RelDiff for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;
    fn rel_diff(&self, other: Self) -> <Self::A as Scalar>::Real {
        use ndarray_linalg::OperationNorm;

        let diff = self - &other;

        diff.opnorm_fro().unwrap() / other.opnorm_fro().unwrap()
    }
}
