//! This module collects the various traits definitions

use crate::prelude::ScalarType;
use ndarray::{ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::OperationNorm;
use ndarray_linalg::{Norm, Scalar};

pub trait RelDiff {
    type A: ScalarType;

    fn rel_diff(&self, other: &Self) -> <Self::A as Scalar>::Real;
}

impl<A, S> RelDiff for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;
    fn rel_diff(&self, other: &Self) -> <Self::A as Scalar>::Real {
        let diff = self - &other;

        diff.opnorm_fro().unwrap() / other.opnorm_fro().unwrap()
    }
}

impl<A, S> RelDiff for ArrayBase<S, Ix1>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;
    fn rel_diff(&self, other: &Self) -> <Self::A as Scalar>::Real {
        let diff = self - &other;

        diff.norm_l2() / other.norm_l2()
    }
}
