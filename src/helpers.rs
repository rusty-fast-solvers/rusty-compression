//! This module collects the various traits definitions

use ndarray::{ArrayView1, ArrayView2};
use ndarray_linalg::Norm;
use ndarray_linalg::OperationNorm;
use rusty_base::types::{c32, c64, Scalar};

pub trait Apply<A, Lhs>{
    type Output;

    fn dot(&self, lhs: &Lhs) -> Self::Output;

}


pub trait RApply<A, Lhs> {
    type Output;

    fn dot(&self, lhs: &Lhs) -> Self::Output;
    
}

pub trait RelDiff {
    type A: Scalar;

    /// Return the relative Frobenius norm difference of `first` and `second`.
    fn rel_diff_fro(
        first: ArrayView2<Self::A>,
        second: ArrayView2<Self::A>,
    ) -> <<Self as RelDiff>::A as Scalar>::Real;

    /// Return the relative l2 vector norm difference of `first` and `second`.
    fn rel_diff_l2(
        first: ArrayView1<Self::A>,
        second: ArrayView1<Self::A>,
    ) -> <<Self as RelDiff>::A as Scalar>::Real;
}

macro_rules! rel_diff_impl {
    ($scalar:ty) => {
        impl RelDiff for $scalar {
            type A = $scalar;
            fn rel_diff_fro(
                first: ArrayView2<Self::A>,
                second: ArrayView2<Self::A>,
            ) -> <<Self as RelDiff>::A as Scalar>::Real {
                let diff = first.to_owned() - &second;
                diff.opnorm_fro().unwrap() / second.opnorm_fro().unwrap()
            }

            fn rel_diff_l2(
                first: ArrayView1<Self::A>,
                second: ArrayView1<Self::A>,
            ) -> <<Self as RelDiff>::A as Scalar>::Real {
                let diff = first.to_owned() - &second;
                diff.norm_l2() / second.norm_l2()
            }
        }
    };
}

rel_diff_impl!(f32);
rel_diff_impl!(f64);
rel_diff_impl!(c32);
rel_diff_impl!(c64);
