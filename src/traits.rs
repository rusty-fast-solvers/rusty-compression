//! This module contains the traits definitions and some standard implementations
//! for the compression package.

use ndarray::{Array2, ArrayView2};
use ndarray_linalg::types::{Lapack, Scalar};

pub trait ArrayProvider<T>
where
    T: Scalar + Lapack,
{
    fn array_view(&self) -> ArrayView2<T>;

    /// Compute the relative distance
    fn rel_diff<A: ArrayProvider<T>>(&self, other: A) -> T::Real {
        use ndarray_linalg::OperationNorm;

        let diff = self.array_view().to_owned() - other.array_view();

        diff.opnorm_fro().unwrap() / other.array_view().opnorm_fro().unwrap()
    }
}

impl<T: Scalar + Lapack> ArrayProvider<T> for ArrayView2<'_, T> {
    fn array_view(&self) -> ArrayView2<'_, T> {
        self.view()
    }
}

impl<T: Scalar + Lapack> ArrayProvider<T> for Array2<T> {
    fn array_view(&self) -> ArrayView2<'_, T> {
        self.view()
    }
}

impl<T: Scalar + Lapack> ArrayProvider<T> for &Array2<T> {
    fn array_view(&self) -> ArrayView2<'_, T> {
        self.view()
    }
}
