//! This module contains the traits definitions and some standard implementations
//! for the compression package.

use ndarray::ArrayView2;
use ndarray_linalg::types::{Scalar, Lapack};

pub trait ArrayProvider<T>
where
    T: Scalar + Lapack,
{
    fn array_view(&self) -> ArrayView2<T>;
}

impl <T: Scalar + Lapack> ArrayProvider<T> for ArrayView2<'_, T> {

    fn array_view(&self) -> ArrayView2<'_, T> {
        self.view()
    }


}