//! A simple trait to wrap SVD Computation.

use crate::svd::SVDData;
use ndarray::ArrayView2;
use ndarray_linalg::{SVDDCInto, UVTFlag};
use rusty_base::types::{c32, c64, Result, Scalar};

pub(crate) trait ComputeSVD {
    type A: Scalar;

    fn compute_svd(arr: ArrayView2<Self::A>) -> Result<SVDData<Self::A>>;
}

macro_rules! compute_svd_impl {
    ($scalar:ty) => {
        impl ComputeSVD for $scalar {
            type A = $scalar;
            fn compute_svd(arr: ArrayView2<Self::A>) -> Result<SVDData<Self::A>> {
                let result = arr.to_owned().svddc_into(UVTFlag::Some);

                let (u, s, vt) = match result {
                    Ok((u, s, vt)) => (u.unwrap(), s, vt.unwrap()),
                    Err(_) => return Err("SVD Computation failed."),
                };

                Ok(SVDData { u, s, vt })
            }
        }
    };
}

compute_svd_impl!(f32);
compute_svd_impl!(f64);
compute_svd_impl!(c32);
compute_svd_impl!(c64);
