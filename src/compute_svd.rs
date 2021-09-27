//! A simple trait to wrap SVD Computation.

use crate::svd::SVDData;
use ndarray::{ArrayBase, Data, Ix2};
use ndarray_linalg::SVDDCInto;
use rusty_base::types::{Result, Scalar};

pub(crate) trait ComputeSVD {
    type A: Scalar;

    fn compute_svd(&self) -> Result<SVDData<Self::A>>;
}

// impl<A, S> ComputeSVD for ArrayBase<S, Ix2>
// where
//     A: ScalarType,
//     S: Data<Elem = A>,
// {
//     type A = A;
//     fn compute_svd(&self) -> Result<SVDContainer<A>> {
//         use ndarray_linalg::UVTFlag;

//         let result = self.to_owned().svddc_into(UVTFlag::Some);

//         let (u, s, vt) = match result {
//             Ok((u, s, vt)) => (u.unwrap(), s, vt.unwrap()),
//             Err(_) => return Err("SVD Computation failed."),
//         };

//         Ok(SVDContainer { u, s, vt })
//     }
// }
