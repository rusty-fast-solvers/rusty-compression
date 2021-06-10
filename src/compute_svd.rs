//! A simple trait to wrap SVD Computation.

use crate::Result;
use crate::prelude::SVDContainer;
use crate::prelude::ScalarType;
use ndarray::{ArrayBase, Data, Ix2};
use ndarray_linalg::SVDDCInto;

pub trait ComputeSVD {
    type A: ScalarType;

    fn compute_svd(&self) -> Result<SVDContainer<Self::A>>;
}


impl<A, S> ComputeSVD for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;
    fn compute_svd(&self) -> Result<SVDContainer<A>> {
        use ndarray_linalg::UVTFlag;

        let result = self.to_owned().svddc_into(UVTFlag::Some);

        let (u, s, vt) = match result {
            Ok((u, s, vt)) => (u.unwrap(), s, vt.unwrap()),
            Err(_) => return Err("SVD Computation failed."),
        };

        Ok(SVDContainer { u, s, vt })
    }
}
