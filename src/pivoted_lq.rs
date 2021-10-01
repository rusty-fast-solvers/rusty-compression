//! This module implements LQ with pivoting by computing the pivoted QR decomposition for
//! the Hermitian Transpose of A.

use crate::prelude::LQContainer;
use crate::prelude::PivotedQR;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{ArrayBase, Data, Ix2};

pub trait PivotedLQ {
    type Q: ScalarType;

    fn pivoted_lq(&self) -> Result<LQContainer<Self::Q>>;
}

impl<A, S> PivotedLQ for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type Q = A;

    fn pivoted_lq(&self) -> Result<LQContainer<Self::Q>> {
        let mat_transpose = self.t().map(|item| item.conj());
        let qr_container = mat_transpose.pivoted_qr()?;

        Ok(LQContainer {
            l: qr_container.r.t().map(|item| item.conj()),
            q: qr_container.q.t().map(|item| item.conj()),
            ind: qr_container.ind,
        })
    }
}
 