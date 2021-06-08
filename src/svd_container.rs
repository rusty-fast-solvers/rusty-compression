//! Define an SVD container and conversion tools.

use crate::prelude::QRContainer;
use crate::prelude::PivotedQR;
use crate::prelude::HasPivotedQR;
use crate::Result;
use crate::prelude::ScalarType;
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray_linalg::{Lapack, Scalar};

pub struct SVDContainer<A: ScalarType> {
    /// The U matrix
    pub u: Array2<A>,
    /// The array of singular values
    pub s: Array1<A::Real>,
    /// The vt matrix
    pub vt: Array2<A>,
}

//impl<A: HasPivotedQR> SVDContainer<A> {
//    pub fn to_qr(self) -> Result<QRContainer<A>> {
//        let (u, s, mut vt) = (self.u, self.s, self.vt);
//
//        Zip::from(vt.axis_iter_mut(Axis(0)))
//            .and(s.view())
//            .apply(|mut row, &s_elem| row.map_inplace(|item| *item *= A::from_real(s_elem)));
//        
//        // Now compute the qr of vt
//        
//        let mut qr = vt.pivoted_qr();
//        qr.q =  
//    }
//}
