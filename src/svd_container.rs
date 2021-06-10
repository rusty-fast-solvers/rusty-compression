//! Define an SVD container and conversion tools.

use crate::prelude::HasPivotedQR;
use crate::prelude::PivotedQR;
use crate::prelude::QRContainer;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{Array1, Array2, Axis, Zip};

pub struct SVDContainer<A: ScalarType> {
    /// The U matrix
    pub u: Array2<A>,
    /// The array of singular values
    pub s: Array1<A::Real>,
    /// The vt matrix
    pub vt: Array2<A>,
}

impl<A: HasPivotedQR> SVDContainer<A> {
    pub fn to_qr(self) -> Result<QRContainer<A>> {
        let (u, s, mut vt) = (self.u, self.s, self.vt);

        Zip::from(vt.axis_iter_mut(Axis(0)))
            .and(s.view())
            .apply(|mut row, &s_elem| row.map_inplace(|item| *item *= A::from_real(s_elem)));

        // Now compute the qr of vt

        let mut qr = vt.pivoted_qr()?;
        qr.q = u.dot(&qr.q);

        Ok(qr)
    }

}

#[cfg(test)]
mod tests {

    use crate::random::Random;
    use crate::compute_svd::ComputeSVD;
    use ndarray_linalg::OperationNorm;
    
    #[test]
    fn test_to_qr() {

        let m = 100;
        let n = 50;

        let mut rng = rand::thread_rng();
        let mat = f64::random_approximate_low_rank_matrix((m, n), 1.0, 1E-10, &mut rng);

        let svd = mat.compute_svd().unwrap();

        // Perform a QR decomposition and recover the original matrix.
        let actual = svd.to_qr().unwrap().to_mat();

        assert!((actual - mat.view()).opnorm_fro().unwrap() / mat.opnorm_fro().unwrap() < 1E-10);
        

    }

}
