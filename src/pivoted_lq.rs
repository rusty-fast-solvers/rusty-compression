//! This module implements LQ with pivoting by computing the pivoted QR decomposition for
//! the Hermitian Transpose of A.

use ndarray::{ArrayBase, Data, Ix2};
use crate::prelude::LQContainer;
use crate::prelude::PivotedQR;
use crate::prelude::ScalarType;
use crate::Result;

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

        let mat_transpose =  self.t().map(|item| item.conj());
        let qr_container = mat_transpose.pivoted_qr()?;

        Ok(LQContainer {
            l: qr_container.r.t().map(|item| item.conj()),
            q: qr_container.q.t().map(|item| item.conj()),
            ind: qr_container.ind,
        })
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    macro_rules! pivoted_lq_tests {

    ($($name:ident: $scalar:ty, $dim:expr,)*) => {

        $(

        #[test]
        fn $name() {
            use crate::prelude::RandomMatrix;
            use ndarray_linalg::Norm;
            use ndarray_linalg::Scalar;

            let m = $dim.0;
            let n = $dim.1;

            let mut rng = rand::thread_rng();
            let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), 1.0, 1E-5, &mut rng);

            let lq_result = mat.pivoted_lq().unwrap();

            let prod = lq_result.l.dot(&lq_result.q);

            // Check orthogonality of Q x Q^T

            let qqt = lq_result.q.dot(&lq_result.q.t().map(|&item| item.conj()));

            for ((i, j), &val) in qqt.indexed_iter() {
                if i == j {
                    let rel_diff = (val - 1.0).abs();
                    assert!(rel_diff < 1E-6);
                } else {
                    assert!(val.abs() < 1E-6);
                }
            }

            // Check that the product is correct.

            for (row_index, row) in prod.axis_iter(ndarray::Axis(0)).enumerate() {
                let perm_index = lq_result.ind[row_index];
                let diff = row.to_owned() - mat.index_axis(ndarray::Axis(0), perm_index);
                let rel_diff = diff.norm_l2() / mat.index_axis(ndarray::Axis(0), perm_index).norm_l2();

                assert!(rel_diff < 1E-6);
            }
        }
                )*
            };
        }

    pivoted_lq_tests! {
        pivoted_lq_test_thin_f64: f64, (100, 50),
        pivoted_lq_test_thin_f32: f32, (100, 50),
        pivoted_lq_test_thin_c64: ndarray_linalg::c64, (100, 50),
        pivoted_lq_test_thin_c32: ndarray_linalg::c32, (100, 50),
        pivoted_lq_test_thick_f64: f64, (50, 100),
        pivoted_lq_test_thick_f32: f32, (50, 100),
        pivoted_lq_test_thick_c64: ndarray_linalg::c64, (50, 100),
        pivoted_lq_test_thick_c32: ndarray_linalg::c32, (50, 100),
    }
}
