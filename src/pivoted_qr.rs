//! This module implements QR with pivoting by calling into the
//! corresponding Lapack routine. Pivoted QR is currently not
//! implemented in ndarray-linalg, making this module necessary.

use ndarray::{Array2, ArrayBase, Data, Ix2, ShapeBuilder};
use ndarray_linalg::{Lapack, Scalar};
use crate::prelude::QRContainer;
use crate::Result;

pub trait PivotedQR {
    type Q: Scalar + Lapack;

    fn pivoted_qr(&self) -> Result<QRContainer<Self::Q>>;
}

impl<A, S> PivotedQR for ArrayBase<S, Ix2>
where
    A: HasPivotedQR,
    S: Data<Elem = A>,
{
    type Q = A;

    fn pivoted_qr(&self) -> Result<QRContainer<Self::Q>> {
        let m = self.nrows();
        let n = self.ncols();

        let mut mat_fortran = Array2::<Self::Q>::zeros((m, n).f());
        mat_fortran.assign(&self);
        A::pivoted_qr_impl(mat_fortran)
    }
}

pub trait HasPivotedQR: imp::PivotedQRImpl {
}

impl<A : imp::PivotedQRImpl> HasPivotedQR for A {}


mod imp {

    use lax;
    use ndarray::{s, Array1, Array2};
    use ndarray_linalg::layout::AllocatedArray;
    use ndarray_linalg::{IntoTriangular, Lapack, MatrixLayout, Scalar};
    use num::traits::{ToPrimitive, Zero};
    use crate::Result;

    pub trait PivotedQRImpl
    where
        Self: Scalar + Lapack,
    {
        fn pivoted_qr_impl(mat: Array2<Self>)
            -> Result<super::QRContainer<Self>>;
        fn pivoted_qr_decomp(
            mat: &mut [Self],
            layout: MatrixLayout,
        ) -> std::result::Result<(Array1<Self>, Array1<usize>), i32>;
    }

    macro_rules! impl_qr_pivot {

    (@real, $scalar:ty, $qrf:path) => {
        impl_qr_pivot!(@body, $scalar, $qrf, );
    };
    (@complex, $scalar:ty, $qrf:path) => {
        impl_qr_pivot!(@body, $scalar, $qrf, rwork);
    };
    (@body, $scalar:ty, $qrf:path, $($rwork_ident:ident),*) => {
            impl PivotedQRImpl for $scalar {
                fn pivoted_qr_impl(
                    mut mat: Array2<Self>,
                ) -> Result<super::QRContainer<$scalar>> {
                    let m = mat.nrows();
                    let n = mat.ncols();
                    let k = m.min(n);

                    let layout = match mat.layout() {
                        Ok(layout) => layout,
                        Err(_) => return Err("Incompatible layout for pivoted QR"),
                    };

                    let result =
                        Self::pivoted_qr_decomp(mat.as_slice_memory_order_mut().unwrap(), layout);
                    let (mut tau, jpvt) = match result {
                        Ok(res) => res,
                        Err(_) => return Err("Lapack routien for pivoted QR failed."),
                    };

                    let mut r_mat = Array2::<$scalar>::zeros((k, n));
                    r_mat.assign(&mat.slice(s![0..k, ..]));
                    let r_mat = r_mat.into_triangular(ndarray_linalg::UPLO::Upper);

                    match lax::QR_::q(
                        layout,
                        mat.as_slice_memory_order_mut().unwrap(),
                        tau.as_slice_memory_order_mut().unwrap(),
                    ) {
                        Ok(_) => (),
                        Err(_) => return Err("Computation of Q matrix failed in pivoted QR."),
                    }

                    let mut q_mat = Array2::<$scalar>::zeros((m as usize, k as usize));
                    q_mat.assign(&mat.slice(s![.., 0..k]));

                    // Finally, return the QR decomposition.

                    Ok(super::QRContainer{q: q_mat, r:r_mat, ind: jpvt})
                }

                fn pivoted_qr_decomp(
                    mat: &mut [Self],
                    layout: MatrixLayout,
                ) -> std::result::Result<(Array1<Self>, Array1<usize>), i32> {
                    let m = layout.lda();
                    let n = layout.len();
                    let k = m.min(n);
                    let mut tau = ndarray::Array1::<$scalar>::zeros(k as usize);

                    let mut info = 0;
                    let mut work_size = [Self::zero()];
                    let mut jpvt = ndarray::Array1::<i32>::zeros(n as usize);

                    $(
                    let mut $rwork_ident = ndarray::Array1::<Self::Real>::zeros(2 * (n as usize));
                    )*

                    unsafe {
                        $qrf(
                            m,
                            n,
                            mat,
                            m,
                            jpvt.as_slice_memory_order_mut().unwrap(),
                            tau.as_slice_memory_order_mut().unwrap(),
                            &mut work_size,
                            -1,
                            $($rwork_ident.as_slice_memory_order_mut().unwrap(),)*
                            &mut info,
                        );
                    }

                    match info {
                        0 => (),
                        _ => return Err(info),
                    }

                    let lwork = work_size[0].to_usize().unwrap();
                    let mut work = ndarray::Array1::<$scalar>::zeros(lwork);
                    unsafe {
                        $qrf(
                            m,
                            n,
                            mat,
                            m,
                            jpvt.as_slice_memory_order_mut().unwrap(),
                            tau.as_slice_memory_order_mut().unwrap(),
                            work.as_slice_memory_order_mut().unwrap(),
                            lwork as i32,
                            $($rwork_ident.as_slice_memory_order_mut().unwrap(),)*
                            &mut info,
                        );
                    }

                    // JPVT for zero-based counting before we return

                    let jpvt = jpvt.map(|&item| (item - 1) as usize);

                    match info {
                        0 => Ok((tau, jpvt)),
                        _ => Err(info),
                    }
                }
            }
        };
    }
    impl_qr_pivot!(@real, f64, lapack::dgeqp3);
    impl_qr_pivot!(@real, f32, lapack::sgeqp3);
    impl_qr_pivot!(@complex, num::complex::Complex<f64>, lapack::zgeqp3);
    impl_qr_pivot!(@complex, num::complex::Complex<f32>, lapack::cgeqp3);
}

#[cfg(test)]
mod tests {

    use super::*;

    macro_rules! pivoted_qr_tests {

    ($($name:ident: $scalar:ty, $dim:expr,)*) => {

        $(

        #[test]
        fn $name() {
            use crate::prelude::Random;
            use ndarray_linalg::Norm;
            use ndarray_linalg::Scalar;

            let m = $dim.0;
            let n = $dim.1;

            let mut rng = rand::thread_rng();
            let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), 1.0, 1E-5, &mut rng);

            let qr_result = mat.pivoted_qr().unwrap();

            let prod = qr_result.q.dot(&qr_result.r);

            // Check orthogonality of Q.T x Q

            let qtq = qr_result.q.t().map(|&item| item.conj()).dot(&qr_result.q);

            for ((i, j), &val) in qtq.indexed_iter() {
                if i == j {
                    let rel_diff = (val - 1.0).abs();
                    assert!(rel_diff < 1E-6);
                } else {
                    assert!(val.abs() < 1E-6);
                }
            }

            // Check that the product is correct.

            for (col_index, col) in prod.axis_iter(ndarray::Axis(1)).enumerate() {
                let perm_index = qr_result.ind[col_index];
                let diff = col.to_owned() - mat.index_axis(ndarray::Axis(1), perm_index);
                let rel_diff = diff.norm_l2() / mat.index_axis(ndarray::Axis(1), perm_index).norm_l2();

                assert!(rel_diff < 1E-6);
            }
        }
                )*
            };
        }

    pivoted_qr_tests! {
        pivoted_qr_test_thin_f64: f64, (100, 50),
        pivoted_qr_test_thin_f32: f32, (100, 50),
        pivoted_qr_test_thin_c64: ndarray_linalg::c64, (100, 50),
        pivoted_qr_test_thin_c32: ndarray_linalg::c32, (100, 50),
        pivoted_qr_test_thick_f64: f64, (50, 100),
        pivoted_qr_test_thick_f32: f32, (50, 100),
        pivoted_qr_test_thick_c64: ndarray_linalg::c64, (50, 100),
        pivoted_qr_test_thick_c32: ndarray_linalg::c32, (50, 100),
    }
}
