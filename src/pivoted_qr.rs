//! This module implements QR with pivoting by calling into the
//! corresponding Lapack routine. Pivoted QR is currently not
//! implemented in ndarray-linalg, making this module necessary.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, ShapeBuilder};
use ndarray_linalg::{Lapack, Scalar};

pub trait PivotedQR {
    type T: Scalar + Lapack;

    fn pivoted_qr(&self)
        -> Result<(Array2<Self::T>, Array2<Self::T>, Array1<usize>), &'static str>;
}

impl<S> PivotedQR for ArrayBase<S, Ix2>
where
    S: Data<Elem = f64>,
{
    type T = f64;

    fn pivoted_qr(
        &self,
    ) -> Result<(Array2<Self::T>, Array2<Self::T>, Array1<usize>), &'static str> {
        use imp::PivotedQRImpl;

        let m = self.nrows();
        let n = self.ncols();

        let mut mat_fortran = Array2::<Self::T>::zeros((m, n).f());
        mat_fortran.assign(&self);
        Self::T::pivoted_qr_impl(mat_fortran)
    }
}

mod imp {

    use lax;
    use ndarray::{s, Array1, Array2};
    use ndarray_linalg::layout::AllocatedArray;
    use ndarray_linalg::{IntoTriangular, Lapack, MatrixLayout, Scalar};
    use num::traits::{ToPrimitive, Zero};

    pub trait PivotedQRImpl
    where
        Self: Scalar + Lapack,
    {
        fn pivoted_qr_impl(
            mat: Array2<Self>,
        ) -> Result<(Array2<Self>, Array2<Self>, Array1<usize>), &'static str>;
        fn pivoted_qr_decomp(
            mat: &mut [Self],
            layout: MatrixLayout,
        ) -> Result<(Array1<Self>, Array1<usize>), i32>;
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
                ) -> Result<(Array2<Self>, Array2<Self>, Array1<usize>), &'static str> {
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

                    Ok((q_mat, r_mat, jpvt))
                }

                fn pivoted_qr_decomp(
                    mat: &mut [Self],
                    layout: MatrixLayout,
                ) -> Result<(Array1<Self>, Array1<usize>), i32> {
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
