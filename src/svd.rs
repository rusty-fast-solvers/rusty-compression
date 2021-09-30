//! Define an SVD container and conversion tools.

use crate::qr::{QRData, QR};
use crate::CompressionType;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Zip};
use num::ToPrimitive;
use rusty_base::types::Result;
use rusty_base::types::{c32, c64, Scalar};

pub struct SVDData<A: Scalar> {
    /// The U matrix
    pub u: Array2<A>,
    /// The array of singular values
    pub s: Array1<A::Real>,
    /// The vt matrix
    pub vt: Array2<A>,
}

pub trait SVD {
    type A: Scalar;

    fn nrows(&self) -> usize {
        self.get_u().nrows()
    }
    fn ncols(&self) -> usize {
        self.get_vt().ncols()
    }
    fn rank(&self) -> usize {
        self.get_u().ncols()
    }
    fn to_mat(&self) -> Array2<Self::A> {
        let mut scaled_vt =
            Array2::<Self::A>::zeros((self.get_vt().nrows(), self.get_vt().ncols()));
        scaled_vt.assign(&self.get_vt());

        Zip::from(scaled_vt.axis_iter_mut(Axis(0)))
            .and(self.get_s().view())
            .apply(|mut row, &s_elem| {
                row.map_inplace(|item| *item *= <Self::A as Scalar>::from_real(s_elem))
            });

        self.get_u().dot(&scaled_vt)
    }
    fn to_qr(self) -> Result<QRData<Self::A>>;

    fn compress(&self, compression_type: CompressionType) -> Result<SVDData<Self::A>> {
        match compression_type {
            CompressionType::ADAPTIVE(tol) => self.compress_svd_tolerance(tol),
            CompressionType::RANK(rank) => self.compress_svd_rank(rank),
        }
    }

    fn compress_svd_rank(&self, mut max_rank: usize) -> Result<SVDData<Self::A>> {
        let (u, s, vt) = (self.get_u(), self.get_s(), self.get_vt());

        if max_rank > s.len() {
            max_rank = s.len()
        }

        let u = u.slice(s![.., 0..max_rank]);
        let s = s.slice(s![0..max_rank]);
        let vt = vt.slice(s![0..max_rank, ..]);

        Ok(SVDData {
            u: u.into_owned(),
            s: s.into_owned(),
            vt: vt.into_owned(),
        })
    }

    fn compress_svd_tolerance(&self, tol: f64) -> Result<SVDData<Self::A>> {
        assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

        let first_val = self.get_s()[0];

        let pos = self
            .get_s()
            .iter()
            .position(|&item| (item / first_val).to_f64().unwrap() < tol);

        match pos {
            Some(index) => self.compress_svd_rank(index),
            None => Err("Could not compress operator to desired tolerance."),
        }
    }

    fn new(arr: ArrayView2<Self::A>) -> Result<SVDData<Self::A>>;

    fn get_u(&self) -> ArrayView2<Self::A>;
    fn get_s(&self) -> ArrayView1<<Self::A as Scalar>::Real>;
    fn get_vt(&self) -> ArrayView2<Self::A>;

    fn get_u_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_s_mut(&mut self) -> ArrayViewMut1<<Self::A as Scalar>::Real>;
    fn get_vt_mut(&mut self) -> ArrayViewMut2<Self::A>;
}

macro_rules! svd_impl {
    ($scalar:ty) => {
        impl SVD for SVDData<$scalar> {
            type A = $scalar;

            fn get_u(&self) -> ArrayView2<Self::A> {
                self.u.view()
            }

            fn get_s(&self) -> ArrayView1<<Self::A as Scalar>::Real> {
                self.s.view()
            }
            fn get_vt(&self) -> ArrayView2<Self::A> {
                self.vt.view()
            }

            fn get_u_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.u.view_mut()
            }
            fn get_s_mut(&mut self) -> ArrayViewMut1<<Self::A as Scalar>::Real> {
                self.s.view_mut()
            }
            fn get_vt_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.vt.view_mut()
            }

            fn to_qr(self) -> Result<QRData<Self::A>> {
                let (u, s, mut vt) = (self.u, self.s, self.vt);

                Zip::from(vt.axis_iter_mut(Axis(0)))
                    .and(s.view())
                    .apply(|mut row, &s_elem| {
                        row.map_inplace(|item| *item *= <Self::A as Scalar>::from_real(s_elem))
                    });

                let mut qr = QRData::<$scalar>::new(vt.view())?;
                qr.q = u.dot(&qr.q);

                Ok(qr)
            }

            fn new(arr: ArrayView2<Self::A>) -> Result<SVDData<Self::A>> {
                use crate::compute_svd::ComputeSVD;

                <$scalar>::compute_svd(arr)
            }
        }
    };
}

svd_impl!(f32);
svd_impl!(f64);
svd_impl!(c32);
svd_impl!(c64);

#[cfg(test)]
mod tests {

    use super::*;
    use crate::helpers::RelDiff;
    use crate::random_matrix::RandomMatrix;
    use crate::CompressionType;
    use ndarray::Axis;
    use ndarray_linalg::OperationNorm;

    macro_rules! svd_to_qr_tests {
        ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {
            $(
            #[test]
            fn $name() {
                let m = $dim.0;
                let n = $dim.1;

                let mut rng = rand::thread_rng();
                let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), 1.0, 1E-10, &mut rng);

                let svd = SVDData::<$scalar>::new(mat.view()).unwrap();

                // Perform a QR decomposition and recover the original matrix.
                let actual = svd.to_qr().unwrap().to_mat();

                assert!(<$scalar>::rel_diff_fro(actual.view(), mat.view()) < $tol);

                assert!(
                    (actual - mat.view()).opnorm_fro().unwrap() / mat.opnorm_fro().unwrap() < $tol
                );
            }
            )*
        };
    }

    macro_rules! svd_compression_by_rank_tests {

        ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

            $(

        #[test]
        fn $name() {
            let m = $dim.0;
            let n = $dim.1;
            let rank: usize = 20;

            let sigma_max = 1.0;
            let sigma_min = 1E-10;
            let mut rng = rand::thread_rng();
            let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

            let svd = SVDData::<$scalar>::new(mat.view()).unwrap().compress(CompressionType::RANK(rank)).unwrap();

            // Compare with original matrix

            assert!(svd.u.len_of(Axis(1)) == rank);
            assert!(svd.vt.len_of(Axis(0)) == rank);

            assert!(<$scalar>::rel_diff_fro(svd.to_mat().view(), mat.view()) < $tol);
        }

            )*

        }
    }

    macro_rules! svd_compression_by_tol_tests {

        ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

            $(

        #[test]
        fn $name() {
            let m = $dim.0;
            let n = $dim.1;

            let sigma_max = 1.0;
            let sigma_min = 1E-10;
            let mut rng = rand::thread_rng();
            let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

            let svd = SVDData::<$scalar>::new(mat.view()).unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();

            // Compare with original matrix

            assert!(<$scalar>::rel_diff_fro(svd.to_mat().view(), mat.view()) < $tol);
        }

            )*

        }
    }

    svd_to_qr_tests! {
        test_svd_to_qr_f32_thin: f32, (100, 50), 1E-5,
        test_svd_to_qr_c32_thin: ndarray_linalg::c32, (100, 50), 1E-5,
        test_svd_to_qr_f64_thin: f64, (100, 50), 1E-12,
        test_svd_to_qr_c64_thin: ndarray_linalg::c64, (100, 50), 1E-12,
        test_svd_to_qr_f32_thick: f32, (50, 100), 1E-5,
        test_svd_to_qr_c32_thick: ndarray_linalg::c32, (50, 100), 1E-5,
        test_svd_to_qr_f64_thick: f64, (50, 100), 1E-12,
        test_svd_to_qr_c64_thick: ndarray_linalg::c64, (50, 100), 1E-12,
    }

    svd_compression_by_rank_tests! {
        test_svd_compression_by_rank_f32_thin: f32, (100, 50), 1E-4,
        test_svd_compression_by_rank_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_svd_compression_by_rank_f64_thin: f64, (100, 50), 1E-4,
        test_svd_compression_by_rank_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_svd_compression_by_rank_f32_thick: f32, (50, 100), 1E-4,
        test_svd_compression_by_rank_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_svd_compression_by_rank_f64_thick: f64, (50, 100), 1E-4,
        test_svd_compression_by_rank_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }

    svd_compression_by_tol_tests! {
        test_svd_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
        test_svd_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
        test_svd_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
        test_svd_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
        test_svd_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
        test_svd_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
        test_svd_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
        test_svd_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
    }
}
