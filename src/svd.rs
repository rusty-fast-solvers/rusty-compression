//! Define an SVD container and conversion tools.

use crate::CompressionType;
use ndarray::{s, Array1, Array2, Axis, Zip};
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
}

macro_rules! svd_impl {
    ($scalar:ty) => {
        impl SVD for $scalar {
            type A = $scalar;
        }
    };
}

svd_impl!(f32);
svd_impl!(f64);
svd_impl!(c32);
svd_impl!(c64);

// impl<A: ScalarType> SVDContainer<A> {
//     pub fn to_qr(self) -> Result<QRContainer<A>> {
//         let (u, s, mut vt) = (self.u, self.s, self.vt);

//         Zip::from(vt.axis_iter_mut(Axis(0)))
//             .and(s.view())
//             .apply(|mut row, &s_elem| row.map_inplace(|item| *item *= A::from_real(s_elem)));

//         // Now compute the qr of vt

//         let mut qr = vt.pivoted_qr()?;
//         qr.q = u.dot(&qr.q);

//         Ok(qr)
//     }

//     pub fn to_mat(&self) -> Array2<A> {
//         let mut scaled_vt = Array2::<A>::zeros((self.vt.nrows(), self.vt.ncols()));
//         scaled_vt.assign(&self.vt);

//         Zip::from(scaled_vt.axis_iter_mut(Axis(0)))
//             .and(self.s.view())
//             .apply(|mut row, &s_elem| row.map_inplace(|item| *item *= A::from_real(s_elem)));

//         self.u.dot(&scaled_vt)
//     }

//     pub fn compress(self, compression_type: CompressionType) -> Result<SVDContainer<A>> {
//         compress_svd(self, compression_type)
//     }
// }

// fn compress_svd<T: ScalarType>(
//     svd_container: SVDContainer<T>,
//     compression_type: CompressionType,
// ) -> Result<SVDContainer<T>> {
//     match compression_type {
//         CompressionType::ADAPTIVE(tol) => compress_svd_tolerance(svd_container, tol),
//         CompressionType::RANK(rank) => compress_svd_rank(svd_container, rank),
//     }
// }

// fn compress_svd_rank<A: ScalarType>(
//     svd_container: SVDContainer<A>,
//     mut max_rank: usize,
// ) -> Result<SVDContainer<A>> {
//     let (u, s, vt) = (svd_container.u, svd_container.s, svd_container.vt);

//     if max_rank > s.len() {
//         max_rank = s.len()
//     }

//     let u = u.slice_move(s![.., 0..max_rank]);
//     let s = s.slice_move(s![0..max_rank]);
//     let vt = vt.slice_move(s![0..max_rank, ..]);

//     Ok(SVDContainer { u, s, vt })
// }

// fn compress_svd_tolerance<A: ScalarType>(
//     svd_container: SVDContainer<A>,
//     tol: f64,
// ) -> Result<SVDContainer<A>> {
//     assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

//     let pos = svd_container
//         .s
//         .iter()
//         .position(|&item| (item / svd_container.s[0]).to_f64().unwrap() < tol);

//     match pos {
//         Some(index) => compress_svd_rank(svd_container, index),
//         None => Err("Could not compress operator to desired tolerance."),
//     }
// }

// #[cfg(test)]
// mod tests {

//     use crate::compute_svd::ComputeSVD;
//     use crate::prelude::CompressionType;
//     use crate::prelude::RelDiff;
//     use crate::random_matrix::RandomMatrix;
//     use ndarray::Axis;
//     use ndarray_linalg::OperationNorm;

//     macro_rules! svd_to_qr_tests {
//         ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {
//             $(
//             #[test]
//             fn $name() {
//                 let m = $dim.0;
//                 let n = $dim.1;

//                 let mut rng = rand::thread_rng();
//                 let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), 1.0, 1E-10, &mut rng);

//                 let svd = mat.compute_svd().unwrap();

//                 // Perform a QR decomposition and recover the original matrix.
//                 let actual = svd.to_qr().unwrap().to_mat();

//                 assert!(
//                     (actual - mat.view()).opnorm_fro().unwrap() / mat.opnorm_fro().unwrap() < $tol
//                 );
//             }
//             )*
//         };
//     }

//     macro_rules! svd_compression_by_rank_tests {

//         ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

//             $(

//         #[test]
//         fn $name() {
//             let m = $dim.0;
//             let n = $dim.1;
//             let rank: usize = 20;

//             let sigma_max = 1.0;
//             let sigma_min = 1E-10;
//             let mut rng = rand::thread_rng();
//             let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

//             let svd = mat.compute_svd().unwrap().compress(CompressionType::RANK(rank)).unwrap();

//             // Compare with original matrix

//             assert!(svd.u.len_of(Axis(1)) == rank);
//             assert!(svd.vt.len_of(Axis(0)) == rank);

//             assert!(svd.to_mat().rel_diff(&mat) < $tol);
//         }

//             )*

//         }
//     }

//     macro_rules! svd_compression_by_tol_tests {

//         ($($name:ident: $scalar:ty, $dim:expr, $tol:expr,)*) => {

//             $(

//         #[test]
//         fn $name() {
//             let m = $dim.0;
//             let n = $dim.1;

//             let sigma_max = 1.0;
//             let sigma_min = 1E-10;
//             let mut rng = rand::thread_rng();
//             let mat = <$scalar>::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

//             let svd = mat.compute_svd().unwrap().compress(CompressionType::ADAPTIVE($tol)).unwrap();

//             // Compare with original matrix

//             assert!(svd.to_mat().rel_diff(&mat) < $tol);
//         }

//             )*

//         }
//     }

//     svd_to_qr_tests! {
//         test_svd_to_qr_f32_thin: f32, (100, 50), 1E-5,
//         test_svd_to_qr_c32_thin: ndarray_linalg::c32, (100, 50), 1E-5,
//         test_svd_to_qr_f64_thin: f64, (100, 50), 1E-12,
//         test_svd_to_qr_c64_thin: ndarray_linalg::c64, (100, 50), 1E-12,
//         test_svd_to_qr_f32_thick: f32, (50, 100), 1E-5,
//         test_svd_to_qr_c32_thick: ndarray_linalg::c32, (50, 100), 1E-5,
//         test_svd_to_qr_f64_thick: f64, (50, 100), 1E-12,
//         test_svd_to_qr_c64_thick: ndarray_linalg::c64, (50, 100), 1E-12,
//     }

//     svd_compression_by_rank_tests! {
//         test_svd_compression_by_rank_f32_thin: f32, (100, 50), 1E-4,
//         test_svd_compression_by_rank_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
//         test_svd_compression_by_rank_f64_thin: f64, (100, 50), 1E-4,
//         test_svd_compression_by_rank_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
//         test_svd_compression_by_rank_f32_thick: f32, (50, 100), 1E-4,
//         test_svd_compression_by_rank_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
//         test_svd_compression_by_rank_f64_thick: f64, (50, 100), 1E-4,
//         test_svd_compression_by_rank_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
//     }

//     svd_compression_by_tol_tests! {
//         test_svd_compression_by_tol_f32_thin: f32, (100, 50), 1E-4,
//         test_svd_compression_by_tol_c32_thin: ndarray_linalg::c32, (100, 50), 1E-4,
//         test_svd_compression_by_tol_f64_thin: f64, (100, 50), 1E-4,
//         test_svd_compression_by_tol_c64_thin: ndarray_linalg::c64, (100, 50), 1E-4,
//         test_svd_compression_by_tol_f32_thick: f32, (50, 100), 1E-4,
//         test_svd_compression_by_tol_c32_thick: ndarray_linalg::c32, (50, 100), 1E-4,
//         test_svd_compression_by_tol_f64_thick: f64, (50, 100), 1E-4,
//         test_svd_compression_by_tol_c64_thick: ndarray_linalg::c64, (50, 100), 1E-4,
//     }
// }
