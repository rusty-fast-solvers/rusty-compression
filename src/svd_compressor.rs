//! Definition of SVD based compression routines

use super::CompressionType;
use crate::traits::ArrayProvider;
use ndarray::{s, Array1, Array2, ArrayView2, Axis, Zip};
use ndarray_linalg::types::{Lapack, Scalar};
use ndarray_linalg::SVDDCInto;
use ndarray_linalg::UVTFlag;

pub fn compress_svd<T: Scalar + Lapack, M: ArrayProvider<T>>(
    mat: M,
    compression_type: CompressionType,
) -> Result<(Array2<T>, Array2<T>), &'static str> {
    match compression_type {
        CompressionType::ADAPTIVE(tol) => compress_svd_tolerance(mat.array_view(), tol),
        CompressionType::RANK(rank) => compress_svd_rank(mat.array_view(), rank),
    }
}

pub fn compress_svd_rank<T: Scalar + Lapack>(
    mat: ArrayView2<T>,
    max_rank: usize,
) -> Result<(Array2<T>, Array2<T>), &'static str> {
    let (u, sigma, vt) = mat
        .to_owned()
        .svddc_into(UVTFlag::Some)
        .expect("`compress_svd_rank_based`: SVD computation failed.");

    Ok(low_rank_from_reduced_svd(
        u.unwrap(),
        sigma,
        vt.unwrap(),
        max_rank,
    ))
}

pub fn compress_svd_tolerance<T: Scalar + Lapack>(
    mat: ArrayView2<T>,
    tol: f64,
) -> Result<(Array2<T>, Array2<T>), &'static str> {
    assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

    let tol = num::traits::cast::cast::<f64, T::Real>(tol).unwrap();

    let (u, sigma, vt) = mat
        .to_owned()
        .svddc_into(UVTFlag::Some)
        .expect("`compress_svd_rank_based`: SVD computation failed.");

    let pos = sigma.iter().position(|&item| item / sigma[0] < tol);

    match pos {
        Some(index) => Ok(low_rank_from_reduced_svd(
            u.unwrap(),
            sigma,
            vt.unwrap(),
            index,
        )),
        None => Err("Could not compress operator to desired tolerance."),
    }
}

fn low_rank_from_reduced_svd<T: Scalar>(
    u: Array2<T>,
    sigma: Array1<T::Real>,
    vt: Array2<T>,
    mut max_rank: usize,
) -> (Array2<T>, Array2<T>) {
    if max_rank > sigma.len() {
        max_rank = sigma.len()
    }
    let u = u.slice(s![.., 0..max_rank]).to_owned();
    let sigma = sigma.slice(s![0..max_rank]).mapv(|item| T::from_real(item));
    let mut vt = vt.slice(s![0..max_rank, ..]).to_owned();
    Zip::from(vt.axis_iter_mut(Axis(0)))
        .and(sigma.view())
        .apply(|mut row, &sigma_elem| row.map_inplace(|item| *item *= sigma_elem));

    (u, vt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Random;

    use ndarray::Axis;

    #[test]
    fn test_svd_compression_by_rank() {
        let m = 50;
        let n = 30;
        let rank: usize = 10;

        let sigma_max = 1.0;
        let sigma_min = 1E-10;
        let mut rng = rand::thread_rng();
        let mat = f64::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

        let (a, bt) = compress_svd(mat.view(), CompressionType::RANK(rank)).unwrap();

        assert!(a.len_of(Axis(0)) == m);
        assert!(a.len_of(Axis(1)) == rank);
        assert!(bt.len_of(Axis(0)) == rank);
        assert!(bt.len_of(Axis(1)) == n);
        assert!(a.dot(&bt).rel_diff(mat) < 1E-3);
    }

    #[test]
    fn test_svd_compression_by_tol() {
        let m = 50;
        let n = 30;
        let tol = 1E-5;

        let sigma_max = 1.0;
        let sigma_min = 1E-10;
        let mut rng = rand::thread_rng();
        let mat = f64::random_approximate_low_rank_matrix((m, n), sigma_max, sigma_min, &mut rng);

        let (a, bt) = compress_svd(mat.view(), CompressionType::ADAPTIVE(tol)).unwrap();

        assert!(a.len_of(Axis(0)) == m);
        assert!(bt.len_of(Axis(1)) == n);
        let rel_diff = a.dot(&bt).rel_diff(&mat);
        assert!(rel_diff < 10.0 * tol);
    }
}