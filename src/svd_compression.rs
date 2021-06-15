//! Definition of SVD based compression routines

use super::prelude::{CompressionType, SVDContainer};
use crate::Result;
use ndarray::s;
use ndarray_linalg::types::{Lapack, Scalar};

pub trait CompressSVD {
    type A: Scalar + Lapack;

    fn compress(
        self,
        compression_type: CompressionType,
    ) -> Result<SVDContainer<Self::A>>;
}

impl<A> CompressSVD for SVDContainer<A>
where
    A: Scalar + Lapack,
{
    type A = A;

    fn compress(
        self,
        compression_type: CompressionType,
    ) -> Result<SVDContainer<A>> {
        compress_svd(self, compression_type)
    }
}

fn compress_svd<T: Scalar + Lapack>(
    svd_container: SVDContainer<T>,
    compression_type: CompressionType,
) -> Result<SVDContainer<T>> {
    match compression_type {
        CompressionType::ADAPTIVE(tol) => compress_svd_tolerance(svd_container, tol),
        CompressionType::RANK(rank) => compress_svd_rank(svd_container, rank),
    }
}

fn compress_svd_rank<A: Scalar + Lapack>(
    svd_container: SVDContainer<A>,
    mut max_rank: usize,
) -> Result<SVDContainer<A>> {
    let (u, s, vt) = (svd_container.u, svd_container.s, svd_container.vt);

    if max_rank > s.len() {
        max_rank = s.len()
    }

    let u = u.slice_move(s![.., 0..max_rank]);
    let s = s.slice_move(s![0..max_rank]);
    let vt = vt.slice_move(s![0..max_rank, ..]);

    Ok(SVDContainer { u, s, vt })
}

fn compress_svd_tolerance<A: Scalar + Lapack>(
    svd_container: SVDContainer<A>,
    tol: f64,
) -> Result<SVDContainer<A>>{
    assert!((tol < 1.0) && (0.0 <= tol), "Require 0 <= tol < 1.0");

    let tol = num::traits::cast::cast::<f64, A::Real>(tol).unwrap();
    let pos = svd_container.s.iter().position(|&item| item / svd_container.s[0] < tol);

    match pos {
        Some(index) => compress_svd_rank(svd_container, index),
        None => Err("Could not compress operator to desired tolerance."),
    }
}

//fn low_rank_from_reduced_svd<T: Scalar>(
//    u: Array2<T>,
//    sigma: Array1<T::Real>,
//    vt: Array2<T>,
//    mut max_rank: usize,
//) -> Result< {
//    if max_rank > sigma.len() {
//        max_rank = sigma.len()
//    }
//    let u = u.slice(s![.., 0..max_rank]).to_owned();
//    let sigma = sigma.slice(s![0..max_rank]).mapv(|item| T::from_real(item));
//    let mut vt = vt.slice(s![0..max_rank, ..]).to_owned();
//    Zip::from(vt.axis_iter_mut(Axis(0)))
//        .and(sigma.view())
//        .apply(|mut row, &sigma_elem| row.map_inplace(|item| *item *= sigma_elem));
//
//    (u, vt)
//}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
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

        let svd = mat.compute_svd().unwrap().compress(CompressionType::RANK(rank)).unwrap();

        // Compare with original matrix
        
        assert!(a.len_of(Axis(0)) == m);
        assert!(a.len_of(Axis(1)) == rank);
        assert!(bt.len_of(Axis(0)) == rank);
        assert!(bt.len_of(Axis(1)) == n);
        assert!(a.dot(&bt).rel_diff(&mat) < 1E-3);
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

        let svd = mat.compute_svd().unwrap().compress(CompressionType::ADAPTIVE(tol)).unwrap();

        assert!(a.len_of(Axis(0)) == m);
        assert!(bt.len_of(Axis(1)) == n);
        let rel_diff = a.dot(&bt).rel_diff(&mat);
        assert!(rel_diff < 10.0 * tol);
    }
}
