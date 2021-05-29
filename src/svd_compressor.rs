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
) -> (Array2<T>, Array2<T>) {
    match compression_type {
        CompressionType::ADAPTIVE(tol) => compress_svd_tolerance(mat.array_view(), tol),
        CompressionType::RANK(rank) => compress_svd_rank(mat.array_view(), rank),
    }
}

pub fn compress_svd_rank<T: Scalar + Lapack>(
    mat: ArrayView2<T>,
    max_rank: usize,
) -> (Array2<T>, Array2<T>) {
    let (u, sigma, vt) = mat
        .to_owned()
        .svddc_into(UVTFlag::Some)
        .expect("`compress_svd_rank_based`: SVD computation failed.");

    low_rank_from_reduced_svd(u.unwrap(), sigma, vt.unwrap(), max_rank)
}

pub fn compress_svd_tolerance<T: Scalar + Lapack>(
    mat: ArrayView2<T>,
    tol: f64,
) -> (Array2<T>, Array2<T>) {
    let tol = num::traits::cast::cast::<f64, T::Real>(tol).unwrap();

    let (u, sigma, vt) = mat
        .to_owned()
        .svddc_into(UVTFlag::Some)
        .expect("`compress_svd_rank_based`: SVD computation failed.");

    let pos = sigma.iter().position(|&item| tol < item / sigma[0]);

    match pos {
        Some(index) => low_rank_from_reduced_svd(u.unwrap(), sigma, vt.unwrap(), index),
        None => (u.unwrap(), vt.unwrap()),
    }
}

fn low_rank_from_reduced_svd<T: Scalar>(
    u: Array2<T>,
    sigma: Array1<T::Real>,
    vt: Array2<T>,
    max_rank: usize,
) -> (Array2<T>, Array2<T>) {
    if max_rank < sigma.len() {
        let u = u.slice(s![.., 0..max_rank]).to_owned();
        let sigma = sigma.slice(s![0..max_rank]).mapv(|item| T::from_real(item));
        let mut vt = vt.slice(s![0..max_rank, ..]).to_owned();
        Zip::from(vt.axis_iter_mut(Axis(0)))
            .and(sigma.view())
            .apply(|mut row, &sigma_elem| row.map_inplace(|item| *item *= sigma_elem));

        (u, vt)
    } else {
        (u, vt)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use ndarray_linalg::QR;


    fn get_matrix(m: usize, n: usize, tol: f64) {

        let mut rng = rand::thread_rng();


        let mut v = Array2::<f64>::zeros((m, n));
        v.map_inplace(|item| *item = rng.gen::<f64>());

        let (v, _) =  v.qr().unwrap();



        
    }

    //#[test]

}