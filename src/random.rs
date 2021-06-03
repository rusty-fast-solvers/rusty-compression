//! Routines needed by the other modules.

use ndarray::Array2;
use ndarray_linalg::{Lapack, SVDDCInto, Scalar, UVTFlag};
use num::complex::Complex;
use num::traits::cast::cast;
use num::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub trait Random {
    type T: Scalar + Lapack;

    /// Generate a random Gaussian matrix.
    ///
    /// # Arguments
    ///
    /// * `dimension`: Tuple (rows, cols) specifying the number of rows and columns.
    /// * `rng`: The random number generator to use.
    fn random_gaussian<R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<Self::T>;

    /// Generate a random orthogonal matrix.
    ///
    /// This function creates a normally distributed random matrix,
    /// orthogonalizes it and returns the resulting orthogonal matrix.
    ///
    /// # Arguments
    ///
    /// * `dimension`: Tuple (rows, cols) specifying the number of rows and columns.
    /// * `rng`: The random number generator to use.
    fn random_orthogonal_matrix<R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<Self::T> {
        assert!(
            dimension.0 >= dimension.1,
            "Matrix must have at least as many rows as columns."
        );

        let mat = Self::random_gaussian(dimension, rng);

        let (u, _, _) = mat
            .svddc_into(UVTFlag::Some)
            .expect("`compress_svd_rank_based`: SVD computation failed.");
        u.unwrap()
    }

    /// Generate a random approximate low-rank matrix.
    ///
    /// This function generates a random approximate low-rank matrix
    /// with singular values logarithmically distributed between
    /// 'sigma_max` and `sigma_min`.
    ///
    /// # Arguments
    ///
    /// * `dimension`: Tuple (rows, cols) specifying the number of rows and columns.
    /// * `sigma_max`: Maximum singular value.
    /// * `sigma_min`: Minimum singular value.
    /// * `rng`: The random number generator to use.
    fn random_approximate_low_rank_matrix<R: Rng>(
        dimension: (usize, usize),
        sigma_max: f64,
        sigma_min: f64,
        rng: &mut R,
    ) -> Array2<Self::T> {
        use ndarray::Array;

        assert!(sigma_min < sigma_max, "`sigma_min` must be smaller than `sigma_max`");
        assert!(sigma_min > 0.0, "`sigma_min` must be positive.");

        let min_dim = std::cmp::min(dimension.0, dimension.1);

        let q1 = Self::random_orthogonal_matrix((dimension.0, min_dim), rng);
        let q2 = Self::random_orthogonal_matrix((min_dim, dimension.1), rng);
        let singvals = Array::geomspace(sigma_min, sigma_max, min_dim)
            .unwrap()
            .map(|&item| cast::<f64, Self::T>(item).unwrap());
        let sigma = Array2::from_diag(&singvals);
        q1.dot(&sigma.dot(&q2))
    }
}

impl Random for f64 {
    type T = f64;

    fn random_gaussian<R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<f64> {
        random_gaussian_real::<f64, R>(dimension, rng)
    }
}

impl Random for f32 {
    type T = f32;

    fn random_gaussian<R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<f32> {
        random_gaussian_real::<f32, R>(dimension, rng)
    }
}

impl Random for Complex<f64> {
    type T = Complex<f64>;

    fn random_gaussian<R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<Complex<f64>> {
        random_gaussian_complex::<f64, R>(dimension, rng)
    }
}

impl Random for Complex<f32> {
    type T = Complex<f32>;

    fn random_gaussian<R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<Complex<f32>> {
        random_gaussian_complex::<f32, R>(dimension, rng)
    }
}

fn random_gaussian_real<T: Float, R: Rng>(dimension: (usize, usize), rng: &mut R) -> Array2<T> {
    let mut mat = Array2::<T>::zeros(dimension);
    let normal = Normal::new(0.0, 1.0).unwrap();
    mat.map_inplace(|item| *item = cast::<f64, T>(normal.sample(rng)).unwrap());
    mat
}

/// Generate a random Gaussian matrix.
///
/// # Arguments
///
/// * `dimension`: Tuple (rows, cols) specifying the number of rows and columns.
/// * `rng`: The random number generator to use.
fn random_gaussian_complex<T: Float, R: Rng>(
    dimension: (usize, usize),
    rng: &mut R,
) -> Array2<Complex<T>> {
    let mut mat = Array2::<Complex<T>>::zeros(dimension);
    let normal = Normal::new(0.0, 1.0).unwrap();
    mat.map_inplace(|item| {
        let re = cast::<f64, T>(normal.sample(rng)).unwrap();
        let im = cast::<f64, T>(normal.sample(rng)).unwrap();
        *item = Complex::new(re, im);
    });
    mat
}
