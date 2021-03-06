//! Random sampling of matrices
//!
//! This module defines traits for the randomized sampling of the range of a linear operator
//! and the associated computation of randomized QR and Singular Value Decompositions.
//!
//! In many applications we want to complete an approximate low-rank approximation of a linear
//! operator but do not have access to the whole matrix, or computing the full QR or SVD of a
//! matrix is too expensive. In these cases techniques from randomized linear algebra can be used
//! to compute approximate low-rank decompositions.
//! 
//! Assume we have an operator $A$. We have two conditions on $A$.
//! 
//! 1. We need to be able to compute $y=Ax$ for a given vector $x$.
//! 2. We need to be able to compute $y=A^Hx$ for a given vector $x$.
//! 
//! These abilities are implemented through the traits [MatVec](crate::types::MatVec) and [ConjMatVec](crate::types::ConjMatVec).
//! Implementing these traits also automatically implements the corresponding traits
//! [MatMat](crate::types::MatMat) and [ConjMatMat](crate::types::ConjMatMat), which
//! are the corresponding versions for multiplications with a matrix $X$ instead of a vector $x$.
//! The routines in this module use the latter traits. For performance reasons it may sometimes be
//! preferable to directly implement [MatMat](crate::types::MatMat) and [ConjMatMat](crate::types::ConjMatMat)
//! instead of relying on the corresponding vector versions.
//! 
//! In the following we describe the traits in more detail.
//! 
//! - [`SampleRange`]: This trait can be used to randomly sample the range of an operator by specifying a target
//! rank. It only requires the [MatMat](crate::types::MatMat) trait.
//! - [`SampleRangePowerIteration`]: This trait is an improved version of [`SampleRange']. It uses a power
//!   iteration to give a more precise result but requires the [ConjMatMat](crate::types::ConjMatMat) trait
//!   to be implemented.
//! - [`AdaptiveSampling`]: This trait samples the range of an operator adaptively through specifying
//!   a target tolerance. The error tolerance is checked probabilistically.
//! 
//! Once we have sampled the range of an operator we can use the method 
//! [compute_from_range_estimate](crate::qr::QRTraits::compute_from_range_estimate) of the
//! [QRTraits](crate::qr::QRTraits) to obtain an approximate QR Decomposition of the operator, from
//! which we can for example compute an interpolative decomposition. Alternatively, we can use
//! the [corresponding method](crate::svd::SVDTraits::compute_from_range_estimate) to compute an
//! approximate Singular Value Decomposition of the operator.


use crate::qr::{QRTraits, QR};
use crate::random_matrix::RandomMatrix;
use crate::CompressionType;
use ndarray::{concatenate, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::Norm;
use num::ToPrimitive;
use rand::Rng;
use crate::types::{c32, c64, Result, Scalar, MatMat, ConjMatMat};


/// Randomly sample the range of an operator
///
/// Let $A\in\mathbb{C}{m\times n}$ be a matrix. To sample the range of rank $k$ one can multiply
/// $A$ by a Gaussian random matrix $\Omega$ of dimension $n\times k + p$, where $p$ is a small oversampling
/// parameter. The result of this product is post-processed by a pivoted QR decomposition and the first $k$
/// columns of the $Q$ matrix in the pivoted QR decomposition returned.
pub trait SampleRange<A: Scalar> {
    /// Randomly sample the range of an operator.
    /// Return an approximate orthogonal basis of the dominant range.
    /// # Arguments
    /// * `k`: The target rank of the basis for the range.
    /// * `p`: Oversampling parameter. `p` should be chosen small. A typical size
    ///      is p=5.
    /// * `rng`: The random number generator.
    fn sample_range_by_rank<R: Rng>(
        &self,
        k: usize,
        p: usize,
        rng: &mut R,
    ) -> Result<Array2<A>>;
}

/// Randomly sample the range of an operator through a power iteration
///
/// Let $A\in\mathbb{C}{m\times n}$ be a matrix. To sample the range of rank $k$ one can multiply
/// $A$ by a Gaussian random matrix $\Omega$ of dimension $n\times k + p$, where $p$ is a small oversampling
/// parameter. To improve the accuracy of the range computation this is then `it_count` times multiplied with the
/// operator $AA^H$. Each intermediate result is QR orthogonalized to stabilise this power iteration.
/// The result of the power iteration is post-processed by a pivoted QR decomposition and the first $k$
/// columns of the $Q$ matrix in the pivoted QR decomposition returned.
pub trait SampleRangePowerIteration<A: Scalar> {
    /// Randomly sample the range of an operator refined through a power iteration
    /// Return an approximate orthogonal basis of the dominant range.
    /// # Arguments
    /// * `k`: The target rank of the basis for the range.
    /// * `p`: Oversampling parameter. `p` should be chosen small. A typical size
    ///      is p=5.
    /// * `it_count`: The number of steps in the power iteration. For `it_count = 0` the
    ///               routine is identical to `sample_range_by_rank`.
    fn sample_range_power_iteration<R: Rng>(
        &self,
        k: usize,
        p: usize,
        it_count: usize,
        rng: &mut R,
    ) -> Result<Array2<A>>;
}

macro_rules! sample_range_impl {
    ($scalar:ty) => {
        impl<Op: MatMat<A = $scalar>> SampleRange<$scalar> for Op {
            fn sample_range_by_rank<R: Rng>(
                &self,
                k: usize,
                p: usize,
                rng: &mut R,
            ) -> Result<Array2<$scalar>> {
                let m = self.ncols();

                let omega = <$scalar>::random_gaussian((m, k + p), rng);
                let basis = self.matmat(omega.view());

                let qr = QR::<$scalar>::compute_from(basis.view())?
                    .compress(CompressionType::RANK(k))?;

                Ok(qr.get_q().to_owned())
            }
        }
    };
}

sample_range_impl!(f32);
sample_range_impl!(f64);
sample_range_impl!(c32);
sample_range_impl!(c64);

macro_rules! sample_range_power_impl {
    ($scalar:ty) => {
        impl<Op: ConjMatMat<A = $scalar>> SampleRangePowerIteration<$scalar> for Op {
            fn sample_range_power_iteration<R: Rng>(
                &self,
                k: usize,
                p: usize,
                it_count: usize,
                rng: &mut R,
            ) -> Result<Array2<$scalar>> {
                let m = self.ncols();

                let omega = <$scalar>::random_gaussian((m, k + p), rng);
                let op_omega = self.matmat(omega.view());
                let mut res = op_omega.clone();

                for index in 0..it_count {
                    let qr = QR::<$scalar>::compute_from(op_omega.view())?;
                    let q = qr.get_q();

                    let qr = QR::<$scalar>::compute_from(self.conj_matmat(q).view())?;
                    let w = qr.get_q();
                    let op_omega = self.matmat(w);
                    if index == it_count - 1 {
                        res.assign(&op_omega);
                    }
                }

                let compressed =
                    QR::<$scalar>::compute_from(res.view())?.compress(CompressionType::RANK(k))?;

                Ok(compressed.get_q().to_owned())
            }
        }
    };
}

sample_range_power_impl!(f32);
sample_range_power_impl!(f64);
sample_range_power_impl!(c32);
sample_range_power_impl!(c64);

/// Trait defining the maximum column norm of an operator
///
/// If $A\in\mathbb{C}^{m\times n}$ the maximum column-norm is
/// computed by first taking the Euclidian norm of each column and then
/// returning the maximum of the column norms.
pub trait MaxColNorm<A: Scalar> {
    // For a given matrix return the maximum column norm.
    fn max_col_norm(&self) -> A::Real;
}

macro_rules! impl_max_col_norm {
    ($scalar:ty) => {
        impl<S: Data<Elem = $scalar>> MaxColNorm<$scalar> for ArrayBase<S, Ix2> {
            // For a given matrix return the maximum column norm.
            fn max_col_norm(&self) -> <$scalar as Scalar>::Real {
                let mut max_val = num::zero::<<$scalar as Scalar>::Real>();

                for col in self.axis_iter(Axis(1)) {
                    max_val = num::Float::max(max_val, col.norm_l2());
                }
                max_val
            }
        }
    };
}

impl_max_col_norm!(f32);
impl_max_col_norm!(f64);
impl_max_col_norm!(c32);
impl_max_col_norm!(c64);

/// This trait defines the adaptive sampling of the range of an operator.
pub trait AdaptiveSampling<A: Scalar> {
    /// Adaptively randomly sample the range of an operator up to a given tolerance.
    /// # Arguments
    /// * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
    /// * `sample_size`: Number of samples drawn together in each iteration.
    /// * `rng`: The random number generator.
    ///
    /// Returns a tuple (q, residuals), where `q` is an ndarray containing the orthogonalized columns of
    /// the range, and `residuals` is a vector of tuples of the form `(rank, rel_res)`, where `rel_res`
    /// is the estimated relative residual for the first `rank` columns of `q`.
    fn sample_range_adaptive<R: Rng>(
        &self,
        rel_tol: f64,
        sample_size: usize,
        rng: &mut R,
    ) -> Result<(Array2<A>, Vec<(usize, f64)>)>;
}

macro_rules! adaptive_sampling_impl {
    ($scalar:ty) => {
        impl<Op: ConjMatMat<A = $scalar>> AdaptiveSampling<$scalar> for Op {
            fn sample_range_adaptive<R: Rng>(
                &self,
                rel_tol: f64,
                sample_size: usize,
                rng: &mut R,
            ) -> Result<(Array2<$scalar>, Vec<(usize, f64)>)> {
                // This is a sampling factor. See Section 4.3 in Halko, Martinsson, Tropp,
                // Finding Structure with Randomness, SIAM Review.
                let tol_factor = num::cast::<f64, <$scalar as Scalar>::Real>(
                    10.0 * <f64 as num::traits::FloatConst>::FRAC_2_PI().sqrt(),
                )
                .unwrap();

                let m = self.ncols();
                let rel_tol = num::cast::<f64, <$scalar as Scalar>::Real>(rel_tol).unwrap();
                let omega = <$scalar>::random_gaussian((m, sample_size), rng);
                let mut op_omega = self.matmat(omega.view());
                // Randomized estimate of the original operator norm.
                let operator_norm = op_omega.max_col_norm() * tol_factor;
                let mut max_norm = operator_norm;
                let mut q = Array2::<$scalar>::zeros((self.nrows(), 0));
                let mut b = Array2::<$scalar>::zeros((0, self.ncols()));

                let mut residuals = Vec::<(usize, f64)>::new();

                while max_norm / operator_norm >= rel_tol {
                    // Orthogonalize against existing basis
                    if q.ncols() > 0 {
                        op_omega -= &q.dot(&q.t().map(|item| item.conj()).dot(&op_omega));
                    }
                    // Now do the QR of the vectors
                    let qr = QR::<$scalar>::compute_from(op_omega.view())?;
                    // Extend the b matrix
                    b = concatenate![
                        Axis(0),
                        b,
                        self.conj_matmat(qr.get_q()).t().map(|item| item.conj())
                    ];
                    // Extend the Q matrix
                    q = concatenate![Axis(1), q, qr.get_q()];

                    // Now compute new vectors
                    let omega = <$scalar>::random_gaussian((m, sample_size), rng);
                    op_omega = self.matmat(omega.view()) - q.dot(&b.dot(&omega));

                    // Update the error tolerance
                    max_norm = op_omega.max_col_norm() * tol_factor;
                    residuals.push((q.ncols(), (max_norm / operator_norm).to_f64().unwrap()));
                }

                Ok((q, residuals))
            }
        }
    };
}

adaptive_sampling_impl!(f32);
adaptive_sampling_impl!(c64);
adaptive_sampling_impl!(f64);
adaptive_sampling_impl!(c32);

// fn qr_from_range_estimate<R: Rng>(
//     &self,
//     rel_tol: f64,
//     sample_size: usize,
//     rng: &mut R,
// ) -> Result<QR<$scalar>> {
//     let (q, _) = self.sample_range_adaptive(rel_tol, sample_size, rng)?;

//     let b = self.conj_matmat(q.view()).t().map(|item| item.conj());
//     let qr = QR::<$scalar>::compute_from(b.view())?;

//     Ok(QR {
//         q: b.dot(&qr.get_q()),
//         r: qr.get_r().into_owned(),
//         ind: qr.get_ind().into_owned(),
//     })
// }

//     // Compute a QR decomposition via adaptive random range sampling.
//     // # Arguments
//     // * `op`: The operator for which to compute the QR decomposition.
//     // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
//     // * `sample_size` Number of samples drawn together in each iteration.
//     // * `rng`: The random number generator.
//     fn randomized_adaptive_qr<R: Rng>(
//         &self,
//         rel_tol: f64,
//         sample_size: usize,
//         rng: &mut R,
//     ) -> Result<QR<A>>;

//     // Compute a SVD decomposition via adaptive random range sampling.
//     // # Arguments
//     // * `op`: The operator for which to compute the QR decomposition.
//     // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
//     // * `sample_size` Number of samples drawn together in each iteration.
//     // * `rng`: The random number generator.
//     fn randomized_adaptive_svd<R: Rng>(
//         &self,
//         rel_tol: f64,
//         sample_size: usize,
//         rng: &mut R,
//     ) -> Result<SVD<A>>;
// }

//             // Compute a SVD decomposition via adaptive random range sampling.
//             // # Arguments
//             // * `op`: The operator for which to compute the QR decomposition.
//             // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
//             // * `sample_size` Number of samples drawn together in each iteration.
//             // * `rng`: The random number generator.
//             fn randomized_adaptive_svd<R: Rng>(
//                 &self,
//                 rel_tol: f64,
//                 sample_size: usize,
//                 rng: &mut R,
//             ) -> Result<SVD<$scalar>> {
//                 let (q, _) = self.sample_range_adaptive(rel_tol, sample_size, rng)?;

//                 let b = self.conj_matmat(q.view()).t().map(|item| item.conj());
//                 let svd = SVD::<$scalar>::compute_from(b.view())?;

//                 Ok(SVD {
//                     u: b.dot(&svd.u),
//                     s: svd.get_s().into_owned(),
//                     vt: svd.get_vt().into_owned(),
//                 })
//             }
//         }

// // Adaptively randomly sample the range of an operator up to a given tolerance.
// // # Arguments
// // * `op`: The operator for which to sample the range.
// // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
// // * `sample_size`: Number of samples drawn together in each iteration.
// // * `rng`: The random number generator.
// //
// // Returns a tuple (q, residuals), where `q` is an ndarray containing the orthogonalized columns of
// // the range, and `residuals` is a vector of tuples of the form `(rank, rel_res)`, where `rel_res`
// // is the estimated relative residual for the first `rank` columns of `q`.
// pub fn sample_range_adaptive<Op: ConjRowMatMat, R: Rng>(
//     op: &Op,
//     rel_tol: f64,
//     sample_size: usize,
//     rng: &mut R,
// ) -> Result<(Array2<Op::A>, Vec<(usize, f64)>)> {
//     // This is a sampling factor. See Section 4.3 in Halko, Martinsson, Tropp,
//     // Finding Structure with Randomness, SIAM Review.
//     let tol_factor = num::cast::<f64, <Op::A as Scalar>::Real>(
//         10.0 * <f64 as num::traits::FloatConst>::FRAC_2_PI().sqrt(),
//     )
//     .unwrap();

//     let m = op.ncols();
//     let rel_tol = num::cast::<f64, <Op::A as Scalar>::Real>(rel_tol).unwrap();
//     let omega = Op::A::random_gaussian((m, sample_size), rng);
//     let mut op_omega = op.matmat(omega.view());
//     // Randomized estimate of the original operator norm.
//     let operator_norm = max_col_norm(&op_omega) * tol_factor;
//     let mut max_norm = operator_norm;
//     let mut q = Array2::<Op::A>::zeros((op.nrows(), 0));
//     let mut b = Array2::<Op::A>::zeros((0, op.ncols()));

//     let mut residuals = Vec::<(usize, f64)>::new();

//     while max_norm / operator_norm >= rel_tol {
//         // Orthogonalize against existing basis
//         if q.ncols() > 0 {
//             op_omega -= &q.dot(&q.t().map(|item| item.conj()).dot(&op_omega));
//         }
//         // Now do the QR of the vectors
//         let (qtemp, _) = op_omega.qr().unwrap();
//         // Extend the b matrix
//         b = concatenate![Axis(0), b, op.conj_row_matmat(qtemp.view())];
//         // Extend the Q matrix
//         q = concatenate![Axis(1), q, qtemp];

//         // Now compute new vectors
//         let omega = Op::A::random_gaussian((m, sample_size), rng);
//         op_omega = op.matmat(omega.view()) - q.dot(&b.dot(&omega));

//         // Update the error tolerance
//         max_norm = max_col_norm(&op_omega) * tol_factor;
//         residuals.push((q.ncols(), (max_norm / operator_norm).to_f64().unwrap()));
//     }

//     Ok((q, residuals))
// }

// // Compute a QR decomposition via adaptive random range sampling.
// // # Arguments
// // * `op`: The operator for which to compute the QR decomposition.
// // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
// // * `sample_size` Number of samples drawn together in each iteration.
// // * `rng`: The random number generator.
// pub fn randomized_adaptive_qr<Op: ConjRowMatMat, R: Rng>(
//     op: &Op,
//     rel_tol: f64,
//     sample_size: usize,
//     rng: &mut R,
// ) -> Result<QRContainer<Op::A>> {
//     let (q, _) = sample_range_adaptive(op, rel_tol, sample_size, rng)?;

//     let b = op.conj_row_matmat(q.view());
//     let mut qr = b.pivoted_qr()?;

//     qr.q = b.dot(&qr.q);

//     Ok(qr)
// }

// // Compute a SVD decomposition via adaptive random range sampling.
// // # Arguments
// // * `op`: The operator for which to compute the QR decomposition.
// // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
// // * `sample_size` Number of samples drawn together in each iteration.
// // * `rng`: The random number generator.
// pub fn randomized_adaptive_svd<Op: ConjRowMatMat, R: Rng>(
//     op: &Op,
//     rel_tol: f64,
//     sample_size: usize,
//     rng: &mut R,
// ) -> Result<SVDContainer<Op::A>> {
//     let (q, _) = sample_range_adaptive(op, rel_tol, sample_size, rng)?;

//     let b = op.conj_row_matmat(q.view());
//     let mut svd = b.compute_svd()?;

//     svd.u = b.dot(&svd.u);

//     Ok(svd)
// }
