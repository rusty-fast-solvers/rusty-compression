//! Random sampling of matrices
//! 
//! This module defines traits for the randomized sampling of the range of a linear operator
//! and the associated computation of randomized QR and Singular Value Decompositions.
//! To use these routines for a custom linear operator simply implement the `MatVec` trait and
//! if required the `ConjMatVec` trait.

use crate::qr::{QRTraits, QR};
use crate::random_matrix::RandomMatrix;
use crate::svd::{SVD, SVDTraits};
use crate::CompressionType;
use ndarray::{concatenate, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_linalg::Norm;
use num::ToPrimitive;
use rand::Rng;
use rusty_base::types::{c32, c64, Result, Scalar};

/// Matrix-Vector Product Trait
/// 
/// This trait defines an interface for operators that provide matrix-vector products.
pub trait MatVec {
    type A: Scalar;

    // Return the number of rows of the operator.
    fn nrows(&self) -> usize;

    // Return the number of columns of the operator.
    fn ncols(&self) -> usize;

    // Return the matrix vector product of an operator with a vector.
    fn matvec(&self, mat: ArrayView1<Self::A>) -> Array1<Self::A>;
}

/// Matrix-Matrix Product Trait
/// 
/// This trait defines the application of a linear operator $A$ to a matrix X representing multiple columns.
/// If it is not implemented then a default implementation is used based on the `MatVec` trait applied to the
/// individual columns of X.
pub trait MatMat: MatVec {
    // Return the matrix-matrix product of an operator with a matrix.
    fn matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        let mut output = Array2::<Self::A>::zeros((self.nrows(), mat.ncols()));

        for (index, col) in mat.axis_iter(Axis(1)).enumerate() {
            output
                .index_axis_mut(Axis(1), index)
                .assign(&self.matvec(col));
        }

        output
    }
}

/// Trait describing the product of the conjugate adjoint of an operator with a vector
/// 
/// In the case that the operator is a matrix then this simply describes the action $A^Hx$,
/// where $x$ is a vector and $A^H$ the complex conjugate adjoint of $A$.
pub trait ConjMatVec: MatVec {
    // If `self` is a linear operator return the product of the conjugate of `self`
    // with a vector.
    fn conj_matvec(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A>;
}

/// Trait describing the action of the conjugate adjoint of an operator with a matrix
/// 
/// In the case that the operator is a matrix then this simply describes the action $A^HX$,
/// where $X$ is another matrix and $A^H$ the complex conjugate adjoint of $A$. If this trait
/// is not implemented then a default implementation based on the `ConjMatVec` trait is used.
pub trait ConjMatMat: MatMat + ConjMatVec {
    // Return the product of the complex conjugate of `self` with a given matrix.
    fn conj_matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        let mut output = Array2::<Self::A>::zeros((self.nrows(), mat.ncols()));

        for (index, col) in mat.axis_iter(Axis(1)).enumerate() {
            output
                .index_axis_mut(Axis(1), index)
                .assign(&self.conj_matvec(col));
        }

        output
    }
}

impl<A, S> MatVec for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type A = A;

    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn matvec(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A> {
        self.dot(&vec)
    }
}

impl<A, S> ConjMatVec for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn conj_matvec(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A> {
        vec.map(|item| item.conj())
            .dot(self)
            .map(|item| item.conj())
    }
}

impl<A, S> MatMat for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        self.dot(&mat)
    }
}

impl<A, S> ConjMatMat for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn conj_matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        mat.t()
            .map(|item| item.conj())
            .dot(self)
            .t()
            .map(|item| item.conj())
    }
}

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
    /// * `op`: The operator for which to sample the range.
    /// * `k`: The target rank of the basis for the range.
    /// * `p`: Oversampling parameter. `p` should be chosen small. A typical size
    ///      is p=5.
    /// * `rng`: The random number generator.
    fn sample_range_by_rank<R: Rng>(
        op: &Self,
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
    /// * `op`: The operator for which to sample the range.
    /// * `k`: The target rank of the basis for the range.
    /// * `p`: Oversampling parameter. `p` should be chosen small. A typical size
    ///      is p=5.
    /// * `it_count`: The number of steps in the power iteration. For `it_count = 0` the
    ///               routine is identical to `sample_range_by_rank`.
    fn sample_range_power_iteration<R: Rng>(
        op: &Self,
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
                op: &Self,
                k: usize,
                p: usize,
                rng: &mut R,
            ) -> Result<Array2<$scalar>> {
                let m = op.ncols();

                let omega = <$scalar>::random_gaussian((m, k + p), rng);
                let basis = op.matmat(omega.view());

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
                op: &Op,
                k: usize,
                p: usize,
                it_count: usize,
                rng: &mut R,
            ) -> Result<Array2<$scalar>> {
                let m = op.ncols();

                let omega = <$scalar>::random_gaussian((m, k + p), rng);
                let op_omega = op.matmat(omega.view());
                let mut res = op_omega.clone();

                for index in 0..it_count {
                    let qr = QR::<$scalar>::compute_from(op_omega.view())?;
                    let q = qr.get_q();

                    let qr = QR::<$scalar>::compute_from(op.conj_matmat(q).view())?;
                    let w = qr.get_q();
                    let op_omega = op.matmat(w);
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

pub trait AdaptiveSampling<A: Scalar> {
    // Adaptively randomly sample the range of an operator up to a given tolerance.
    // # Arguments
    // * `op`: The operator for which to sample the range.
    // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
    // * `sample_size`: Number of samples drawn together in each iteration.
    // * `rng`: The random number generator.
    //
    // Returns a tuple (q, residuals), where `q` is an ndarray containing the orthogonalized columns of
    // the range, and `residuals` is a vector of tuples of the form `(rank, rel_res)`, where `rel_res`
    // is the estimated relative residual for the first `rank` columns of `q`.
    fn sample_range_adaptive<R: Rng>(
        &self,
        rel_tol: f64,
        sample_size: usize,
        rng: &mut R,
    ) -> Result<(Array2<A>, Vec<(usize, f64)>)>;

    // Compute a QR decomposition via adaptive random range sampling.
    // # Arguments
    // * `op`: The operator for which to compute the QR decomposition.
    // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
    // * `sample_size` Number of samples drawn together in each iteration.
    // * `rng`: The random number generator.
    fn randomized_adaptive_qr<R: Rng>(
        &self,
        rel_tol: f64,
        sample_size: usize,
        rng: &mut R,
    ) -> Result<QR<A>>;

    // Compute a SVD decomposition via adaptive random range sampling.
    // # Arguments
    // * `op`: The operator for which to compute the QR decomposition.
    // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
    // * `sample_size` Number of samples drawn together in each iteration.
    // * `rng`: The random number generator.
    fn randomized_adaptive_svd<R: Rng>(
        &self,
        rel_tol: f64,
        sample_size: usize,
        rng: &mut R,
    ) -> Result<SVD<A>>;
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
            fn randomized_adaptive_qr<R: Rng>(
                &self,
                rel_tol: f64,
                sample_size: usize,
                rng: &mut R,
            ) -> Result<QR<$scalar>> {
                let (q, _) = self.sample_range_adaptive(rel_tol, sample_size, rng)?;

                let b = self.conj_matmat(q.view()).t().map(|item| item.conj());
                let qr = QR::<$scalar>::compute_from(b.view())?;

                Ok(QR {
                    q: b.dot(&qr.get_q()),
                    r: qr.get_r().into_owned(),
                    ind: qr.get_ind().into_owned(),
                })
            }

            // Compute a SVD decomposition via adaptive random range sampling.
            // # Arguments
            // * `op`: The operator for which to compute the QR decomposition.
            // * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
            // * `sample_size` Number of samples drawn together in each iteration.
            // * `rng`: The random number generator.
            fn randomized_adaptive_svd<R: Rng>(
                &self,
                rel_tol: f64,
                sample_size: usize,
                rng: &mut R,
            ) -> Result<SVD<$scalar>> {
                let (q, _) = self.sample_range_adaptive(rel_tol, sample_size, rng)?;

                let b = self.conj_matmat(q.view()).t().map(|item| item.conj());
                let svd = SVD::<$scalar>::compute_from(b.view())?;

                Ok(SVD {
                    u: b.dot(&svd.u),
                    s: svd.get_s().into_owned(),
                    vt: svd.get_vt().into_owned(),
                })
            }
        }
    };
}

adaptive_sampling_impl!(f32);
adaptive_sampling_impl!(c64);
adaptive_sampling_impl!(f64);
adaptive_sampling_impl!(c32);

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
