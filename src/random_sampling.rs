//! Random sampling of matrices

use crate::prelude::CompressionType;
use crate::prelude::ComputeSVD;
use crate::prelude::PivotedQR;
use crate::prelude::QRContainer;
use crate::prelude::RandomMatrix;
use crate::prelude::SVDContainer;
use crate::prelude::ScalarType;
use crate::Result;
use ndarray::{concatenate, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_linalg::{Lapack, Norm, Scalar, QR};
use num::ToPrimitive;
use rand::Rng;

pub trait MatVec {
    type A: ScalarType;

    // Return the number of rows of the operator.
    fn nrows(&self) -> usize;

    // Return the number of columns of the operator.
    fn ncols(&self) -> usize;

    // Return the matrix vector product of an operator with a vector.
    fn matvec(&self, mat: ArrayView1<Self::A>) -> Array1<Self::A>;
}

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

pub trait ConjRowMatVec: MatVec {
    // If `self` is a linear operator return the product of the conjugate of the
    // row vector vec with `self`, e.g. `vec.conj().dot(&self)` if `self` is an `ndarray`.
    fn conj_row_matvec(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A>;
}

pub trait ConjRowMatMat: MatMat + ConjRowMatVec {
    // Return the product of the complex conjugate transpose of mat from the left
    // with `self`, e.g. `mat.t().conj().dot(&self)` if  `self` is an `ndarray`.
    fn conj_row_matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        let mut output = Array2::<Self::A>::zeros((mat.nrows(), self.ncols()));

        for (index, row) in mat.axis_iter(Axis(0)).enumerate() {
            output
                .index_axis_mut(Axis(0), index)
                .assign(&self.conj_row_matvec(row));
        }

        output
    }
}

impl<A, S> MatVec for ArrayBase<S, Ix2>
where
    A: ScalarType,
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

impl<A, S> ConjRowMatVec for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    fn conj_row_matvec(&self, vec: ArrayView1<Self::A>) -> Array1<Self::A> {
        vec.map(|item| item.conj()).dot(self)
    }
}

impl<A, S> MatMat for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    fn matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        self.dot(&mat)
    }
}

impl<A, S> ConjRowMatMat for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    fn conj_row_matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
        mat.t().map(|item| item.conj()).dot(self)
    }
}

// Randomly sample the range of an operator.
// Return an approximate orthogonal basis of the dominant range.
// # Arguments
// * `op`: The operator for which to sample the range.
// * `k`: The target rank of the basis for the range.
// * `p`: Oversampling parameter. `p` should be chosen small. A typical size
//      is p=5.
// * `rng`: The random number generator.
pub fn sample_range_by_rank<Op: MatMat, R: Rng>(
    op: &Op,
    k: usize,
    p: usize,
    rng: &mut R,
) -> Result<Array2<Op::A>> {
    let m = op.ncols();

    let omega = Op::A::random_gaussian((m, k + p), rng);
    let qr = op
        .matmat(omega.view())
        .pivoted_qr()?
        .compress(CompressionType::RANK(k))?;

    Ok(qr.q)
}

// Randomly sample the range of an operator refined through a power iteration
// Return an approximate orthogonal basis of the dominant range.
// # Arguments
// * `op`: The operator for which to sample the range.
// * `k`: The target rank of the basis for the range.
// * `p`: Oversampling parameter. `p` should be chosen small. A typical size
//      is p=5.
// * `it_count`: The number of steps in the power iteration. For `it_count = 0` the
//               routine is identical to `sample_range_by_rank`.
pub fn sample_range_by_rank_power_iteration<Op: ConjRowMatMat, R: Rng>(
    op: &Op,
    k: usize,
    p: usize,
    it_count: usize,
    rng: &mut R,
) -> Result<Array2<Op::A>> {
    let m = op.ncols();

    let omega = Op::A::random_gaussian((m, k + p), rng);
    let op_omega = op.matmat(omega.view());
    let mut res = op_omega.clone();

    for index in 0..it_count {
        let (q, _) = op_omega.qr().unwrap();
        let (w, _) = op
            .conj_row_matmat(q.view())
            .t()
            .map(|item| item.conj())
            .qr()
            .unwrap();
        let op_omega = op.matmat(w.view());
        if index == it_count - 1 {
            res.assign(&op_omega);
        }
    }

    let qr = res
        .matmat(omega.view())
        .pivoted_qr()?
        .compress(CompressionType::RANK(k))?;

    Ok(qr.q)
}

// For a given matrix return the maximum column norm.
fn max_col_norm<A: Scalar + Lapack, S: Data<Elem = A>>(mat: &ArrayBase<S, Ix2>) -> A::Real {
    let mut max_val = num::zero::<A::Real>();

    for col in mat.axis_iter(Axis(1)) {
        max_val = num::Float::max(max_val, col.norm_l2());
    }
    max_val
}

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
pub fn sample_range_adaptive<Op: ConjRowMatMat, R: Rng>(
    op: &Op,
    rel_tol: f64,
    sample_size: usize,
    rng: &mut R,
) -> Result<(Array2<Op::A>, Vec<(usize, f64)>)> {
    // This is a sampling factor. See Section 4.3 in Halko, Martinsson, Tropp,
    // Finding Structure with Randomness, SIAM Review.
    let tol_factor = num::cast::<f64, <Op::A as Scalar>::Real>(
        10.0 * <f64 as num::traits::FloatConst>::FRAC_2_PI().sqrt(),
    )
    .unwrap();

    let m = op.ncols();
    let rel_tol = num::cast::<f64, <Op::A as Scalar>::Real>(rel_tol).unwrap();
    let omega = Op::A::random_gaussian((m, sample_size), rng);
    let mut op_omega = op.matmat(omega.view());
    // Randomized estimate of the original operator norm.
    let operator_norm = max_col_norm(&op_omega) * tol_factor;
    let mut max_norm = operator_norm;
    let mut q = Array2::<Op::A>::zeros((op.nrows(), 0));
    let mut b = Array2::<Op::A>::zeros((0, op.ncols()));

    let mut residuals = Vec::<(usize, f64)>::new();

    while max_norm / operator_norm >= rel_tol {
        // Orthogonalize against existing basis
        if q.ncols() > 0 {
            op_omega -= &q.dot(&q.t().map(|item| item.conj()).dot(&op_omega));
        }
        // Now do the QR of the vectors
        let (qtemp, _) = op_omega.qr().unwrap();
        // Extend the b matrix
        b = concatenate![Axis(0), b, op.conj_row_matmat(qtemp.view())];
        // Extend the Q matrix
        q = concatenate![Axis(1), q, qtemp];

        // Now compute new vectors
        let omega = Op::A::random_gaussian((m, sample_size), rng);
        op_omega = op.matmat(omega.view()) - q.dot(&b.dot(&omega));

        // Update the error tolerance
        max_norm = max_col_norm(&op_omega) * tol_factor;
        residuals.push((q.ncols(), (max_norm / operator_norm).to_f64().unwrap()));
    }

    Ok((q, residuals))
}

// Compute a QR decomposition via adaptive random range sampling.
// # Arguments
// * `op`: The operator for which to compute the QR decomposition.
// * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
// * `sample_size` Number of samples drawn together in each iteration.
// * `rng`: The random number generator.
pub fn randomized_adaptive_qr<Op: ConjRowMatMat, R: Rng>(
    op: &Op,
    rel_tol: f64,
    sample_size: usize,
    rng: &mut R,
) -> Result<QRContainer<Op::A>> {
    let (q, _) = sample_range_adaptive(op, rel_tol, sample_size, rng)?;

    let b = op.conj_row_matmat(q.view());
    let mut qr = b.pivoted_qr()?;

    qr.q = b.dot(&qr.q);

    Ok(qr)
}

// Compute a SVD decomposition via adaptive random range sampling.
// # Arguments
// * `op`: The operator for which to compute the QR decomposition.
// * `rel_tol`: The relative error tolerance. The error is checked probabilistically.
// * `sample_size` Number of samples drawn together in each iteration.
// * `rng`: The random number generator.
pub fn randomized_adaptive_svd<Op: ConjRowMatMat, R: Rng>(
    op: &Op,
    rel_tol: f64,
    sample_size: usize,
    rng: &mut R,
) -> Result<SVDContainer<Op::A>> {
    let (q, _) = sample_range_adaptive(op, rel_tol, sample_size, rng)?;

    let b = op.conj_row_matmat(q.view());
    let mut svd = b.compute_svd()?;

    svd.u = b.dot(&svd.u);

    Ok(svd)
}
