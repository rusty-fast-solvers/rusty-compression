//! This module collects the various traits definitions

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::Norm;
use ndarray_linalg::OperationNorm;
use thiserror::Error;

pub use ndarray_linalg::{c32, c64, Scalar};

#[derive(Error, Debug)]
pub enum RustyCompressionError {
    #[error("Lapack Error")]
    LinalgError(LinalgError),
    #[error("Could not compress to desired tolerance")]
    CompressionError,
    #[error("Incompatible memory layout")]
    LayoutError,
    #[error("Pivoted QR failed")]
    PivotedQRError,
}

pub type Result<T> = std::result::Result<T, RustyCompressionError>;

pub trait Apply<A, Lhs> {
    type Output;

    fn dot(&self, lhs: &Lhs) -> Self::Output;
}

pub trait RApply<A, Lhs> {
    type Output;

    fn dot(&self, lhs: &Lhs) -> Self::Output;
}

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
        let mut output = Array2::<Self::A>::zeros((self.ncols(), mat.ncols()));

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

// impl<A, S> MatMat for ArrayBase<S, Ix2>
// where
//     A: Scalar,
//     S: Data<Elem = A>,
// {
//     fn matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
//         self.dot(&mat)
//     }
// }

impl<A: Scalar, T: MatVec<A=A>> MatMat for T {}
impl<A: Scalar, T: ConjMatVec<A=A>> ConjMatMat for T {}

// impl<A, S> ConjMatMat for ArrayBase<S, Ix2>
// where
//     A: Scalar,
//     S: Data<Elem = A>,
// {
//     fn conj_matmat(&self, mat: ArrayView2<Self::A>) -> Array2<Self::A> {
//         mat.t()
//             .map(|item| item.conj())
//             .dot(self)
//             .t()
//             .map(|item| item.conj())
//     }
// }

pub trait RelDiff {
    type A: Scalar;

    /// Return the relative Frobenius norm difference of `first` and `second`.
    fn rel_diff_fro(
        first: ArrayView2<Self::A>,
        second: ArrayView2<Self::A>,
    ) -> <<Self as RelDiff>::A as Scalar>::Real;

    /// Return the relative l2 vector norm difference of `first` and `second`.
    fn rel_diff_l2(
        first: ArrayView1<Self::A>,
        second: ArrayView1<Self::A>,
    ) -> <<Self as RelDiff>::A as Scalar>::Real;
}

macro_rules! rel_diff_impl {
    ($scalar:ty) => {
        impl RelDiff for $scalar {
            type A = $scalar;
            fn rel_diff_fro(
                first: ArrayView2<Self::A>,
                second: ArrayView2<Self::A>,
            ) -> <<Self as RelDiff>::A as Scalar>::Real {
                let diff = first.to_owned() - &second;
                diff.opnorm_fro().unwrap() / second.opnorm_fro().unwrap()
            }

            fn rel_diff_l2(
                first: ArrayView1<Self::A>,
                second: ArrayView1<Self::A>,
            ) -> <<Self as RelDiff>::A as Scalar>::Real {
                let diff = first.to_owned() - &second;
                diff.norm_l2() / second.norm_l2()
            }
        }
    };
}

rel_diff_impl!(f32);
rel_diff_impl!(f64);
rel_diff_impl!(c32);
rel_diff_impl!(c64);
