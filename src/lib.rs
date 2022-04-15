//! This library provides routines to compute low-rank approximations of matrices.
//! Let $A\in\mathbb{C}^{m\times n}$ be a given matrix. A low-rank approximation is
//! a representation of the form
//! $$
//! A\approx UV^H
//! $$
//! with $U\in\mathbb{C}^{m\tmes k}$ and $V\in\mathbb{C}{n\times k}$, where $k$ is
//! sufficiently small for the required application. A frequently used
//! representation is also
//! $$
//! A\approx \tilde{U}X\tilde{V}^H
//! $$
//! with a $k\times k$ matrix $X$.
//! 
//! The basis of the library is a pivoted QR decomposition of $A$ of the form
//! $$
//! AP = QR
//! $$
//! with $P$ a permutation matrix, $Q\in\mathbb{C}^{m\times \ell}$ a matrix
//! with orthogonal columns of unit norm, $R\in\mathbb{C}^{\ell\times n}$
//! upper triangular, and $\ell=\min\{m, n\}$.
//! 
//! In a pivoted QR decomposition the diagonal elements of $R$
//! are sorted in non-increasing order, that is $|r_{11}|\geq |r_{22}|\geq \dots$. 
//! We can therefore compress $A$ by choosing in index $k$ and only keeping the first
//! $k$ columns of $Q$ and the first $k$ rows of $R$. The library allows to choose the
//! parameter $k$ directly or by a given tolerance $\tau$ such that 
//! $\langle|\frac{r_{kk}}{r_{11}}\rangle|< \tau$.
//! 
//! Alternatively, one can also low-rank compress by first computing the Singular Value
//! Decomposition. This is more accurate but also more costly.
//! 
//! From the compressed QR decomposition we can compute a column pivoted interpolative
//! decomposition of the form
//! $$
//! A \approx CZ
//! $$
//! The matrix $C$ is defined as
//! $$
//! C = A[:, [i_1, i_2, \dots]],
//! $$
//! where $i_1, i_2, \dots$ are column indices. 
//! The column pivoted interpolative decomposition has
//! the advantage that we are using columns of $A$ as low-rank basis, which in
//! applications often have physical meaning as opposed to the columns of $Q$.
//! 
//! From the column pivoted decomposition we can also compute a two-sided
//! decomposition of the form
//! $$
//! A\approx C X R
//! $$
//! Here, $X$ is a $k\times k$ matrix, which is the submatrix of $A$ formed of the column indices
//! $i_1, i_2, \dots$ and row indices $j_1, j_2, \dots$.
//! 


pub mod col_interp_decomp;
pub mod compute_svd;
pub mod types;
pub mod permutation;

pub mod random_matrix;
pub mod random_sampling;
pub mod row_interp_decomp;
pub mod svd;
pub mod two_sided_interp_decomp;

pub(crate) mod pivoted_qr;
pub mod qr;

pub enum CompressionType {
    /// Adaptive compression with a specified tolerance
    ADAPTIVE(f64),
    /// Rank based compression with specified rank
    RANK(usize),
}


pub use qr::{QR, LQ, QRTraits, LQTraits};
pub use col_interp_decomp::{ColumnID, ColumnIDTraits};
pub use row_interp_decomp::{RowID, RowIDTraits};
pub use two_sided_interp_decomp::{TwoSidedID, TwoSidedIDTraits};
pub use svd::{SVD, SVDTraits};
pub use random_matrix::RandomMatrix;
pub use permutation::*;
pub use types::RelDiff;
pub use random_sampling::*;

pub use types::{c32, c64, Scalar};

pub use types::Result;