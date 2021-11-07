pub mod col_interp_decomp;
pub mod compute_svd;
pub mod helpers;
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
pub use helpers::RelDiff;
pub use random_sampling::*;