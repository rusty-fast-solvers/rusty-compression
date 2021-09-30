pub mod col_interp_decomp;
pub mod compute_svd;
pub mod helpers;
// pub mod lq_container;
pub mod permutation;
// pub mod pivoted_lq;
// pub mod pivoted_qr;
// pub mod prelude;
pub mod random_matrix;
// pub mod random_sampling;
// pub mod row_interp_decomp;
pub mod svd;
// pub mod two_sided_interp_decomp;

// pub use rusty_base::Result;

pub(crate) mod pivoted_qr;
pub mod qr;

pub enum CompressionType {
    /// Adaptive compression with a specified tolerance
    ADAPTIVE(f64),
    /// Rank based compression with specified rank
    RANK(usize),
}
