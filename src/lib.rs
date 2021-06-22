pub mod compute_svd;
pub mod helpers;
pub mod permutation;
pub mod pivoted_qr;
pub mod pivoted_lq;
pub mod random_matrix;
pub mod col_interp_decomp;
pub mod row_interp_decomp;
pub mod two_sided_interp_decomp;
pub mod prelude;
pub mod qr_container;
pub mod lq_container;
pub mod svd_container;
pub mod random_sampling;

type Result<T> = std::result::Result<T, &'static str>;

//pub use compute_svd::ComputeSVD;
//pub use svd_compression::CompressSVD;
