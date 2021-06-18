pub mod compute_svd;
pub mod helpers;
pub mod permutation;
pub mod pivoted_qr;
pub mod pivoted_lq;
pub mod random;
//pub mod svd_compression;
pub mod col_interp_decomp;
pub mod prelude;
pub mod qr_container;
pub mod lq_container;
pub mod svd_container;

type Result<T> = std::result::Result<T, &'static str>;

//pub use compute_svd::ComputeSVD;
//pub use svd_compression::CompressSVD;
