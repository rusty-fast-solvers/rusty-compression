pub mod permutation;
pub mod random;
pub mod traits;
pub mod pivoted_qr;
pub mod compute_svd;
//pub mod svd_compression;
pub mod prelude;
pub mod svd_container;
pub mod qr_container;



//pub mod interp_decomp;


type Result<T> = std::result::Result<T, &'static str>;

//pub use compute_svd::ComputeSVD;
//pub use svd_compression::CompressSVD;
