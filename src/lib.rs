pub mod random;
pub mod traits;
pub mod pivoted_qr;
pub mod svd_compressor;
pub mod interpolative;

pub enum CompressionType {
    /// Adaptive compression with a specified tolerance
    ADAPTIVE(f64),
    /// Rank based compression with specified rank
    RANK(usize),
}

pub use traits::*;
pub use random::*;
pub use svd_compressor::*;
