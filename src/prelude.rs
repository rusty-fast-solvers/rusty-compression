//! Collect all traits and other exports here.

pub use crate::svd_container::SVDContainer;
pub use crate::qr_container::QRContainer;
pub use crate::pivoted_qr::PivotedQR;
pub use crate::pivoted_qr::HasPivotedQR;
pub use crate::compute_svd::ComputeSVD;
//pub use crate::svd_compression::CompressSVD;
pub use crate::random::Random;


pub enum CompressionType {
    /// Adaptive compression with a specified tolerance
    ADAPTIVE(f64),
    /// Rank based compression with specified rank
    RANK(usize),
}

pub trait ScalarType: HasPivotedQR {}

impl<A: HasPivotedQR> ScalarType for A {}

