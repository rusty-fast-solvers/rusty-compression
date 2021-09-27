//! Collect all traits and other exports here.

pub use crate::compute_svd::ComputeSVD;
pub use crate::helpers::*;
pub use crate::lq_container::LQContainer;
pub use crate::permutation::*;
pub use crate::pivoted_lq::PivotedLQ;
pub use crate::pivoted_qr::HasPivotedQR;
pub use crate::pivoted_qr::PivotedQR;
pub use crate::qr_container::QRContainer;
pub use crate::random_matrix::RandomMatrix;
pub use crate::random_sampling::*;
pub use crate::svd_container::SVDContainer;

pub trait ScalarType: HasPivotedQR + RandomMatrix {}

impl<A: HasPivotedQR + RandomMatrix> ScalarType for A {}
