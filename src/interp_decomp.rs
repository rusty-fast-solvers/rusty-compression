//! Implementation of the interpolative decomposition.

use crate::pivoted_qr::{HasPivotedQR, PivotedQR, PivotedQRResult};
use crate::CompressionType;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::{Lapack, Scalar};
use num::traits::ToPrimitive;

pub struct ColumnIDResult<A: HasPivotedQR> {
    pub c: Array2<A>,
    pub z: Array2<A>,
    pub col_ind: Array1<usize>,
}

pub struct RowIDResult<A: HasPivotedQR> {
    pub x: Array2<A>,
    pub row_ind: Array1<usize>,
}

pub struct TwoSidedIDResult<A: HasPivotedQR> {
    pub x: Array2<A>,
    pub row_ind: Array1<usize>,
    pub col_ind: Array1<usize>,
}

pub trait InterpolativeDecomp {
    type A: HasPivotedQR;

    fn column_id(
        compression_type: CompressionType,
    ) -> Result<ColumnIDResult<Self::A>, &'static str>;
    fn row_id(compression_type: CompressionType) -> Result<RowIDResult<Self::A>, &'static str>;
    fn two_sided_id(
        compression_type: CompressionType,
    ) -> Result<TwoSidedIDResult<Self::A>, &'static str>;
}

fn low_rank_from_pivoted_qr<A: HasPivotedQR>(
    mat: ArrayView2<A>,
    compression_type: CompressionType,
) -> Result<PivotedQRResult<A>, &'static str> {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);
    let mut qr_result = mat.pivoted_qr()?;

    let new_rank = match compression_type {
        CompressionType::RANK(rank) => rank.min(k),
        CompressionType::ADAPTIVE(tol) => {
            let diag = qr_result.r.diag();
            let pos = diag
                .iter()
                .position(|&item| (item.abs() / diag[0].abs()).to_f64().unwrap() < tol);
            match pos {
                Some(p) => p,
                None => return Err("Compression to desired tolerance not possible."),
            }
        }
    };

    qr_result.q = qr_result.q.slice_move(s![.., 0..new_rank]);
    qr_result.r = qr_result.r.slice_move(s![0..new_rank, ..]);
    qr_result.ind = qr_result.ind.slice_move(s![0..new_rank]);

    Ok(qr_result)
}

fn column_id_impl<A: HasPivotedQR>(
    mat: ArrayView2<A>,
    compression_type: CompressionType,
) -> Result<ColumnIDResult<A>, &'static str> {
    let qr_result = low_rank_from_pivoted_qr(mat, compression_type)?;

    let m = mat.nrows();
    let n = mat.ncols();
    let rank = qr_result.r.nrows();

    // Compute the permutation matrix.

    let pt = Array2::<A>::zeros((rank, rank));
    let z = Array2::<A>::zeros((rank, n));
    let c = Array2::<A>::zeros((m, rank));

    for (col_index, &row_value) in qr_result.ind.iter().enumerate() {
        pt[[col_index, row_value]] = num::traits::one();
    }

    z.slice_mut(s![0..rank, 0..rank]).assign(&pt);
    for (index, mut col) in c.axis_iter_mut(Axis(1)).enumerate() {
        col.assign(&mat.index_axis(Axis(1), qr_result.ind[index]));
    }

    if n == k {
        Ok(ColumnIDResult {
            z,
            col_ind: qr_result.ind,
        })
    } else {
    }
}
