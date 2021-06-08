//! Traits and functions for permutation vectors.

use crate::prelude::ScalarType;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

pub enum MatrixPermutationMode {
    COL,
    ROW,
    COLTRANS,
    ROWTRANS,
}

pub enum VectorPermutationMode {
    TRANS,
    NOTRANS,
}

pub fn invert_permutation_vector<S: Data<Elem = usize>>(&perm: ArrayBase<S, Ix1>) ->  {
    let n = perm.len();

    let inverse = Array1::<usize>::zeros(n);

    for (index, &elem) in perm.iter().enumerate() {
        inverse[elem] = index
    }

    inverse
}

pub trait ApplyPermutationToMatrix {
    type A;

    fn apply_permutation(
        self,
        index_array: Array1<usize>,
        mode: MatrixPermutationMode,
    ) -> Array2<Self::A>;
}

impl<A, S> ApplyPermutationToMatrix for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;

    fn apply_permutation(
        self,
        index_array: Array1<usize>,
        mode: MatrixPermutationMode,
    ) -> Array2<Self::A> {
        let m = self.nrows();
        let n = self.ncols();

        let permuted = Array2::<A>::zeros((m, n));

        match mode {
            COL => for (index, col) in self.axis_iter(Axis(1)).enumerate() {
                permuted.index_axis_mut(Axis(1), permuted[index]).assign(&col) }
            ROW => for (index, row) in self.axis_iter(Axis(0)).enumerate() {
                permuted.index_axis_mut(Axis(0), permuted[index]).assign(&row) }
            COLTRANS => {
                let inverse = invert_permutation_vector(&perm);
                for (index, col) in self.axis_iter(Axis(1)).enumerate() {
                permuted.index_axis_mut(Axis(1), inverse[index]).assign(&col) }}
            ROWTRANS => {
                let inverse = invert_permutation_vector(&perm);
                for (index, row) in self.axis_iter(Axis(0)).enumerate() {
                permuted.index_axis_mut(Axis(0), inverse[index]).assign(&row) }}
            };

        permuted
    }
}
