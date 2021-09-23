//! Traits and functions for permutation vectors.

use crate::prelude::ScalarType;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2};

pub enum MatrixPermutationMode {
    COL,
    ROW,
    COLINV,
    ROWINV,
}

pub enum VectorPermutationMode {
    INV,
    NOINV,
}

pub fn invert_permutation_vector<S: Data<Elem = usize>>(perm: &ArrayBase<S, Ix1>) -> Array1<usize> {
    let n = perm.len();

    let mut inverse = Array1::<usize>::zeros(n);

    for (index, &elem) in perm.iter().enumerate() {
        inverse[elem] = index
    }

    inverse
}

pub trait ApplyPermutationToMatrix {
    type A;

    /// Apply a permutation to rows or columns of a matrix
    ///
    /// # Arguments
    /// * `index_array` : A permutation array. If index_array[i] = j then after
    ///                   permutation the ith row/column of the permuted matrix
    ///                   will contain the jth row/column of the original matrix.
    /// * `mode` : The permutation mode. If the permutation mode is `ROW` or `COL` then
    ///            permute the rows/columns of the matrix. If the permutation mode is `ROWINV` or
    ///            `COLINV` then apply the inverse permutation to the rows/columns.
    fn apply_permutation(
        &self,
        index_array: ArrayView1<usize>,
        mode: MatrixPermutationMode,
    ) -> Array2<Self::A>;
}

pub trait ApplyPermutationToVector {
    type A;

    fn apply_permutation(
        &self,
        index_array: ArrayView1<usize>,
        mode: VectorPermutationMode,
    ) -> Array1<Self::A>;
}

impl<A, S> ApplyPermutationToMatrix for ArrayBase<S, Ix2>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;

    fn apply_permutation(
        &self,
        index_array: ArrayView1<usize>,
        mode: MatrixPermutationMode,
    ) -> Array2<Self::A> {
        let m = self.nrows();
        let n = self.ncols();

        let mut permuted = Array2::<A>::zeros((m, n));

        match mode {
            MatrixPermutationMode::COL => {
                assert!(
                    index_array.len() == n,
                    "Length of index array and number of columns differ."
                );
                for index in 0..n {
                    permuted
                        .index_axis_mut(Axis(1), index)
                        .assign(&self.index_axis(Axis(1), index_array[index]));
                }
            }
            MatrixPermutationMode::ROW => {
                assert!(
                    index_array.len() == m,
                    "Length of index array and number of rows differ."
                );
                for index in 0..m {
                    permuted
                        .index_axis_mut(Axis(0), index)
                        .assign(&self.index_axis(Axis(0), index_array[index]));
                }
            }
            MatrixPermutationMode::COLINV => {
                assert!(
                    index_array.len() == n,
                    "Length of index array and number of columns differ."
                );
                let inverse = invert_permutation_vector(&index_array);
                for index in 0..n {
                    permuted
                        .index_axis_mut(Axis(1), index)
                        .assign(&self.index_axis(Axis(1), inverse[index]));
                }
            }
            MatrixPermutationMode::ROWINV => {
                assert!(
                    index_array.len() == m,
                    "Length of index array and number of rows differ."
                );
                let inverse = invert_permutation_vector(&index_array);
                for index in 0..m {
                    permuted
                        .index_axis_mut(Axis(0), index)
                        .assign(&self.index_axis(Axis(0), inverse[index]));
                }
            }
        };

        permuted
    }
}

impl<A, S> ApplyPermutationToVector for ArrayBase<S, Ix1>
where
    A: ScalarType,
    S: Data<Elem = A>,
{
    type A = A;

    fn apply_permutation(
        &self,
        index_array: ArrayView1<usize>,
        mode: VectorPermutationMode,
    ) -> Array1<Self::A> {
        let n = self.len();

        assert!(
            index_array.len() == n,
            "The input vector and the index array must have the same length"
        );

        let mut permutation = Array1::<Self::A>::zeros(n);

        match mode {
            VectorPermutationMode::INV => {
                let inverse = invert_permutation_vector(&index_array);
                for index in 0..n {
                    permutation[index] = self[inverse[index]];
                }
            }
            VectorPermutationMode::NOINV => {
                for index in 0..n {
                    permutation[index] = self[index_array[index]];
                }
            }
        }

        permutation
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::{arr1, arr2, Array1};

    #[test]
    fn test_matrix_permutation() {
        let mat = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let mat_right_row_shift = arr2(&[[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let mat_left_row_shift = arr2(&[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]);

        let mat_right_col_shift = arr2(&[[3.0, 1.0, 2.0], [6.0, 4.0, 5.0], [9.0, 7.0, 8.0]]);

        let mat_left_col_shift = arr2(&[[2.0, 3.0, 1.0], [5.0, 6.0, 4.0], [8.0, 9.0, 7.0]]);

        let mut perm = Array1::<usize>::zeros(3);
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;

        assert!(
            mat_right_col_shift == mat.apply_permutation(perm.view(), MatrixPermutationMode::COL)
        );
        assert!(
            mat_left_col_shift == mat.apply_permutation(perm.view(), MatrixPermutationMode::COLINV)
        );
        assert!(
            mat_right_row_shift == mat.apply_permutation(perm.view(), MatrixPermutationMode::ROW)
        );
        assert!(
            mat_left_row_shift == mat.apply_permutation(perm.view(), MatrixPermutationMode::ROWINV)
        );
    }

    #[test]
    fn test_vector_permutaiton() {
        let vec = arr1(&[1.0, 2.0, 3.0]);

        let mut perm = Array1::<usize>::zeros(3);
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;

        let vec_right_shift = arr1(&[3.0, 1.0, 2.0]);
        let vec_left_shift = arr1(&[2.0, 3.0, 1.0]);

        assert!(
            vec_right_shift == vec.apply_permutation(perm.view(), VectorPermutationMode::NOINV)
        );
        assert!(vec_left_shift == vec.apply_permutation(perm.view(), VectorPermutationMode::INV));
    }
}
