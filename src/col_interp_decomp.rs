//! Data structure for Column Interpolative Decomposition

use crate::helpers::Apply;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Data, Ix1, Ix2,
};
use rusty_base::types::{c32, c64, Scalar};

pub struct ColumnIDData<A: Scalar> {
    c: Array2<A>,
    z: Array2<A>,
    col_ind: Array1<usize>,
}

pub trait ColumnID {
    type A: Scalar;

    fn nrows(&self) -> usize {
        self.get_c().nrows()
    }

    fn ncols(&self) -> usize {
        self.get_z().ncols()
    }

    fn rank(&self) -> usize {
        self.get_c().ncols()
    }

    fn to_mat(&self) -> Array2<Self::A> {
        self.get_c().dot(&self.get_z())
    }

    fn get_c(&self) -> ArrayView2<Self::A>;
    fn get_z(&self) -> ArrayView2<Self::A>;
    fn get_col_ind(&self) -> ArrayView1<usize>;

    fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_z_mut(&mut self) -> ArrayViewMut2<Self::A>;
    fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize>;

    fn new(c: Array2<Self::A>, z: Array2<Self::A>, col_ind: Array1<usize>) -> Self;
}

macro_rules! impl_col_id {
    ($scalar:ty) => {
        impl ColumnID for ColumnIDData<$scalar> {
            type A = $scalar;
            fn get_c(&self) -> ArrayView2<Self::A> {
                self.c.view()
            }
            fn get_z(&self) -> ArrayView2<Self::A> {
                self.z.view()
            }

            fn get_col_ind(&self) -> ArrayView1<usize> {
                self.col_ind.view()
            }

            fn get_c_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.c.view_mut()
            }
            fn get_z_mut(&mut self) -> ArrayViewMut2<Self::A> {
                self.z.view_mut()
            }
            fn get_col_ind_mut(&mut self) -> ArrayViewMut1<usize> {
                self.col_ind.view_mut()
            }

            fn new(c: Array2<Self::A>, z: Array2<Self::A>, col_ind: Array1<usize>) -> Self {
                ColumnIDData::<$scalar> { c, z, col_ind }
            }
        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix1>> for ColumnIDData<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array1<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix1>) -> Self::Output {
                self.c.dot(&self.z.dot(rhs))
            }
        }

        impl<S> Apply<$scalar, ArrayBase<S, Ix2>> for ColumnIDData<$scalar>
        where
            S: Data<Elem = $scalar>,
        {
            type Output = Array2<$scalar>;

            fn dot(&self, rhs: &ArrayBase<S, Ix2>) -> Self::Output {
                self.c.dot(&self.z.dot(rhs))
            }
        }
    };
}

impl_col_id!(f32);
impl_col_id!(f64);
impl_col_id!(c32);
impl_col_id!(c64);
