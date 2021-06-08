//! This module collects the various traits definitions


//pub trait ArrayProvider<T>
//where
//    T: Scalar + Lapack,
//{
//    fn array_view(&self) -> ArrayView2<T>;
//
//    /// Compute the relative distance
//    fn rel_diff<A: ArrayProvider<T>>(&self, other: A) -> T::Real {
//        use ndarray_linalg::OperationNorm;
//
//        let diff = self.array_view().to_owned() - other.array_view();
//
//        diff.opnorm_fro().unwrap() / other.array_view().opnorm_fro().unwrap()
//    }
//}



