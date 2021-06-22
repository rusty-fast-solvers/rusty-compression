use lapack;
use ndarray;
use ndarray::{s, Array1, Array2, ArrayView2, ShapeBuilder};
use ndarray_linalg::layout::AllocatedArray;
use ndarray_linalg::IntoTriangular;
use ndarray_linalg::Norm;
use num::traits::ToPrimitive;
use rusty_compression::prelude::RandomMatrix;

pub fn pivoted_qr(mat: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>, Array1<usize>) {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);

    // Create a new array in Fortran order
    let mut mat_fortran = ndarray::Array2::<f64>::zeros((m, n).f());

    // Assign the mat array to it.

    mat_fortran.assign(&mat);

    let layout = mat_fortran.layout().unwrap();

    match layout {
        ndarray_linalg::MatrixLayout::C { row, lda } => println!("{} {} {}", "C", row, lda),
        ndarray_linalg::MatrixLayout::F { col, lda } => println!("{} {} {}", "F", col, lda),
    };

    let result = pivoted_qr_decomp(mat_fortran.as_slice_memory_order_mut().unwrap(), layout);
    let (mut tau, jpvt) = result.unwrap();

    let mut r_mat = ndarray::Array2::<f64>::zeros((k, n));
    r_mat.assign(&mat_fortran.slice(s![0..k, ..]));
    let r_mat = r_mat.into_triangular(ndarray_linalg::UPLO::Upper);

    lax::QR_::q(
        layout,
        mat_fortran.as_slice_memory_order_mut().unwrap(),
        tau.as_slice_memory_order_mut().unwrap(),
    )
    .expect("Pivoted QR computation failed.");

    let mut q_mat = ndarray::Array2::<f64>::zeros((m as usize, k as usize));
    q_mat.assign(&mat_fortran.slice(s![.., 0..k]));

    // Finally, return the QR decomposition.

    (q_mat, r_mat, jpvt.map(|item| (item - 1) as usize))
}

fn pivoted_qr_decomp(
    mat: &mut [f64],
    layout: ndarray_linalg::MatrixLayout,
) -> Result<(ndarray::Array1<f64>, ndarray::Array1<i32>), i32> {
    let m = layout.lda();
    let n = layout.len();
    let k = m.min(n);
    let mut tau = ndarray::Array1::<f64>::zeros(k as usize);

    let mut info = 0;
    let mut work_size = [0.0];
    let mut jpvt = ndarray::Array1::<i32>::zeros(n as usize);

    unsafe {
        lapack::dgeqp3(
            m,
            n,
            mat,
            m,
            jpvt.as_slice_memory_order_mut().unwrap(),
            tau.as_slice_memory_order_mut().unwrap(),
            &mut work_size,
            -1,
            &mut info,
        );
    }

    match info {
        0 => (),
        _ => return Err(info),
    }

    let lwork = work_size[0].to_usize().unwrap();
    let mut work = ndarray::Array1::<f64>::zeros(lwork);
    unsafe {
        lapack::dgeqp3(
            m,
            n,
            mat,
            m,
            jpvt.as_slice_memory_order_mut().unwrap(),
            tau.as_slice_memory_order_mut().unwrap(),
            work.as_slice_memory_order_mut().unwrap(),
            lwork as i32,
            &mut info,
        );
    }

    match info {
        0 => Ok((tau, jpvt)),
        _ => Err(info),
    }
}

pub fn main() {
    let m = 1000;
    let n = 1000;

    let mut rng = rand::thread_rng();
    let mat = f64::random_approximate_low_rank_matrix((m, n), 1.0, 1E-10, &mut rng);

    let (q, r, indices) = pivoted_qr(mat.view());

    println!("Q shape: {} x {}", q.nrows(), q.ncols());
    println!("R shape: {} x {}", r.nrows(), r.ncols());

    let prod = q.dot(&r);

    // Check orthogonality of Q.T x Q

    let qtq = q.t().dot(&q);

    for ((i, j), &val) in qtq.indexed_iter() {
        if i == j {
            let rel_diff = (val - 1.0).abs();
            assert!(rel_diff < 1E-10);
        } else {
            assert!(val.abs() < 1E-10);
        }
    }

    // Check that the product is correct.

    for (col_index, col) in prod.axis_iter(ndarray::Axis(1)).enumerate() {
        let perm_index = indices[col_index];
        let diff = col.to_owned() - mat.index_axis(ndarray::Axis(1), perm_index);
        let rel_diff = diff.norm_l2() / mat.index_axis(ndarray::Axis(1), perm_index).norm_l2();

        assert!(rel_diff < 1E-10);
    }

    println!("Success");
}
