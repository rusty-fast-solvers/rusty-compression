//! Adaptive sampling of a low-rank matrix.

use ndarray::s;
use ndarray_linalg::OperationNorm;
use plotters::prelude::*;
use rusty_compression::AdaptiveSampling;
use rusty_compression::QRTraits;
use rusty_compression::RandomMatrix;
use rusty_compression::RelDiff;
use rusty_compression::QR;

pub fn main() {

    // The dimension of the original matrix
    let dimension = (500, 200);

    // Set the relative tolerance for the range estimate
    let rel_tol = 1E-5;

    // Initialize the random number generator
    let mut rng = rand::thread_rng();

    // Generate a random matrix with singular values logarithmically spaced
    // between 1 and 1E-10,
    let mat = f64::random_approximate_low_rank_matrix(dimension, 1.0, 1E-10, &mut rng);

    // This command adaptively samples the range of the matrix and returns an orthogonal basis
    // of the dominant range in the variable `q`. In addition it returns a vector `res`, which
    // contains the convergence history of the range estimation. The residuals are computed
    // probabilistically.
    let (q, res) = mat.sample_range_adaptive(rel_tol, 5, &mut rng).unwrap();

    // The following code generates a plot that draws a convergence graph comparing the
    // probabilistic error bound with the actual error in each step. The result is saved
    // in `residuals.png`.
    let root = BitMapBackend::new("residuals.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(20)
        .y_label_area_size(50)
        .build_cartesian_2d(1..(1 + res.last().unwrap().0), (1E-7..1.0).log_scale())
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .y_label_formatter(&|item| format!("{:.1E}", item))
        .y_desc("Relative Residual")
        .draw()
        .unwrap();
    chart
        .draw_series(LineSeries::new(res, &BLACK))
        .unwrap()
        .label("estimated residual")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart
        .draw_series(LineSeries::new(
            (0..q.ncols()).step_by(5).map(|index| {
                let qtemp = q.slice(s![.., 0..index]);
                let diff = &mat - &qtemp.dot(&qtemp.t().dot(&mat));
                (
                    index,
                    diff.opnorm_fro().unwrap() / mat.opnorm_fro().unwrap(),
                )
            }),
            &RED,
        ))
        .unwrap()
        .label("exact residual")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw().unwrap();

    println!("Rank: {}", q.ncols());

    // At the end we compute an approximate QR decomposition of the original matrix
    // from the range estimate.
    let qr = QR::<f64>::compute_from_range_estimate(q.view(), &mat).unwrap();
    let mat_approx = qr.to_mat();

    // We now compute the relative distance of the approximate QR decomposition from
    // the original matrix.
    let rel_err = f64::rel_diff_fro(mat.view(), mat_approx.view());

    // Finally, we print the relative distance.
    println!(
        "Desired relative tolerance {:E}. Actual relative residual: {:1.2E}",
        rel_tol, rel_err
    );
}
