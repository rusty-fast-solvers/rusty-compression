// Adaptive sampling of a low-rank matrix.

// use ndarray::s;
// use ndarray_linalg::OperationNorm;
// use plotters::prelude::*;
// use rusty_compression::prelude::*;

pub fn main() {}

// pub fn main() {
//     let dimension = (500, 200);

//     let mut rng = rand::thread_rng();

//     let mat = f64::random_approximate_low_rank_matrix(dimension, 1.0, 1E-10, &mut rng);

//     let (q, res) = sample_range_adaptive(&mat, 1E-5, 5, &mut rng).unwrap();

//     let root = BitMapBackend::new("residuals.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE).unwrap();
//     let mut chart = ChartBuilder::on(&root)
//         .margin(20)
//         .x_label_area_size(20)
//         .y_label_area_size(50)
//         .build_cartesian_2d(1..(1 + res.last().unwrap().0), (1E-7..1.0).log_scale())
//         .unwrap();

//     chart
//         .configure_mesh()
//         .x_labels(10)
//         .y_labels(10)
//         .y_label_formatter(&|item| format!("{:.1E}", item))
//         .y_desc("Relative Residual")
//         .draw()
//         .unwrap();
//     chart
//         .draw_series(LineSeries::new(res, &BLACK))
//         .unwrap()
//         .label("estimated residual")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

//     chart
//         .draw_series(LineSeries::new(
//             (0..q.ncols()).step_by(5).map(|index| {
//                 let qtemp = q.slice(s![.., 0..index]);
//                 let diff = &mat - &qtemp.dot(&qtemp.t().dot(&mat));
//                 (
//                     index,
//                     diff.opnorm_fro().unwrap() / mat.opnorm_fro().unwrap(),
//                 )
//             }),
//             &RED,
//         ))
//         .unwrap()
//         .label("exact residual")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

//     chart.configure_series_labels().draw().unwrap();

//     println!("Rank: {}", q.ncols());
// }
