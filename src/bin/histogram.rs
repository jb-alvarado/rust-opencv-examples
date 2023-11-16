/// Create histogram and paint it
///
extern crate opencv;
use opencv::{
    core::{self, Mat, Scalar, CV_8UC3},
    highgui, imgcodecs, imgproc,
    prelude::*,
    types::{VectorOfMat, VectorOfi32},
};

use rust_opencv_examples::utils::errors::ProcessError;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[clap(version,
    about = "Show Histogram",
    long_about = None)]
pub struct Args {
    #[clap(short, long, help = "Path to input image.")]
    pub image: String,
}

fn calculate_histogram(
    image: &Mat,
    hist_size: i32,
    hist_range: &core::Vector<f32>,
    accumulate: bool,
) -> opencv::Result<Mat> {
    let mut hist = Mat::default();
    let channels = vec![0].into();
    imgproc::calc_hist(
        &VectorOfMat::from(vec![image.clone()]),
        &channels,
        &Mat::default(),
        &mut hist,
        &VectorOfi32::from(vec![hist_size]),
        hist_range,
        accumulate,
    )?;
    Ok(hist)
}

fn normalize_histogram(hist: &Mat, rows: f64) -> Mat {
    let mut hist_norm = Mat::default();
    core::normalize(
        &hist,
        &mut hist_norm,
        0.0,
        rows,
        core::NORM_MINMAX,
        -1,
        &Mat::default(),
    )
    .unwrap();
    hist_norm
}

fn draw_histograms(
    b_hist: &Mat,
    g_hist: &Mat,
    r_hist: &Mat,
    hist_size: i32,
) -> Result<Mat, ProcessError> {
    let hist_w = 512;
    let hist_h = 400;
    let bin_w = hist_w / hist_size;

    let mut hist_image =
        Mat::new_rows_cols_with_default(hist_h, hist_w, CV_8UC3, Scalar::all(0.0))?;

    let b_norm = normalize_histogram(b_hist, hist_image.rows().into());
    let g_norm = normalize_histogram(g_hist, hist_image.rows().into());
    let r_norm = normalize_histogram(r_hist, hist_image.rows().into());

    for i in 1..hist_size {
        let mut draw_line = |hist_norm: &Mat, color: Scalar| {
            imgproc::line(
                &mut hist_image,
                opencv::core::Point {
                    x: bin_w * (i - 1),
                    y: hist_h - *hist_norm.at::<f32>(i - 1).unwrap() as i32,
                },
                opencv::core::Point {
                    x: bin_w * i,
                    y: hist_h - *hist_norm.at::<f32>(i).unwrap() as i32,
                },
                color,
                2,
                8,
                0,
            )
            .unwrap();
        };

        draw_line(&b_norm, Scalar::from((255., 0., 0.)));
        draw_line(&g_norm, Scalar::from((0., 255., 0.)));
        draw_line(&r_norm, Scalar::from((0., 0., 255.)));
    }

    Ok(hist_image)
}

fn main() -> opencv::Result<(), ProcessError> {
    let args = Args::parse();

    let src = imgcodecs::imread(&args.image, imgcodecs::IMREAD_COLOR)?;

    let mut bgr_planes = VectorOfMat::new();
    core::split(&src, &mut bgr_planes)?;

    // Establish the number of bins
    let hist_size = 256;
    let hist_range: core::Vector<f32> = vec![0., 256.].into();
    let accumulate = false;

    let b_hist = calculate_histogram(&bgr_planes.get(0)?, hist_size, &hist_range, accumulate)?;
    let g_hist = calculate_histogram(&bgr_planes.get(1)?, hist_size, &hist_range, accumulate)?;
    let r_hist = calculate_histogram(&bgr_planes.get(2)?, hist_size, &hist_range, accumulate)?;

    let hist_image = draw_histograms(&b_hist, &g_hist, &r_hist, hist_size)?;

    // Display images
    highgui::imshow("Histogram", &hist_image)?;
    highgui::imshow("Source image", &src)?;
    highgui::wait_key(0)?;

    Ok(())
}
