/// Measure image sharpness based on sobel
///
use opencv::{
    core::{self, norm, Mat, NormTypes, BORDER_DEFAULT, CV_32F},
    highgui, imgcodecs,
    imgproc::{self, sobel},
    prelude::*,
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

pub fn laplacian_sharpness(src: &Mat) -> Result<Mat, ProcessError> {
    let mut norm_image = Mat::default();
    let mut gray_image = Mat::default();
    let mut gx = Mat::default();
    let mut gy = Mat::default();
    let size = src.size()?;

    core::normalize(
        &src,
        &mut norm_image,
        0.0,
        255.0,
        core::NORM_MINMAX,
        -1,
        &core::no_array(),
    )?;

    imgproc::cvt_color(
        &norm_image.clone(),
        &mut gray_image,
        imgproc::COLOR_BGR2GRAY,
        0,
    )?;

    let center_rec = core::Rect {
        x: (size.width / 2 - 100) as i32,
        y: (size.height / 2 - 100) as i32,
        width: 200,
        height: 200,
    };

    let cropped_image = Mat::roi(&gray_image, center_rec)?;

    sobel(
        &cropped_image,
        &mut gx,
        CV_32F,
        1,
        0,
        3,
        1.0,
        0.0,
        BORDER_DEFAULT,
    )?;

    sobel(
        &cropped_image,
        &mut gy,
        CV_32F,
        0,
        1,
        3,
        1.0,
        0.0,
        BORDER_DEFAULT,
    )?;

    let norm_gx = norm(&gx, NormTypes::NORM_L2 as i32, &Mat::default())?;
    let norm_gy = norm(&gy, NormTypes::NORM_L2 as i32, &Mat::default())?;
    let sum_sq = norm_gx * norm_gx + norm_gy * norm_gy;

    let sharpness = (1.0 / (sum_sq / (src.size().unwrap().area() as f64) + 1e-6) as f32) * 1000.0;

    println!("Sobel Sharpness: {sharpness}");

    Ok(cropped_image)
}

fn main() -> opencv::Result<(), ProcessError> {
    let args = Args::parse();
    let src = imgcodecs::imread(&args.image, imgcodecs::IMREAD_COLOR)?;

    match laplacian_sharpness(&src) {
        Ok(image) => {
            highgui::imshow("Center Area", &image)?;
            highgui::wait_key(0)?;
        }
        Err(e) => println!("Sharpness error: {e:?}"),
    };

    Ok(())
}
