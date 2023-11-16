/// Measure image sharpness based on laplacian
///
use opencv::{
    core::{self, Mat},
    highgui, imgcodecs, imgproc,
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
    let mut lap_image = Mat::default();
    let mut mean = core::Scalar::default();
    let mut stddev = core::Scalar::default();

    // try to get a sharp image
    imgproc::laplacian(&cropped_image, &mut lap_image, -1, 5, 1.0, 0.0, 0)?;
    core::mean_std_dev(&lap_image, &mut mean, &mut stddev, &core::no_array())?;

    println!("Laplacian Sharpness: {}", stddev[0]);

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
