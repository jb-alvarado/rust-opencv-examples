/// Measure Colors and print colorfulness value
///
use opencv::{
    core::{self, Mat},
    imgcodecs,
    prelude::*,
    types::VectorOfMat,
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

fn image_colorfulness(channels: &core::Vector<Mat>) -> Result<f64, ProcessError> {
    let mut rg = Mat::default();
    let mut yb = Mat::default();
    let mut temp = Mat::default();
    let mut rg_mean = Mat::default();
    let mut rg_std = Mat::default();
    let mut yb_mean = Mat::default();
    let mut yb_std = Mat::default();

    core::subtract(
        &channels.get(2)?,
        &channels.get(1)?,
        &mut rg,
        &core::no_array(),
        -1,
    )?;

    core::add_weighted(
        &channels.get(2)?,
        0.5,
        &channels.get(1)?,
        0.5,
        0.0,
        &mut temp,
        -1,
    )?;

    core::subtract(&temp, &channels.get(0)?, &mut yb, &core::no_array(), -1)?;

    core::mean_std_dev(&rg, &mut rg_mean, &mut rg_std, &core::no_array())?;
    core::mean_std_dev(&yb, &mut yb_mean, &mut yb_std, &core::no_array())?;

    let std_root = (rg_std.at::<f64>(0)?.powi(2) + yb_std.at::<f64>(0)?.powi(2)).sqrt();
    let mean_root = (rg_mean.at::<f64>(0)?.powi(2) + yb_mean.at::<f64>(0)?.powi(2)).sqrt();

    Ok(std_root + (0.3 * mean_root))
}

fn main() -> opencv::Result<(), ProcessError> {
    let args = Args::parse();

    let src = imgcodecs::imread(&args.image, imgcodecs::IMREAD_COLOR)?;

    let mut bgr_planes = VectorOfMat::new();
    core::split(&src, &mut bgr_planes)?;

    let colorfulness = image_colorfulness(&bgr_planes)?;

    println!("Colorfulness: {}", colorfulness);

    Ok(())
}
