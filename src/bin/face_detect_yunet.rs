/// Face detection with YuNet model
/// YuNet from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
///
use opencv::{
    core::{self, Mat, Point},
    highgui, imgcodecs, imgproc,
    objdetect::FaceDetectorYN,
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

pub fn detect_faces(src: &Mat) -> Result<Mat, ProcessError> {
    let mut matched_image = src.clone();
    let size = src.size()?;
    let fd_model = "./assets/face_detection_yunet_2023mar.onnx";
    let mut faces = Mat::default();
    let mut model = FaceDetectorYN::create(
        fd_model,
        "",
        core::Size {
            width: size.width as i32,
            height: size.height as i32,
        },
        0.9,
        0.3,
        5000,
        0,
        0,
    )?;

    model.set_input_size(core::Size {
        width: size.width as i32,
        height: size.height as i32,
    })?;

    model.detect(&src, &mut faces)?;

    if faces.dims() > 0 {
        println!("Face detected!");

        let f_vectors = faces.to_vec_2d::<f32>()?;

        for face in f_vectors.iter().filter(|c| c.len() > 14) {
            let accuracy = face[14];
            let face_x = face[0];
            let face_y = face[1];
            let f_width = face[2];
            let f_height = face[3];

            println!("Detection accuracy: {accuracy}");

            let eye_l = Point {
                x: face[4] as i32,
                y: face[5] as i32,
            };

            let eye_r = Point {
                x: face[6] as i32,
                y: face[7] as i32,
            };

            let nose = Point {
                x: face[8] as i32,
                y: face[9] as i32,
            };

            let face_rec = core::Rect {
                x: face_x as i32,
                y: face_y as i32,
                width: f_width as i32,
                height: f_height as i32,
            };

            imgproc::circle(
                &mut matched_image,
                nose,
                1,
                core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                2,
                0,
                0,
            )?;

            imgproc::circle(
                &mut matched_image,
                eye_l,
                1,
                core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                2,
                0,
                0,
            )?;

            imgproc::circle(
                &mut matched_image,
                eye_r,
                1,
                core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                2,
                0,
                0,
            )?;

            imgproc::rectangle(
                &mut matched_image,
                face_rec,
                core::Scalar::new(0.0, 255.0, 20.0, 0.0),
                1,
                8,
                0,
            )?;
        }
    } else {
        println!("No face found...");
    }

    Ok(matched_image)
}

fn main() -> opencv::Result<(), ProcessError> {
    let args = Args::parse();
    let src = imgcodecs::imread(&args.image, imgcodecs::IMREAD_COLOR)?;

    match detect_faces(&src) {
        Ok(face) => {
            highgui::imshow("Face Detect", &face)?;
            highgui::wait_key(0)?;
        }
        Err(e) => println!("Detection error: {e:?}"),
    };

    Ok(())
}
