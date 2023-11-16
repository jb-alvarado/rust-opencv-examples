use std::io;

use derive_more::Display;

#[derive(Debug, Display)]
pub enum ProcessError {
    #[display(fmt = "Failed to spawn ffmpeg/ffprobe. {}", _0)]
    CommandSpawn(io::Error),
    #[display(fmt = "Failed to read data from ffmpeg/ffprobe. {}", _0)]
    IO(io::Error),
    #[display(fmt = "Failed to read data from opencv. {}", _0)]
    Image(String),
}

impl From<std::io::Error> for ProcessError {
    fn from(err: std::io::Error) -> ProcessError {
        ProcessError::CommandSpawn(err)
    }
}

impl From<opencv::Error> for ProcessError {
    fn from(err: opencv::Error) -> ProcessError {
        ProcessError::Image(err.message)
    }
}
