use thiserror::Error;

#[derive(Error, Debug)]
pub enum PerceptronError {
    #[error("{0}")]
    MisMatchLength(String),
    #[error("Number of interations greater than number train instances")]
    NIterLenError,
}
