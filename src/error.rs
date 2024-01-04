use thiserror::Error;

#[derive(Error, Debug)]
pub enum PerceptronError {
  #[error("{0}")]
  MisMatchLength(String),
}
