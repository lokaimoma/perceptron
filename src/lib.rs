use ndarray::{Array1, Array2};

pub enum StepFunction {
    HEAVISIDE,
    SIGNUM,
}

#[allow(non_snake_case)]
pub struct Perceptron {
    pub X: Array2<f64>,
    pub w: Array1<f64>,
    pub bias: f64,
    pub stepFunction: StepFunction,
}
