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

#[inline]
fn heaviside(n: f64) -> i64 {
    return if n < 0.0 { 0 } else { 1 };
}

#[inline]
fn signum(n: f64) -> i64 {
    return if n < 0.0 {
        -1
    } else if n == 0.0 {
        0
    } else {
        1
    };
}
