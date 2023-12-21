use ndarray::{Array1, Array2};
use rand::{self, prelude::*};

pub enum StepFunction {
    HEAVISIDE,
    SIGNUM,
}

#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct Perceptron {
    step_function: StepFunction,
    w: Array1<f64>,
    bias: f64,
}

#[allow(non_snake_case)]
#[allow(dead_code)]
impl Perceptron {
    pub fn new(dim: usize, step_function: StepFunction) -> Self {
        let mut rng = rand::thread_rng();
        let w = Array1::from_elem(dim, rng.gen());

        return Self {
            step_function,
            w,
            bias: 1f64,
        };
    }
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
