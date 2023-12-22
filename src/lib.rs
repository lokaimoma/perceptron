use ndarray::{array, concatenate, Array1, Array2, ArrayView1, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::{RandomExt, SamplingStrategy};

pub mod error;

pub type Result<T> = std::result::Result<T, error::PerceptronError>;

pub enum StepFunction {
    HEAVISIDE,
    SIGNUM,
}

#[allow(dead_code)]
pub struct Perceptron {
    step_function: StepFunction,
    w: Array1<f64>,
    learning_rate: f64,
    decay_rate: f64,
}

#[allow(non_snake_case)]
impl Perceptron {
    pub fn new(
        dim: usize,
        step_function: StepFunction,
        learning_rate: f64,
        decay_rate: f64,
    ) -> Self {
        let w = Array1::random(dim + 1, Uniform::new(0.0, 1.0)); //weights for bias & features

        return Self {
            step_function,
            w,
            learning_rate,
            decay_rate,
        };
    }

    pub fn train(&mut self, X: Array2<f64>, targets: Array1<i64>, n_iter: usize) -> Result<()> {
        if X.dim().0 != targets.len() {
            return Err(error::PerceptronError::MisMatchLength(format!(
                "Feature matrix and targets array of different length: X={}, targets={}",
                X.len(),
                targets.len()
            )));
        }

        if n_iter > X.len() {
            return Err(error::PerceptronError::NIterLenError);
        }

        let indexes = Array1::range(0f64, X.dim().0 as f64, 1f64);
        let indexes = indexes.sample_axis(Axis(0), n_iter, SamplingStrategy::WithReplacement);

        for i in indexes {
            self._train(X.index_axis(Axis(0), i as usize), targets[i as usize]);

            if self.learning_rate > 0.001 {
                self.learning_rate *= self.decay_rate;
            }

            if self.learning_rate < 0.001 {
                self.learning_rate = 0.001;
            }
        }

        return Ok(());
    }

    #[inline]
    pub fn predict(&self, x: ArrayView1<f64>) -> i64 {
        let bias_with_features = concatenate![Axis(0), array![1.0], x];
        let prediction = (bias_with_features * &self.w).sum();
        let prediction = match self.step_function {
            StepFunction::HEAVISIDE => heaviside(prediction),
            StepFunction::SIGNUM => signum(prediction),
        };
        return prediction;
    }

    pub fn weights(&self) -> ArrayView1<f64> {
        return self.w.view();
    }

    #[inline]
    fn _train(&mut self, x: ArrayView1<f64>, target: i64) {
        let prediction = self.predict(x);
        let error: f64 = (target - prediction) as f64;
        let bias_with_features = concatenate![Axis(0), array![1.0], x];
        self.w = &self.w + self.learning_rate * error * bias_with_features;
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
