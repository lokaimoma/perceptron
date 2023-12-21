use ndarray::array;
use perceptron::Perceptron;

#[allow(non_snake_case)]
fn main() {
    let X = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let y = array![0, 1, 1, 1];

    let mut perceptron = Perceptron::new(X.dim().1, perceptron::StepFunction::HEAVISIDE, 1.5, 0.9);
    println!("Training perceptron....");
    perceptron.train(X, y, 4).unwrap();
    println!("Done training perceptron....");
    println!(
        "Predicting 1 or 0: {}",
        perceptron.predict(array![1f64, 0f64].view())
    );
    println!("Weights: {}", perceptron.weights());
}
