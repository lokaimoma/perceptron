use ndarray::array;
use perceptron::Perceptron;

#[allow(non_snake_case)]
fn main() {
    // random data generated with the formula: Heaviside(x1*1.8 + x2*0.2 + x3*0.3)
    let X = array![
        [0.3, 0.1, 0.5],
        [-1.5, -1.2, 0.6],
        [0.2, 0.5, 0.1],
        [-0.7, 0.5, 0.3],
        [0.2, 0.0, -0.2],
        [-0.8, 0.1, 0.7],
    ];
    let y = array![1, 0, 1, 0, 1, 0];

    let mut perceptron = Perceptron::new(X.dim().1, perceptron::StepFunction::HEAVISIDE, 0.05, 0.9);
    println!("Ndim = {}", X.dim().1);
    println!("Training perceptron....");
    perceptron.train(X, y, 6).unwrap();
    println!("Done training perceptron....");
    println!(
        "Predicting [0.3, 0.3, 0.3] Expected output: 1 \nGot: {}",
        perceptron.predict(array![0.3, 0.3, 0.3].view())
    );
    println!(
        "Predicting [-0.3, -0.3, 0.3] Expected output: 0 \nGot: {}",
        perceptron.predict(array![-0.3, -0.3, 0.3].view())
    );
    println!("Weights: {}", perceptron.weights());
}
