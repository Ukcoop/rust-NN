use std::error::Error;

pub mod matrix;

use matrix::Matrix;

#[derive(Debug)]
pub struct NeuralNetwork {
    pub input_nodes: u32,
    pub hidden_nodes: u32,
    pub output_nodes: u32,
    pub learning_rate: f64,
    pub weights_ih: Matrix,
    pub weights_ho: Matrix,
    pub bias_h: Matrix,
    pub bias_o: Matrix,
}

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + f64::exp(-x));
}

fn sigmoid_prime(y: f64) -> f64 {
    return y * (1.0 - y);
}

impl NeuralNetwork {
    pub fn new(
        input_nodes: u32,
        hidden_nodes: u32,
        output_nodes: u32,
        learning_rate: f64,
    ) -> NeuralNetwork {
        let mut nn = NeuralNetwork {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            weights_ih: Matrix::new(hidden_nodes as usize, input_nodes as usize),
            weights_ho: Matrix::new(output_nodes as usize, hidden_nodes as usize),
            bias_h: Matrix::new(hidden_nodes as usize, 1),
            bias_o: Matrix::new(output_nodes as usize, 1),
        };

        nn.weights_ih.randomize();
        nn.weights_ho.randomize();
        nn.bias_h.randomize();
        nn.bias_o.randomize();

        return nn;
    }

    pub fn feed_forward(&self, input: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
        let inputs = Matrix::from_vector(input);

        let mut hidden = Matrix::multiply(&self.weights_ih, &inputs)?;
        hidden.add(&self.bias_h)?;
        hidden.map(sigmoid);

        let mut outputs = Matrix::multiply(&self.weights_ho, &hidden)?;
        outputs.add(&self.bias_o)?;
        outputs.map(sigmoid);

        return Ok(outputs.to_vector());
    }

    pub fn train(&mut self, input: &[f64], target: &[f64]) -> Result<(), Box<dyn Error>> {
        let inputs = Matrix::from_vector(input);

        let mut hidden = Matrix::multiply(&self.weights_ih, &inputs)?;
        hidden.add(&self.bias_h)?;
        hidden.map(sigmoid);

        let mut outputs = Matrix::multiply(&self.weights_ho, &hidden)?;
        outputs.add(&self.bias_o)?;
        outputs.map(sigmoid);

        let targets = Matrix::from_vector(target);

        let output_errors = Matrix::subtract(&targets, &outputs)?;

        let mut gradients = Matrix::static_map(&outputs, sigmoid_prime);
        gradients = Matrix::elementwise_multiply(&gradients, &output_errors)?;
        gradients.scale(self.learning_rate);

        let hidden_t = Matrix::transpose(&hidden);
        let who_deltas = Matrix::multiply(&gradients, &hidden_t)?;

        self.bias_o.add(&gradients)?;
        self.weights_ho.add(&who_deltas)?;

        let who_t = Matrix::transpose(&self.weights_ho);
        let hidden_errors = Matrix::multiply(&who_t, &output_errors)?;

        let mut hidden_gradients = Matrix::static_map(&hidden, sigmoid_prime);
        hidden_gradients = Matrix::elementwise_multiply(&hidden_gradients, &hidden_errors)?;
        hidden_gradients.scale(self.learning_rate);

        let input_t = Matrix::transpose(&inputs);
        let wih_deltas = Matrix::multiply(&hidden_gradients, &input_t)?;

        self.bias_h.add(&hidden_gradients)?;
        self.weights_ih.add(&wih_deltas)?;

        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn xor_problem() {
        let mut nn = NeuralNetwork::new(2, 4, 1, 0.1);

        let inputs = vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let mut timer = Instant::now();
        let mut cycle_count: usize = 0;

        loop {
            for (input, target) in &inputs {
                nn.train(input, target).unwrap();
                cycle_count += 1;
            }

            let mut total_error = 0.0;
            for (input, target) in &inputs {
                let output = nn.feed_forward(input).unwrap();
                total_error += (target[0] - output[0]).abs();
            }

            if timer.elapsed().as_secs() >= 5 {
                println!("cycles: {}, error: {}", cycle_count / 5, total_error);
                cycle_count = 0;
                timer = Instant::now();
            }

            if total_error < 0.04 {
                break;
            }
        }
    }
}
