use rand::random_range;

#[derive(Debug)]
pub struct Perceptron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
}

impl Perceptron {
    pub fn new(weights_count: u32, learning_rate: f64) -> Perceptron {
        let mut weights: Vec<f64> = vec![];

        for _ in 0..weights_count {
            weights.push(random_range(-1.0..1.0));
        }

        return Perceptron {
            weights,
            bias: 1.0,
            learning_rate,
        };
    }

    fn activation(n: f64) -> f64 {
        if n >= 0.0 {
            return 1.0;
        } else {
            return -1.0;
        }
    }

    pub fn guess(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }

        let output = Perceptron::activation(sum);
        return output;
    }

    pub fn train(&mut self, inputs: &Vec<f64>, target: &f64) {
        let guess = self.guess(inputs);
        let error = target - guess;

        for i in 0..self.weights.len() {
            self.weights[i] += error * inputs[i] * self.learning_rate;
        }
    }
}
