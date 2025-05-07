use std::{error::Error, fs::File};

use base64::{Engine as _, engine::general_purpose};

use serde::{Deserialize, Serialize};
use serde_json::to_writer_pretty;

use tch::{Device, Kind, Tensor, nn, nn::ModuleT, nn::OptimizerConfig, no_grad};

#[derive(Serialize, Deserialize)]
pub struct ModelJson {
    pub input_size: i64,
    pub hidden_layers: Vec<i64>,
    pub output_size: i64,
    pub lr: f64,
    weights: String,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub input_size: i64,
    pub hidden_layers: Vec<i64>,
    pub output_size: i64,
    pub lr: f64,
    vs: nn::VarStore,
    net: nn::SequentialT,
    opt: nn::Optimizer,
}

impl NeuralNetwork {
    pub fn new(
        input_size: i64,
        hidden_layers: &[i64],
        output_size: i64,
        lr: f64,
    ) -> Result<Self, Box<dyn Error>> {
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let vs = nn::VarStore::new(device);
        let mut net = nn::seq_t();

        let mut last_size = input_size;
        for (i, &h) in hidden_layers.iter().enumerate() {
            net = net.add(nn::linear(
                &vs.root() / format!("layer{}", i),
                last_size,
                h,
                Default::default(),
            ));
            net = net.add_fn(|xs| xs.relu());
            last_size = h;
        }

        net = net.add(nn::linear(
            &vs.root() / "output",
            last_size,
            output_size,
            Default::default(),
        ));
        net = net.add_fn(|xs| xs.sigmoid());

        let opt = nn::Adam::default().build(&vs, lr)?;

        return Ok(NeuralNetwork {
            input_size,
            hidden_layers: hidden_layers.to_vec(),
            output_size,
            lr,
            vs,
            net,
            opt,
        });
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut buffer = Vec::<u8>::new();
        self.vs.save_to_stream(&mut buffer)?;
        let b64 = general_purpose::STANDARD.encode(&buffer);

        let model = ModelJson {
            input_size: self.input_size,
            hidden_layers: self.hidden_layers.clone(),
            output_size: self.output_size,
            lr: self.lr,
            weights: b64,
        };

        let file = File::create(path)?;
        to_writer_pretty(file, &model)?;

        return Ok(());
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let model: ModelJson = serde_json::from_reader(file)?;

        let b64 = model.weights.as_str();
        let raw = general_purpose::STANDARD.decode(b64)?;

        let mut loaded_model = NeuralNetwork::new(
            model.input_size,
            &model.hidden_layers,
            model.output_size,
            model.lr,
        )?;

        loaded_model
            .vs
            .load_from_stream(std::io::Cursor::new(raw))?;

        return Ok(loaded_model);
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32]) -> f32 {
        let preds = self.net.forward_t(&Tensor::from_slice(inputs), true);

        let loss_tensor = (&preds - Tensor::from_slice(targets))
            .pow_tensor_scalar(2)
            .mean(Kind::Float);
        let loss_value = loss_tensor.double_value(&[]);

        self.opt.zero_grad();
        loss_tensor.backward();
        self.opt.step();

        return loss_value as f32;
    }

    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        let tensor = no_grad(|| self.net.forward_t(&Tensor::from_slice(inputs), false));
        let mut buffer = vec![0.0; tensor.size().iter().product::<i64>() as usize];

        let len = buffer.len();

        tensor.copy_data(&mut buffer, len);

        return buffer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn save_and_load() {
        let nn = NeuralNetwork::new(2, &[4, 4], 1, 1e-2).unwrap();
        nn.save("/tmp/nn.json").unwrap();
        let nn2 = NeuralNetwork::load("/tmp/nn.json").unwrap();

        assert_eq!(nn.predict(&[0.0, 0.0]), nn2.predict(&[0.0, 0.0]));
        assert_eq!(nn.predict(&[0.0, 1.0]), nn2.predict(&[0.0, 1.0]));
        assert_eq!(nn.predict(&[1.0, 0.0]), nn2.predict(&[1.0, 0.0]));
        assert_eq!(nn.predict(&[1.0, 1.0]), nn2.predict(&[1.0, 1.0]));
    }

    #[test]
    fn xor_problem() {
        let mut nn = NeuralNetwork::new(2, &[8, 8], 1, 1e-2).unwrap();

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
                nn.train(input, target);
                cycle_count += 1;
            }

            let mut total_error = 0.0;
            for (input, target) in &inputs {
                let output = nn.predict(input);
                total_error += (target[0] - output[0]).abs();
            }

            if timer.elapsed().as_secs() >= 5 {
                println!("cycles: {}, error: {}", cycle_count / 5, total_error);
                cycle_count = 0;
                timer = Instant::now();
            }

            if total_error < 0.01 {
                break;
            }
        }
    }
}
