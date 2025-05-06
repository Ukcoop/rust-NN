use std::error::Error;

use tch::{nn, nn::ModuleT, nn::OptimizerConfig, no_grad, Device, Kind, Tensor};

pub struct NeuralNetwork {
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

        return Ok(NeuralNetwork { net, opt });
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
    fn xor_problem() {
        let mut nn = NeuralNetwork::new(2, &[4, 4], 1, 1e-2).unwrap();

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
