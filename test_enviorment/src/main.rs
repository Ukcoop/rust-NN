use nn_library::NeuralNetwork;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
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
            nn.train(input, target)?;
            cycle_count += 1;
        }

        let mut total_error = 0.0;
        for (input, target) in &inputs {
            let output = nn.feed_forward(input)?;
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

    return Ok(());
}
