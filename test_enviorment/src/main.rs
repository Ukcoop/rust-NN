use nn_library::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(2, 2, 1);

    let inputs = vec![
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    loop {
        for (input, target) in inputs.iter() {
            nn.train(input, target);
        }

        let mut total = 0.0;
        for (input, target) in inputs.iter() {
            let output = nn.feed_forward(input);
            total += (target[0] - output[0]).abs();
        }

        if total < 0.04 {
            break;
        }

        println!("Error: {}", total);
    }
}
