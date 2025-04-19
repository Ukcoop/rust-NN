use rand::random_range;

use nn_library::Perceptron;

fn f(x: f64) -> f64 {
    return 3.0 * x + 2.0;
}

fn get_expected_output(point: (f64, f64)) -> f64 {
    if f(point.0) < point.1 {
        return 1.0;
    } else {
        return -1.0;
    }
}

fn main() {
    let mut points: Vec<(f64, f64)> = vec![];

    for _ in 0..100 {
        points.push((random_range(-100.0..100.0), random_range(-100.0..100.0)));
    }

    let mut p = Perceptron::new(3, 0.01);

    loop {
        let mut error_sum = 0.0;

        for point in &points {
            let expected_output = get_expected_output(*point);

            let output = p.guess(&vec![point.0, point.1, p.bias]);
            error_sum += (expected_output - output).abs();

            p.train(&vec![point.0, point.1, p.bias], &expected_output);
        }

        println!("Error: {}", error_sum);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
