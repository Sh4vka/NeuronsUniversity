mod hebb;
mod delta;
mod backprop;

pub const AND: [([f64; 2], f64); 4] = [
    ([-1.0, -1.0], -1.0),
    ([-1.0, 1.0], -1.0),
    ([1.0, -1.0], -1.0),
    ([1.0, 1.0], 1.0),
];

pub const OR: [([f64; 2], f64); 4] = [
    ([-1.0, -1.0], -1.0),
    ([-1.0, 1.0], 1.0),
    ([1.0, -1.0], 1.0),
    ([1.0, 1.0], 1.0),
];

pub const XOR: [([f64; 2], f64); 4] = [
    ([-1.0, -1.0], -1.0),
    ([-1.0, 1.0], 1.0),
    ([1.0, -1.0], 1.0),
    ([1.0, 1.0], -1.0),
];

pub const ZERO: [f64; 15] = [
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
];

pub const ONE: [f64; 15] = [
    -1.0,  1.0, -1.0,
     1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
];

pub const TWO: [f64; 15] = [
     1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0, -1.0,
     1.0,  1.0,  1.0,
];

pub fn heb(mas: &[([f64; 2], f64); 4]) {
    println!("\n\nHebb:");
    let mut neuron = hebb::Neuron::new(2, 1.0);
    for (inputs, target) in mas {
        neuron.train(inputs, *target);
    }
    println!("Weights: {:?}", neuron.weights);
    println!("Bias: {}", neuron.bias);
    for (inputs, expected) in mas {
        let predicted = neuron.predict(inputs);
        println!("Inputs: {:?}, Expected: {}, Predicted: {}", inputs, expected, predicted);
    }
}

pub fn heb_dig(mas: &[f64; 15]) {
    println!("\n\nHebb DIGIT:");
    let digits = [ZERO, ONE, TWO];
    let mut neurons = [
        hebb::Neuron::new(15, 1.0),
        hebb::Neuron::new(15, 1.0),
        hebb::Neuron::new(15, 1.0),
    ];

    for (digit_index, digit) in digits.iter().enumerate() {
        for (neuron_index, neuron) in neurons.iter_mut().enumerate() {
            let target = if digit_index == neuron_index {
                1.0
            } else {
                -1.0
            };
            neuron.train(digit, target);
        }
    }
    let mut best_digit = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (index, neuron) in neurons.iter().enumerate() {
        let score = neuron.net(mas);
        println!("Neuron {}: {:.2}", index, score);
        if score > best_score {
            best_score = score;
            best_digit = index;
        }
    }
    let names = ["Zero", "One", "Two"];
    println!("\nRecognized digit: {}", names[best_digit]);
}

pub fn delta(mas: &[([f64; 2], f64); 4]) {
    println!("\n\nDelta rule:");
    let mut neuron = delta::Neuron::new(2, 0.1);
    let epochs = 100;
    for epoch in 0..epochs {
        let mut total_error = 0.0;
        for (inputs, target) in mas {
            let error = neuron.train(inputs, *target);
            total_error += error;
        }
        let mse = total_error / AND.len() as f64;
        if epoch % 10 == 0 {
            println!("Epoch: {}, MSE: {:.6}", epoch, mse);
        }
    }
    println!("\nWeights: {:?}", neuron.weights);
    println!("Bias: {}", neuron.bias);
    for (inputs, expected) in mas{
        let out = neuron.predict(inputs);
        let predicted = if out >= 0.0 {1} else {-1};
        println!("Input: {:?}, Expected: {}, Output: {:.4}, Predicted: {}", inputs, expected, out, predicted);
    }
}

pub fn delta_dig(mas: &[f64; 15]) {
    println!("\n\nDelta rule DIGIT:");
    let digits = [ZERO, ONE, TWO];
    let mut neurons = [
        delta::Neuron::new(15, 0.01),
        delta::Neuron::new(15, 0.01),
        delta::Neuron::new(15, 0.01),
    ];

    let epochs = 1000;
    for _ in 0..epochs {
        for (digit_index, digit) in digits.iter().enumerate() {
            for (neuron_index, neuron) in neurons.iter_mut().enumerate() {
                let target = if digit_index == neuron_index {
                    1.0
                } else {
                    -1.0
                };
                neuron.train(digit, target);
            }
        }
    }

    let mut best_digit = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (index, neuron) in neurons.iter().enumerate() {
        let score = neuron.predict(mas);
        println!("Neuron {}: {:.4}", index, score);
        if score > best_score {
            best_score = score;
            best_digit = index;
        }
    }
    let names = ["Zero", "One", "Two"];
    println!("\nRecognized digit: {}", names[best_digit]);
}


pub fn back(mas: &[([f64; 2], f64); 4]) {
    println!("\n\nBackprop:");
    let dataset: Vec<(Vec<f64>, Vec<f64>)> = mas
        .iter()
        .map(|(inputs, target)| {
            let inputs = inputs
                .iter()
                .map(|x| (x + 1.0) / 2.0)
                .collect();

            let target = vec![(target + 1.0) / 2.0];

            (inputs, target)
        })
        .collect();
    let mut network = backprop::Network::new(&[2, 4, 1], 0.5);
    let epochs = 20_000;
    for epoch in 0..epochs {
        for (inputs, targets) in &dataset {
            network.train(inputs, targets);
        }
        if epoch % 1000 == 0 {
            println!(
                "Epoch: {}, MSE: {:.6}",
                epoch,
                network.mse(&dataset)
            );
        }
    }

    println!("\nResults:");
    for (inputs, expected) in mas {
        let converted: Vec<f64> = inputs
            .iter()
            .map(|x| (x + 1.0) / 2.0)
            .collect();
        let output = network.predict(&converted);
        let predicted = if output[0] >= 0.5 { 1 } else { -1 };
        println!(
            "Input: {:?}, Expected: {}, Output: {:.4}, Predicted: {}",
            inputs,
            expected,
            output[0],
            predicted
        );
    }
}


pub fn back_dig(mas: &[f64; 15]) {
    println!("\n\nBackprop DIGIT:");

    let dataset: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (ZERO.to_vec(), vec![1.0, 0.0, 0.0]),
        (ONE.to_vec(),  vec![0.0, 1.0, 0.0]),
        (TWO.to_vec(),  vec![0.0, 0.0, 1.0]),
    ]
        .into_iter()
        .map(|(inputs, targets)| {
            let inputs = inputs
                .iter()
                .map(|x| (x + 1.0) / 2.0)
                .collect();

            (inputs, targets)
        })
        .collect();

    let mut network = backprop::Network::new(&[15, 10, 3], 0.1);
    let epochs = 10_000;
    for epoch in 0..epochs {
        for (inputs, targets) in &dataset {
            network.train(inputs, targets);
        }
        if epoch % 1000 == 0 {
            println!(
                "Epoch: {}, MSE: {:.6}",
                epoch,
                network.mse(&dataset)
            );
        }
    }
    let converted: Vec<f64> = mas
        .iter()
        .map(|x| (x + 1.0) / 2.0)
        .collect();
    let output = network.predict(&converted);

    let names = ["Zero", "One", "Two"];

    println!("\nRecognition results:");
    for (index, score) in output.iter().enumerate() {
        println!("{}: {:.4}", names[index], score);
    }
    let recognized = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    println!("\nRecognized digit: {}", names[recognized]);
}

