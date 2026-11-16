pub struct Network {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl Network {
    pub fn new(layers: &[usize], learning_rate: f64) -> Self {
        assert!(layers.len() >= 2);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..layers.len() - 1 {
            let input_size = layers[i];
            let output_size = layers[i + 1];
            weights.push(vec![vec![0.0; output_size]; input_size]);
            biases.push(vec![0.0; output_size]);
        }
        let mut network = Self {
            weights,
            biases,
            learning_rate
        };
        network.init_weights();
        network
    }

    fn init_weights(&mut self) {
        let mut seed: u64 = 12345;
        for layer in &mut self.weights {
            for neuron in layer {
                for weight in neuron {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let value = ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
                    *weight = value * 0.5;
                }
            }
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(output: f64) -> f64 {
        output * (1.0 - output)
    }

    fn forward_all(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activaions = vec![inputs.to_vec()];
        for (layer_index, layer_weights) in self.weights.iter().enumerate() {
            let mut layer_output = Vec::new();
            for neuron_index in 0..layer_weights[0].len() {
                let mut sum = self.biases[layer_index][neuron_index];
                for input_index in 0..layer_weights.len() {
                    sum += activaions[layer_index][input_index] * layer_weights[input_index][neuron_index];
                }
                layer_output.push(Self::sigmoid(sum));
            }
            activaions.push(layer_output);
        }
        activaions
    }

    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let activation = self.forward_all(inputs);
        activation.last().unwrap().clone()
    }

    pub fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        let activations = self.forward_all(inputs);
        let last_layer = activations.len() - 1;
        let mut loss = 0.0;
        for (output, target) in activations[last_layer].iter().zip(targets.iter()) {
            loss += 0.5 * (target - output).powi(2);
        }
        let mut deltas: Vec<Vec<f64>> = vec![Vec::new(); self.weights.len()];
        deltas[last_layer - 1] = activations[last_layer].iter().zip(targets.iter()).map(|(output, target)| {(output - target) * Self::sigmoid_derivative(*output)}).collect();
        for layer in (0..last_layer - 1).rev() {
            let mut layer_delta = vec![0.0; self.weights[layer][0].len()];
            for neuron in 0..layer_delta.len() {
                let mut error = 0.0;
                for next_neuron in 0..deltas[layer + 1].len() {
                    error += deltas[layer + 1][next_neuron] * self.weights[layer + 1][neuron][next_neuron];
                }
                let output = activations[layer + 1][neuron];
                layer_delta[neuron] = error * Self::sigmoid_derivative(output);
            }
            deltas[layer] = layer_delta;
        }
        for layer in 0..self.weights.len() {
            for input_index in 0..self.weights[layer].len() {
                for neuron_index in 0..self.weights[layer][input_index].len() {
                    self.weights[layer][input_index][neuron_index] -= self.learning_rate * deltas[layer][neuron_index] * activations[layer][input_index];
                }
            }
            for neuron_index in 0..self.biases[layer].len() {
                self.biases[layer][neuron_index] -= self.learning_rate * deltas[layer][neuron_index];
            }
        }
        loss
    }

    pub fn mse(&self, dataset: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let mut total_loss = 0.0;
        for (inputs, targets) in dataset {
            let outputs = self.predict(inputs);
            for (output, target) in outputs.iter().zip(targets.iter()) {
                total_loss += (target - output).powi(2);
            }
        }
        total_loss / dataset.len() as f64
    }
}
