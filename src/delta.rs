pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
}

impl Neuron {
    pub fn new(inputs: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; inputs],
            bias: 0.0,
            learning_rate,
        }
    }

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        inputs.iter().zip(self.weights.iter()).map(|(x, w)| x * w).sum::<f64>() + self.bias
    }

    pub fn train(&mut self, inputs: &[f64], target: f64) -> f64 {
        let output = self.predict(inputs);
        let error = target - output;
        for (weight, input) in self.weights.iter_mut().zip(inputs.iter()) {
            *weight += self.learning_rate * error * input;
        }
        self.bias += self.learning_rate * error;
        error
    }
}
