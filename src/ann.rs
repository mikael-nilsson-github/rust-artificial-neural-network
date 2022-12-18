//#![allow(dead_code, unused)]

use nalgebra;
use rand;
use rand_distr::{self, Distribution};

const ETA: f64 = 0.01;

fn activation_function(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn activation_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

/* fn activation_function(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

fn activation_derivative(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
} */

#[derive(Debug)]
pub struct Layer {
    pub num_in: usize,
    pub num_out: usize,
    pub weights: nalgebra::DMatrix<f64>,
    pub biases: nalgebra::DMatrix<f64>,
    pub batch_vector: nalgebra::DMatrix<f64>,
    pub activation: nalgebra::DMatrix<f64>,
    pub deda: nalgebra::DMatrix<f64>,
    pub target: nalgebra::DMatrix<f64>,
}

impl Layer {
    pub fn new(num_in: usize, num_out: usize, batch_size: usize) -> Self {
        let dist = rand_distr::Normal::new(0.0, 0.25).unwrap();
        let mut rng = rand::thread_rng();
        Self {
            num_in: num_in,
            num_out: num_out,
            weights: nalgebra::DMatrix::from_fn(num_in, num_out, |_, _| dist.sample(&mut rng)),
            biases: nalgebra::DMatrix::from_fn(1, num_out, |_, _| dist.sample(&mut rng)),
            batch_vector: nalgebra::DMatrix::<f64>::repeat(batch_size, 1, 1.0),
            activation: nalgebra::DMatrix::<f64>::zeros(batch_size, num_out),
            deda: nalgebra::DMatrix::<f64>::zeros(batch_size, num_out),
            target: nalgebra::DMatrix::<f64>::zeros(batch_size, num_out),
        }
    }
}
pub struct Network {
    pub topology: Vec<usize>,
    pub num_layers: usize,
    pub network: Vec<Layer>,
}

impl Network {
    pub fn new(topology: Vec<usize>, batch_size: usize) -> Self {
        let length = topology.len();
        let mut tmp: Vec<Layer> = Vec::new();
        tmp.push(Layer::new(topology[0], topology[0], batch_size));
        for i in 1..length {
            tmp.push(Layer::new(topology[i - 1], topology[i], batch_size));
        }
        Self {
            topology: topology,
            num_layers: length,
            network: tmp,
        }
    }

    pub fn forward_propagation(&mut self, input: &nalgebra::DMatrix<f64>) {
        self.network[0].activation.copy_from(input);
        for i in 1..self.num_layers {
            let sum = &self.network[i - 1].activation * &self.network[i].weights
                + &self.network[i].batch_vector * &self.network[i].biases;
            self.network[i].activation = sum.map(activation_function);
        }
    }

    pub fn backward_propagation(&mut self, target: &nalgebra::DMatrix<f64>) {
        self.network[self.num_layers - 1].deda =
            &self.network[self.num_layers - 1].activation - target;
        for i in (1..self.num_layers).rev() {
            let dads = self.network[i].activation.map(activation_derivative);
            let deds = dads.component_mul(&self.network[i].deda);
            let gradw = self.network[i - 1].activation.transpose() * &deds;
            let gradb = self.network[i - 1].batch_vector.transpose() * &deds;
            self.network[i - 1].deda = deds * self.network[i].weights.transpose();
            self.network[i].weights = &self.network[i].weights - ETA * gradw;
            self.network[i].biases = &self.network[i].biases - ETA * gradb;
        }
    }

    pub fn print(&self) {
        for layer in &self.network {
            println!("num_in : {}", layer.num_in);
            println!("num_out: {}", layer.num_out);
            println!("activation -> {:.6}", layer.activation);
            println!("- - - - - - - - - - - - -");
        }
    }
}

pub fn run() {
    let input = nalgebra::DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

    let target = nalgebra::DMatrix::from_row_slice(4, 2, &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);

    let topology = vec![2, 8, 8, 8, 2];
    let mut my_network = Network::new(topology, 4);
    //my_network.initialize(input, target);
    for _ in 0..300_000 {
        my_network.forward_propagation(&input);
        my_network.backward_propagation(&target);
    }
    my_network.print();
}

#[cfg(test)]
mod tests {
    use crate::ann;
    #[test]
    fn test_activation_function() {
        let result = ann::activation_function(0.0);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_activation_derivative() {
        let result = ann::activation_derivative(0.5);
        assert_eq!(result, 0.25);
    }
}
