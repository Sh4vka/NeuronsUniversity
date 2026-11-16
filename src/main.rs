use neuron::{AND, OR, XOR, delta, heb, back, heb_dig, delta_dig};

const ZERO: [f64; 15] = [
    -1.0, -1.0, -1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
];

const ONE: [f64; 15] = [
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
];

pub const TWO: [f64; 15] = [
    -1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0,  1.0,
];

fn main() {
    heb(&AND);
    heb(&OR);
    heb(&XOR);

    heb_dig(&ZERO);
    heb_dig(&ONE);
    heb_dig(&TWO);

    delta(&AND);
    delta(&OR);
    delta(&XOR);

    delta_dig(&ZERO);
    delta_dig(&ONE);
    delta_dig(&TWO);

    back();
}
