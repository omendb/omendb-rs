#![no_main]

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use omendb::distance::{cosine_distance, dot_product, l2_distance, l2_distance_squared};

const MAX_DIM: usize = 2048;

#[derive(Debug, Clone)]
enum DistanceOp {
    L2 { a: Vec<f32>, b: Vec<f32> },
    L2Squared { a: Vec<f32>, b: Vec<f32> },
    Dot { a: Vec<f32>, b: Vec<f32> },
    Cosine { a: Vec<f32>, b: Vec<f32> },
}

fn generate_vector(u: &mut Unstructured<'_>, dim: usize) -> arbitrary::Result<Vec<f32>> {
    let vector: Vec<f32> = (0..dim)
        .map(|_| u.arbitrary::<f32>().unwrap_or(0.0))
        .collect();
    Ok(vector)
}

impl<'a> Arbitrary<'a> for DistanceOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op_type: u8 = u.int_in_range(0..=3)?;
        let dim: usize = u.int_in_range(1..=MAX_DIM)?;

        let a = generate_vector(u, dim)?;
        let b = generate_vector(u, dim)?;

        match op_type {
            0 => Ok(DistanceOp::L2 { a, b }),
            1 => Ok(DistanceOp::L2Squared { a, b }),
            2 => Ok(DistanceOp::Dot { a, b }),
            3 => Ok(DistanceOp::Cosine { a, b }),
            _ => unreachable!(),
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);

    let ops: Vec<DistanceOp> = match u.arbitrary() {
        Ok(ops) => ops,
        Err(_) => return,
    };

    for op in ops {
        match op {
            DistanceOp::L2 { a, b } => {
                // Must not panic, even with NaN/Inf inputs
                let result = l2_distance(&a, &b);
                // Result should be non-negative (or NaN for invalid inputs)
                assert!(result >= 0.0 || result.is_nan());
            }
            DistanceOp::L2Squared { a, b } => {
                let result = l2_distance_squared(&a, &b);
                assert!(result >= 0.0 || result.is_nan());
            }
            DistanceOp::Dot { a, b } => {
                // Dot product can be any value
                let _result = dot_product(&a, &b);
            }
            DistanceOp::Cosine { a, b } => {
                let result = cosine_distance(&a, &b);
                // Cosine distance should be in [0, 2] for valid inputs
                // or 1.0 for zero vectors, or NaN for invalid inputs
                assert!((result >= 0.0 && result <= 2.0) || result.is_nan() || result == 1.0);
            }
        }
    }
});
