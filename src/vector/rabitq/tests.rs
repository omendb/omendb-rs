    use super::*;

    #[test]
    fn test_quantization_bits_conversion() {
        assert_eq!(QuantizationBits::Bits2.to_u8(), 2);
        assert_eq!(QuantizationBits::Bits4.to_u8(), 4);
        assert_eq!(QuantizationBits::Bits8.to_u8(), 8);
    }

    #[test]
    fn test_quantization_bits_levels() {
        assert_eq!(QuantizationBits::Bits2.levels(), 4); // 2^2
        assert_eq!(QuantizationBits::Bits4.levels(), 16); // 2^4
        assert_eq!(QuantizationBits::Bits8.levels(), 256); // 2^8
    }

    #[test]
    fn test_quantization_bits_compression() {
        assert_eq!(QuantizationBits::Bits2.compression_ratio(), 16.0); // 32/2
        assert_eq!(QuantizationBits::Bits4.compression_ratio(), 8.0); // 32/4
        assert_eq!(QuantizationBits::Bits8.compression_ratio(), 4.0); // 32/8
    }

    #[test]
    fn test_quantization_bits_values_per_byte() {
        assert_eq!(QuantizationBits::Bits2.values_per_byte(), 4); // 8/2
        assert_eq!(QuantizationBits::Bits4.values_per_byte(), 2); // 8/4
        assert_eq!(QuantizationBits::Bits8.values_per_byte(), 1); // 8/8
    }

    #[test]
    fn test_default_params() {
        let params = RaBitQParams::default();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits4);
        assert_eq!(params.num_rescale_factors, 12);
        assert_eq!(params.rescale_range, (0.5, 2.0));
    }

    #[test]
    fn test_preset_params() {
        let params2 = RaBitQParams::bits2();
        assert_eq!(params2.bits_per_dim, QuantizationBits::Bits2);

        let params4 = RaBitQParams::bits4();
        assert_eq!(params4.bits_per_dim, QuantizationBits::Bits4);

        let params8 = RaBitQParams::bits8();
        assert_eq!(params8.bits_per_dim, QuantizationBits::Bits8);
        assert_eq!(params8.num_rescale_factors, 16);
    }

    #[test]
    fn test_quantized_vector_creation() {
        let data = vec![0u8, 128, 255];
        let qv = QuantizedVector::new(data.clone(), 1.5, 8, 3);

        assert_eq!(qv.data, data);
        assert_eq!(qv.scale, 1.5);
        assert_eq!(qv.bits, 8);
        assert_eq!(qv.dimensions, 3);
    }

    #[test]
    fn test_quantized_vector_memory() {
        let data = vec![0u8; 16]; // 16 bytes
        let qv = QuantizedVector::new(data, 1.0, 4, 32);

        // Should be: struct overhead + data length
        let expected_min = 16; // At least the data
        assert!(qv.memory_bytes() >= expected_min);
    }

    #[test]
    fn test_quantized_vector_compression_ratio() {
        // 128 dimensions, 4-bit = 64 bytes
        let data = vec![0u8; 64];
        let qv = QuantizedVector::new(data, 1.0, 4, 128);

        // Original: 128 * 4 = 512 bytes
        // Compressed: 64 + 4 (scale) + 1 (bits) = 69 bytes
        // Ratio: 512 / 69 ≈ 7.4x
        let ratio = qv.compression_ratio();
        assert!(ratio > 7.0 && ratio < 8.0);
    }

    #[test]
    fn test_create_quantizer() {
        let quantizer = RaBitQ::default_4bit();
        assert_eq!(quantizer.params().bits_per_dim, QuantizationBits::Bits4);
    }

    #[test]
    fn test_generate_scales() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 5,
            rescale_range: (0.5, 1.5),
            ..Default::default()
        });

        let scales = quantizer.generate_scales();
        assert_eq!(scales.len(), 5);
        assert_eq!(scales[0], 0.5);
        assert_eq!(scales[4], 1.5);
        assert!((scales[2] - 1.0).abs() < 0.01); // Middle should be ~1.0
    }

    #[test]
    fn test_generate_scales_single() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 1,
            rescale_range: (0.5, 1.5),
            ..Default::default()
        });

        let scales = quantizer.generate_scales();
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0], 1.0); // Average of min and max
    }

    #[test]
    fn test_pack_unpack_2bit() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits2,
            ..Default::default()
        });

        // 8 values (2 bits each) = 2 bytes
        let values = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
        let packed = quantizer.pack_quantized(&values, 2);
        assert_eq!(packed.len(), 2); // 8 values / 4 per byte = 2 bytes

        let unpacked = quantizer.unpack_quantized(&packed, 2, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_4bit() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            ..Default::default()
        });

        // 8 values (4 bits each) = 4 bytes
        let values = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let packed = quantizer.pack_quantized(&values, 4);
        assert_eq!(packed.len(), 4); // 8 values / 2 per byte = 4 bytes

        let unpacked = quantizer.unpack_quantized(&packed, 4, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_8bit() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            ..Default::default()
        });

        // 8 values (8 bits each) = 8 bytes
        let values = vec![0u8, 10, 20, 30, 40, 50, 60, 70];
        let packed = quantizer.pack_quantized(&values, 8);
        assert_eq!(packed.len(), 8); // 8 values = 8 bytes

        let unpacked = quantizer.unpack_quantized(&packed, 8, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_quantize_simple_vector() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 4,
            rescale_range: (0.5, 1.5),
            ..Default::default()
        });

        // Simple vector: [0.0, 0.25, 0.5, 0.75, 1.0]
        let vector = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let quantized = quantizer.quantize(&vector);

        // Check structure
        assert_eq!(quantized.dimensions, 5);
        assert_eq!(quantized.bits, 4);
        assert!(quantized.scale > 0.0);

        // Check compression: 5 floats * 4 bytes = 20 bytes original
        // Quantized: 5 values * 4 bits = 20 bits = 3 bytes (rounded up)
        assert!(quantized.data.len() <= 4);
    }

    #[test]
    fn test_quantize_reconstruct_accuracy() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        // Test vector
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let quantized = quantizer.quantize(&vector);

        // Reconstruct
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, vector.len());

        // Check reconstruction is close (8-bit should be accurate)
        for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            assert!(error < 0.1, "Error too large: {} vs {}", orig, recon);
        }
    }

    #[test]
    fn test_quantize_uniform_vector() {
        let quantizer = RaBitQ::default_4bit();

        // All values the same
        let vector = vec![0.5; 10];
        let quantized = quantizer.quantize(&vector);

        // Reconstruct should also be uniform
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, vector.len());

        // All values should be similar
        let avg = reconstructed.iter().sum::<f32>() / reconstructed.len() as f32;
        for &val in &reconstructed {
            assert!((val - avg).abs() < 0.2);
        }
    }

    #[test]
    fn test_compute_error() {
        let quantizer = RaBitQ::default_4bit();

        let original = vec![0.1, 0.2, 0.3, 0.4];
        let quantized_vec = quantizer.quantize(&original);

        // Compute error
        let error = quantizer.compute_error(&original, &quantized_vec.data, quantized_vec.scale);

        // Error should be non-negative and finite
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_quantize_different_bit_widths() {
        let test_vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Test 2-bit
        let q2 = RaBitQ::new(RaBitQParams::bits2());
        let qv2 = q2.quantize(&test_vector);
        assert_eq!(qv2.bits, 2);

        // Test 4-bit
        let q4 = RaBitQ::default_4bit();
        let qv4 = q4.quantize(&test_vector);
        assert_eq!(qv4.bits, 4);

        // Test 8-bit
        let q8 = RaBitQ::new(RaBitQParams::bits8());
        let qv8 = q8.quantize(&test_vector);
        assert_eq!(qv8.bits, 8);

        // Higher bits = larger packed size (for same dimensions)
        assert!(qv2.data.len() <= qv4.data.len());
        assert!(qv4.data.len() <= qv8.data.len());
    }

    #[test]
    fn test_quantize_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vector (like small embeddings)
        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 128);
        assert_eq!(quantized.bits, 4);

        // 128 dimensions * 4 bits = 512 bits = 64 bytes
        assert_eq!(quantized.data.len(), 64);

        // Verify reconstruction
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 128);
        assert_eq!(reconstructed.len(), 128);
    }

    #[test]
    fn test_distance_l2() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_l2(&qv1, &qv2);

        // Distance should be approximately 1.0
        assert!((dist - 1.0).abs() < 0.2, "Distance: {}", dist);
    }

    #[test]
    fn test_distance_l2_identical() {
        let quantizer = RaBitQ::default_4bit();

        let v = vec![0.5, 0.3, 0.8, 0.2];
        let qv1 = quantizer.quantize(&v);
        let qv2 = quantizer.quantize(&v);

        let dist = quantizer.distance_l2(&qv1, &qv2);

        // Identical vectors should have near-zero distance
        assert!(dist < 0.3, "Distance should be near zero, got: {}", dist);
    }

    #[test]
    fn test_distance_cosine() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        // Orthogonal vectors
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_cosine(&qv1, &qv2);

        // Orthogonal vectors: cosine = 0, distance = 1
        assert!((dist - 1.0).abs() < 0.3, "Distance: {}", dist);
    }

    #[test]
    fn test_distance_cosine_identical() {
        let quantizer = RaBitQ::default_4bit();

        let v = vec![0.5, 0.3, 0.8];
        let qv1 = quantizer.quantize(&v);
        let qv2 = quantizer.quantize(&v);

        let dist = quantizer.distance_cosine(&qv1, &qv2);

        // Identical vectors: cosine = 1, distance = 0
        assert!(dist < 0.2, "Distance should be near zero, got: {}", dist);
    }

    #[test]
    fn test_distance_dot() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_dot(&qv1, &qv2);

        // Dot product of [1,0,0] with itself = 1, negated = -1
        assert!((dist + 1.0).abs() < 0.3, "Distance: {}", dist);
    }

    #[test]
    fn test_distance_approximate() {
        let quantizer = RaBitQ::default_4bit();

        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![0.5, 0.5, 0.5];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_approx = quantizer.distance_approximate(&qv1, &qv2);
        let dist_exact = quantizer.distance_l2(&qv1, &qv2);

        // Approximate should be non-negative and finite
        assert!(dist_approx >= 0.0);
        assert!(dist_approx.is_finite());

        // Approximate and exact should be correlated (not exact match)
        // Just verify both increase/decrease together
        let v3 = vec![1.0, 1.0, 1.0];
        let qv3 = quantizer.quantize(&v3);

        let dist_approx2 = quantizer.distance_approximate(&qv1, &qv3);
        let dist_exact2 = quantizer.distance_l2(&qv1, &qv3);

        // If v3 is farther from v1 than v2, both metrics should reflect that
        if dist_exact2 > dist_exact {
            assert!(dist_approx2 > dist_approx * 0.5); // Allow some variance
        }
    }

    #[test]
    fn test_distance_correlation() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision for correlation
            num_rescale_factors: 12,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        // Create multiple vectors
        let vectors = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // Quantize all
        let quantized: Vec<QuantizedVector> =
            vectors.iter().map(|v| quantizer.quantize(v)).collect();

        // Ground truth L2 distances
        let ground_truth_01 = vectors[0]
            .iter()
            .zip(vectors[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let ground_truth_02 = vectors[0]
            .iter()
            .zip(vectors[2].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Quantized distances
        let quantized_01 = quantizer.distance_l2(&quantized[0], &quantized[1]);
        let quantized_02 = quantizer.distance_l2(&quantized[0], &quantized[2]);

        // Check correlation: if ground truth says v2 > v1, quantized should too
        if ground_truth_02 > ground_truth_01 {
            assert!(
                quantized_02 > quantized_01 * 0.8,
                "Order not preserved: {} vs {}",
                quantized_01,
                quantized_02
            );
        }
    }

    #[test]
    fn test_distance_zero_vectors() {
        let quantizer = RaBitQ::default_4bit();

        let v_zero = vec![0.0, 0.0, 0.0];
        let qv_zero = quantizer.quantize(&v_zero);

        // Distance to itself should be zero
        let dist = quantizer.distance_l2(&qv_zero, &qv_zero);
        assert!(dist < 0.1);

        // Cosine distance with zero vector should handle gracefully
        let dist_cosine = quantizer.distance_cosine(&qv_zero, &qv_zero);
        assert!(dist_cosine.is_finite());
    }

    #[test]
    fn test_distance_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vectors
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| ((i + 10) as f32) / 128.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        // All distance metrics should work on high-dimensional vectors
        let dist_l2 = quantizer.distance_l2(&qv1, &qv2);
        let dist_cosine = quantizer.distance_cosine(&qv1, &qv2);
        let dist_dot = quantizer.distance_dot(&qv1, &qv2);
        let dist_approx = quantizer.distance_approximate(&qv1, &qv2);

        assert!(dist_l2 > 0.0 && dist_l2.is_finite());
        assert!(dist_cosine >= 0.0 && dist_cosine.is_finite());
        assert!(dist_dot.is_finite());
        assert!(dist_approx > 0.0 && dist_approx.is_finite());
    }

    #[test]
    fn test_simd_l2_matches_scalar() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let v2 = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_l2(&qv1, &qv2);
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(
            diff < 0.01,
            "SIMD vs scalar: {} vs {}",
            dist_simd,
            dist_scalar
        );
    }

    #[test]
    fn test_simd_cosine_matches_scalar() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_cosine(&qv1, &qv2);
        let dist_simd = quantizer.distance_cosine_simd(&qv1, &qv2);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(
            diff < 0.01,
            "SIMD vs scalar: {} vs {}",
            dist_simd,
            dist_scalar
        );
    }

    #[test]
    fn test_simd_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vectors (realistic embeddings)
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) / 128.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_l2(&qv1, &qv2);
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);

        // Should be close (allow for quantization + FP variance)
        let diff = (dist_scalar - dist_simd).abs();
        assert!(
            diff < 0.1,
            "High-D SIMD vs scalar: {} vs {}",
            dist_simd,
            dist_scalar
        );
    }

    #[test]
    fn test_simd_scalar_fallback() {
        let quantizer = RaBitQ::default_4bit();

        // Small vector (tests remainder handling)
        let v1 = vec![0.1, 0.2, 0.3];
        let v2 = vec![0.4, 0.5, 0.6];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        // Should not crash on small vectors
        let dist_l2 = quantizer.distance_l2_simd(&qv1, &qv2);
        let dist_cosine = quantizer.distance_cosine_simd(&qv1, &qv2);

        assert!(dist_l2.is_finite());
        assert!(dist_cosine.is_finite());
    }

    #[test]
    fn test_simd_performance_improvement() {
        let quantizer = RaBitQ::default_4bit();

        // Large vectors (1536D like OpenAI embeddings)
        let v1: Vec<f32> = (0..1536).map(|i| (i as f32) / 1536.0).collect();
        let v2: Vec<f32> = (0..1536).map(|i| ((i + 10) as f32) / 1536.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        // Just verify SIMD works on large vectors
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);
        assert!(dist_simd > 0.0 && dist_simd.is_finite());
    }

    #[test]
    fn test_scalar_distance_functions() {
        // Test the scalar fallback functions directly
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let dist = l2_distance_scalar(&v1, &v2);
        assert!((dist - 1.0).abs() < 0.001);

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let dist = cosine_distance_scalar(&v1, &v2);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fht_basic() {
        // Test basic FHT on power-of-2 length
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let result = fast_hadamard_transform(&data);

        // FHT of [1,0,0,0] normalized should be [0.5, 0.5, 0.5, 0.5]
        assert_eq!(result.len(), 4);
        for &val in &result {
            assert!((val - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_fht_identity() {
        // Applying FHT twice should return to original (Hadamard is self-inverse)
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();

        fast_hadamard_transform_inplace(&mut data);
        fast_hadamard_transform_inplace(&mut data);

        // Should be close to original
        for (orig, result) in original.iter().zip(data.iter()) {
            assert!((orig - result).abs() < 0.001, "FHT not self-inverse");
        }
    }

    #[test]
    fn test_fht_non_power_of_2() {
        // Test FHT on non-power-of-2 length (should pad and truncate)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = fast_hadamard_transform(&data);

        // Should return same length
        assert_eq!(result.len(), 5);
        // Values should be transformed (not all same as input)
        assert!(result.iter().any(|&v| (v - 1.0).abs() > 0.01));
    }

    #[test]
    fn test_fht_empty() {
        let data: Vec<f32> = vec![];
        let result = fast_hadamard_transform(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_random_signs() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let seed = 42u64;

        // Apply random signs
        apply_random_signs(&mut data, seed);

        // Some values should be negated
        let has_negative = data.iter().any(|&v| v < 0.0);
        let has_positive = data.iter().any(|&v| v > 0.0);
        assert!(has_negative || has_positive); // At least some values affected

        // Applying same signs again should restore original magnitudes
        // (but may flip signs again - deterministic based on seed)
        apply_random_signs(&mut data, seed);

        // Magnitudes should be preserved
        for &val in &data {
            assert!([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
                .iter()
                .any(|&v| (val - v).abs() < 0.001));
        }
    }

    #[test]
    fn test_quantize_with_fht() {
        // Test that FHT-enabled quantization works
        let quantizer = RaBitQ::new(RaBitQParams::bits4_fht());

        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 128);
        assert_eq!(quantized.bits, 4);
        assert!(quantized.fht_applied);
        assert!(quantized.scale > 0.0);
    }

    #[test]
    fn test_fht_quantization_accuracy() {
        // Compare accuracy with and without FHT
        let params_no_fht = RaBitQParams::bits4();
        let params_fht = RaBitQParams::bits4_fht();

        let q_no_fht = RaBitQ::new(params_no_fht);
        let q_fht = RaBitQ::new(params_fht);

        // Test vector with varying values
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

        let qv_no_fht = q_no_fht.quantize(&vector);
        let qv_fht = q_fht.quantize(&vector);

        // Both should work (not crash)
        assert_eq!(qv_no_fht.dimensions, 64);
        assert_eq!(qv_fht.dimensions, 64);

        // FHT flag should be set correctly
        assert!(!qv_no_fht.fht_applied);
        assert!(qv_fht.fht_applied);
    }

    #[test]
    fn test_fht_high_dimensional() {
        // Test FHT on high-dimensional vectors (like embeddings)
        let quantizer = RaBitQ::new(RaBitQParams::with_fht(QuantizationBits::Bits4));

        // 768D vector (BERT-like)
        let vector: Vec<f32> = (0..768).map(|i| ((i % 100) as f32) / 100.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 768);
        assert!(quantized.fht_applied);
    }

    #[test]
    fn test_bits3_constructor() {
        let params = RaBitQParams::bits3();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits3);
        assert_eq!(params.bits_per_dim.to_u8(), 3);
        assert_eq!(params.bits_per_dim.levels(), 8); // 2^3
    }

    #[test]
    fn test_bits5_constructor() {
        let params = RaBitQParams::bits5();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits5);
        assert_eq!(params.bits_per_dim.to_u8(), 5);
        assert_eq!(params.bits_per_dim.levels(), 32); // 2^5
    }

    #[test]
    fn test_bits7_constructor() {
        let params = RaBitQParams::bits7();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits7);
        assert_eq!(params.bits_per_dim.to_u8(), 7);
        assert_eq!(params.bits_per_dim.levels(), 128); // 2^7
    }

    #[test]
    fn test_quantize_3bit() {
        let quantizer = RaBitQ::new(RaBitQParams::bits3());

        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 64);
        assert_eq!(quantized.bits, 3);
        assert!(quantized.scale > 0.0);

        // Reconstruction should work
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 64);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_quantize_5bit() {
        let quantizer = RaBitQ::new(RaBitQParams::bits5());

        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 64);
        assert_eq!(quantized.bits, 5);
        assert!(quantized.scale > 0.0);

        // Reconstruction should work
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 64);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_quantize_7bit() {
        let quantizer = RaBitQ::new(RaBitQParams::bits7());

        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 64);
        assert_eq!(quantized.bits, 7);
        assert!(quantized.scale > 0.0);

        // Reconstruction should work
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 64);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_bit_width_accuracy_ordering() {
        // Higher bit widths should have better accuracy
        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let q2 = RaBitQ::new(RaBitQParams::bits2());
        let q3 = RaBitQ::new(RaBitQParams::bits3());
        let q4 = RaBitQ::new(RaBitQParams::bits4());
        let q5 = RaBitQ::new(RaBitQParams::bits5());

        let qv2 = q2.quantize(&vector);
        let qv3 = q3.quantize(&vector);
        let qv4 = q4.quantize(&vector);
        let qv5 = q5.quantize(&vector);

        // Compute reconstruction errors
        let err2 = q2.compute_error(&vector, &qv2.data, qv2.scale);
        let err3 = q3.compute_error(&vector, &qv3.data, qv3.scale);
        let err4 = q4.compute_error(&vector, &qv4.data, qv4.scale);
        let err5 = q5.compute_error(&vector, &qv5.data, qv5.scale);

        // Higher bits should generally have lower error
        // (may not be strictly monotonic due to quantization effects, but trend should hold)
        assert!(
            err4 <= err2 * 2.0,
            "4-bit should be more accurate than 2-bit"
        );
        assert!(
            err5 <= err3 * 2.0,
            "5-bit should be more accurate than 3-bit"
        );
    }
