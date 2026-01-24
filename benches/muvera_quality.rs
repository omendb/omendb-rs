//! MUVERA quality benchmark (MUV-17)
//!
//! Compares MUVERA retrieval quality vs brute-force MaxSim.
//!
//! Run: cargo bench --bench muvera_quality

use omendb::vector::muvera::{maxsim, MultiVectorConfig, MuveraEncoder};
use rand::prelude::*;
use std::time::Instant;

/// Generate random normalized tokens.
fn random_tokens(rng: &mut StdRng, num_tokens: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..num_tokens)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            normalize(&v)
        })
        .collect()
}

/// L2-normalize a vector.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Compute recall@k for MUVERA vs brute-force MaxSim.
fn compute_recall(
    ground_truth: &[usize], // Top-k from brute-force
    predictions: &[usize],  // Top-k from MUVERA
    k: usize,
) -> f32 {
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
    let pred_set: std::collections::HashSet<_> = predictions.iter().take(k).collect();
    let intersection = gt_set.intersection(&pred_set).count();
    intersection as f32 / k as f32
}

/// Brute-force MaxSim search (ground truth).
fn brute_force_maxsim(query: &[&[f32]], docs: &[Vec<&[f32]>], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, maxsim(query, doc)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

/// FDE-based search (MUVERA without reranking).
fn fde_search(query_fde: &[f32], doc_fdes: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = doc_fdes
        .iter()
        .enumerate()
        .map(|(i, doc_fde)| (i, dot(query_fde, doc_fde)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

/// FDE-based search with MaxSim reranking.
fn fde_search_rerank(
    query: &[&[f32]],
    query_fde: &[f32],
    docs: &[Vec<&[f32]>],
    doc_fdes: &[Vec<f32>],
    k: usize,
    rerank_factor: usize,
) -> Vec<usize> {
    // Step 1: Get candidates from FDE search
    let num_candidates = k * rerank_factor;
    let candidates = fde_search(query_fde, doc_fdes, num_candidates);

    // Step 2: Rerank with MaxSim
    let mut reranked: Vec<(usize, f32)> = candidates
        .into_iter()
        .map(|i| (i, maxsim(query, &docs[i])))
        .collect();
    reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    reranked.into_iter().take(k).map(|(i, _)| i).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn main() {
    println!("MUVERA Quality Benchmark (MUV-17)");
    println!("=================================\n");

    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    // Parameters
    let num_docs = 1000; // Reduced for faster benchmark
    let tokens_per_doc = 50; // Average tokens per document
    let query_tokens = 10; // Query tokens
    let dim = 128; // Token dimension
    let k = 10; // Top-k
    let num_queries = 50; // Number of test queries
    let rerank_factor = 4;

    // Config - higher params for better quality
    let config = MultiVectorConfig {
        repetitions: 10,
        partition_bits: 4,
        seed: 42,
    };
    let encoder = MuveraEncoder::new(dim, config);

    println!("Configuration:");
    println!("  Documents: {num_docs}");
    println!("  Tokens/doc: ~{tokens_per_doc}");
    println!("  Query tokens: {query_tokens}");
    println!("  Dimension: {dim}");
    println!("  FDE dimension: {}", encoder.fde_dimension());
    println!("  k: {k}");
    println!("  Rerank factor: {rerank_factor}");
    println!();

    // Generate documents
    print!("Generating {num_docs} documents... ");
    let start = Instant::now();
    let docs: Vec<Vec<Vec<f32>>> = (0..num_docs)
        .map(|_| {
            let num_tokens = rng.gen_range(tokens_per_doc / 2..tokens_per_doc * 3 / 2);
            random_tokens(&mut rng, num_tokens, dim)
        })
        .collect();
    let docs_refs: Vec<Vec<&[f32]>> = docs
        .iter()
        .map(|doc| doc.iter().map(|t| t.as_slice()).collect())
        .collect();
    println!("done ({:.2}s)", start.elapsed().as_secs_f32());

    // Encode documents to FDEs
    print!("Encoding documents to FDEs... ");
    let start = Instant::now();
    let doc_fdes: Vec<Vec<f32>> = docs_refs
        .iter()
        .map(|doc| encoder.encode_document(doc))
        .collect();
    let encode_time = start.elapsed().as_secs_f32();
    println!(
        "done ({:.2}s, {:.0} docs/s)",
        encode_time,
        num_docs as f32 / encode_time
    );

    // Generate REALISTIC queries by sampling tokens from existing documents
    // This simulates ColBERT-style retrieval where queries share vocabulary with docs
    print!("Generating {num_queries} queries (sampled from docs)... ");
    let queries: Vec<Vec<Vec<f32>>> = (0..num_queries)
        .map(|_| {
            // Pick a random document to base query on
            let doc_idx = rng.gen_range(0..num_docs);
            let doc = &docs[doc_idx];

            // Sample query_tokens from the document (with some noise)
            let mut query_vecs = Vec::with_capacity(query_tokens);
            for _ in 0..query_tokens {
                let token_idx = rng.gen_range(0..doc.len());
                // Add small noise to the token
                let noisy: Vec<f32> = doc[token_idx]
                    .iter()
                    .map(|&v| v + (rng.gen::<f32>() - 0.5) * 0.2)
                    .collect();
                query_vecs.push(normalize(&noisy));
            }
            query_vecs
        })
        .collect();
    println!("done");

    let query_refs: Vec<Vec<&[f32]>> = queries
        .iter()
        .map(|q| q.iter().map(|t| t.as_slice()).collect())
        .collect();
    let query_fdes: Vec<Vec<f32>> = query_refs.iter().map(|q| encoder.encode_query(q)).collect();

    println!("\nRunning {num_queries} queries...\n");

    // Compute recalls
    let mut fde_recalls = Vec::new();
    let mut rerank_recalls = Vec::new();

    for i in 0..num_queries {
        let query = &query_refs[i];
        let query_fde = &query_fdes[i];

        // Ground truth: brute-force MaxSim
        let ground_truth = brute_force_maxsim(query, &docs_refs, k);

        // MUVERA-only (FDE search)
        let fde_results = fde_search(query_fde, &doc_fdes, k);
        let fde_recall = compute_recall(&ground_truth, &fde_results, k);
        fde_recalls.push(fde_recall);

        // MUVERA + reranking
        let rerank_results =
            fde_search_rerank(query, query_fde, &docs_refs, &doc_fdes, k, rerank_factor);
        let rerank_recall = compute_recall(&ground_truth, &rerank_results, k);
        rerank_recalls.push(rerank_recall);
    }

    // Compute averages
    let avg_fde_recall: f32 = fde_recalls.iter().sum::<f32>() / num_queries as f32;
    let avg_rerank_recall: f32 = rerank_recalls.iter().sum::<f32>() / num_queries as f32;

    println!("Results (Recall@{k}):");
    println!("  MUVERA-only (FDE):     {:.1}%", avg_fde_recall * 100.0);
    println!(
        "  MUVERA + rerank ({}x): {:.1}%",
        rerank_factor,
        avg_rerank_recall * 100.0
    );
    println!();

    // Timing comparison
    println!("Search latency (single query):");

    let start = Instant::now();
    for i in 0..num_queries {
        let _ = brute_force_maxsim(&query_refs[i], &docs_refs, k);
    }
    let brute_time = start.elapsed().as_secs_f32() / num_queries as f32;

    let start = Instant::now();
    for i in 0..num_queries {
        let _ = fde_search(&query_fdes[i], &doc_fdes, k);
    }
    let fde_time = start.elapsed().as_secs_f32() / num_queries as f32;

    let start = Instant::now();
    for i in 0..num_queries {
        let _ = fde_search_rerank(
            &query_refs[i],
            &query_fdes[i],
            &docs_refs,
            &doc_fdes,
            k,
            rerank_factor,
        );
    }
    let rerank_time = start.elapsed().as_secs_f32() / num_queries as f32;

    println!("  Brute-force MaxSim: {:.3}ms", brute_time * 1000.0);
    println!("  MUVERA-only (FDE):  {:.3}ms", fde_time * 1000.0);
    println!("  MUVERA + rerank:    {:.3}ms", rerank_time * 1000.0);
    println!("  Speedup (FDE vs brute): {:.1}x", brute_time / fde_time);
    println!(
        "  Speedup (rerank vs brute): {:.1}x",
        brute_time / rerank_time
    );

    // Validate expectations
    // Note: MUVERA paper reports 70% quality for retrieval tasks with semantic data.
    // With random synthetic data, recall is lower because FDE approximates MaxSim
    // which requires semantic similarity to work well.
    println!("\n=== Analysis ===");
    println!("Note: MUVERA is designed for semantic retrieval (ColBERT, etc.)");
    println!("Random synthetic data shows lower recall because:");
    println!("  1. No semantic structure in random vectors");
    println!("  2. FDE approximation quality depends on query-doc similarity distribution");
    println!();

    // The key metric is that reranking improves recall
    let improvement = avg_rerank_recall / avg_fde_recall;
    println!("Key findings:");
    println!("  Rerank improvement: {:.1}x", improvement);
    if improvement > 1.2 {
        println!("  ✓ Reranking provides meaningful improvement");
    }

    // FDE provides speedup over brute force
    if brute_time / fde_time > 1.0 {
        println!(
            "  ✓ FDE search is {:.1}x faster than brute-force",
            brute_time / fde_time
        );
    }
}
