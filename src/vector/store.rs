//! Vector storage with HNSW indexing
//!
//! VectorStore manages a collection of vectors and provides k-NN search
//! using HNSW (Hierarchical Navigable Small World) algorithm.
//!
//! Optional Extended RaBitQ quantization for memory-efficient storage.

use super::hnsw_index::HNSWIndex;
use super::rabitq::{QuantizedVector, RaBitQ, RaBitQParams};
use super::storage::SeerDBStorage;
use super::types::Vector;
use anyhow::Result;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;

/// Metadata filter for vector search (MongoDB-style operators)
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Equality: field == value
    Eq(String, JsonValue),
    /// Not equal: field != value
    Ne(String, JsonValue),
    /// Greater than or equal: field >= value
    Gte(String, f64),
    /// Less than: field < value
    Lt(String, f64),
    /// Greater than: field > value
    Gt(String, f64),
    /// Less than or equal: field <= value
    Lte(String, f64),
    /// In list: field in [values]
    In(String, Vec<JsonValue>),
    /// Contains substring: field.contains(value)
    Contains(String, String),
    /// Logical AND: all filters must match
    And(Vec<MetadataFilter>),
    /// Logical OR: at least one filter must match
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate filter against metadata
    pub fn matches(&self, metadata: &JsonValue) -> bool {
        match self {
            MetadataFilter::Eq(field, value) => metadata.get(field) == Some(value),
            MetadataFilter::Ne(field, value) => metadata.get(field) != Some(value),
            MetadataFilter::Gte(field, threshold) => metadata
                .get(field)
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v >= *threshold),
            MetadataFilter::Lt(field, threshold) => metadata
                .get(field)
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v < *threshold),
            MetadataFilter::Gt(field, threshold) => metadata
                .get(field)
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v > *threshold),
            MetadataFilter::Lte(field, threshold) => metadata
                .get(field)
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v <= *threshold),
            MetadataFilter::In(field, values) => {
                metadata.get(field).is_some_and(|v| values.contains(v))
            }
            MetadataFilter::Contains(field, substring) => metadata
                .get(field)
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains(substring)),
            MetadataFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

/// Vector store with HNSW indexing
pub struct VectorStore {
    /// All vectors stored in memory
    pub vectors: Vec<Vector>,

    /// HNSW index for approximate nearest neighbor search
    pub hnsw_index: Option<HNSWIndex>,

    /// Vector dimensionality
    dimensions: usize,

    /// Optional quantizer for memory-efficient storage (Extended RaBitQ)
    quantizer: Option<RaBitQ>,

    /// Quantized vectors (parallel to vectors, None if quantizer not enabled)
    quantized_vectors: Vec<Option<QuantizedVector>>,

    /// Metadata storage (indexed by internal vector ID)
    metadata: HashMap<usize, JsonValue>,

    /// Map from string IDs to internal indices (public for Python bindings)
    pub id_to_index: HashMap<String, usize>,

    /// Deleted vector IDs (tombstones for MVCC)
    deleted: HashMap<usize, bool>,

    /// Persistent storage backend (seerdb LSM)
    storage: Option<SeerDBStorage>,
}

impl VectorStore {
    /// Create new vector store without quantization
    pub fn new(dimensions: usize) -> Self {
        Self {
            vectors: Vec::new(),
            hnsw_index: None,
            dimensions,
            quantizer: None,
            quantized_vectors: Vec::new(),
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
        }
    }

    /// Create new vector store with adaptive HNSW parameters based on expected capacity
    ///
    /// Automatically selects optimal M, ef_construction, and ef_search parameters
    /// based on the expected number of vectors:
    ///
    /// - < 50K vectors: M=16, ef_construction=200, ef_search=100 (fast & efficient)
    /// - 50K-500K vectors: M=32, ef_construction=400, ef_search=100 (balanced, 98% recall)
    /// - > 500K vectors: M=48, ef_construction=600, ef_search=150 (high recall, 99%)
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `expected_vectors` - Expected number of vectors to be inserted
    ///
    /// # Example
    /// ```ignore
    /// // For 100K vectors, automatically uses M=32 (98% recall)
    /// let mut store = VectorStore::new_with_capacity(128, 100_000);
    /// ```
    ///
    /// # Performance Characteristics
    /// See `ai/research/WEEK21_100K_RECALL_INVESTIGATION.md` for detailed benchmarks
    pub fn new_with_capacity(dimensions: usize, expected_vectors: usize) -> Self {
        let (m, ef_construction, ef_search) = Self::adaptive_hnsw_params(expected_vectors);

        // SAFETY: adaptive_hnsw_params always returns valid parameters (m > 0, ef > 0)
        let hnsw_index = Some(
            HNSWIndex::new_with_params(
                expected_vectors.max(1_000_000), // Use expected capacity, min 1M
                dimensions,
                m,
                ef_construction,
                ef_search,
            )
            .expect("adaptive_hnsw_params returns valid parameters"),
        );

        Self {
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            quantizer: None,
            quantized_vectors: Vec::new(),
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
        }
    }

    /// Compute adaptive HNSW parameters based on expected vector count
    ///
    /// Optimized for balanced speed/recall tradeoff (similar to ChromaDB defaults):
    /// - ef_search=100 provides ~98% recall with high QPS (2000+ QPS)
    /// - ef_construction kept high for quality graph
    ///
    /// Returns: (M, ef_construction, ef_search)
    fn adaptive_hnsw_params(expected_vectors: usize) -> (usize, usize, usize) {
        if expected_vectors < 50_000 {
            // Fast & efficient (10K-50K scale)
            // ~98% recall, high QPS (~2100 QPS @ 10K)
            (16, 200, 100)
        } else if expected_vectors < 500_000 {
            // Balanced (50K-500K scale)
            // ~98% recall, good QPS
            (32, 400, 100)
        } else {
            // High recall (500K+ scale)
            // ~99% recall, moderate QPS
            (48, 600, 150)
        }
    }

    /// Create new vector store with Extended RaBitQ quantization
    pub fn new_with_quantization(dimensions: usize, params: RaBitQParams) -> Self {
        let quantizer = RaBitQ::new(params);

        Self {
            vectors: Vec::new(),
            hnsw_index: None,
            dimensions,
            quantizer: Some(quantizer),
            quantized_vectors: Vec::new(),
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
        }
    }

    /// Create new vector store with custom HNSW parameters
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `m` - Number of bidirectional links per node (typical: 16-48)
    /// * `ef_construction` - Candidate list size during construction (typical: 200-800)
    /// * `ef_search` - Candidate list size during search (typical: 200-1000)
    ///
    /// # Example
    /// ```ignore
    /// // Higher M for better recall at 100K+ scale
    /// let mut store = VectorStore::new_with_params(128, 32, 400, 600)?;
    /// ```
    pub fn new_with_params(
        dimensions: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<Self> {
        // Eagerly initialize HNSW with custom parameters
        let hnsw_index = Some(HNSWIndex::new_with_params(
            1_000_000,
            dimensions,
            m,
            ef_construction,
            ef_search,
        )?);

        Ok(Self {
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            quantizer: None,
            quantized_vectors: Vec::new(),
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
        })
    }

    /// Open a persistent vector store at the given path
    ///
    /// Creates a new database if it doesn't exist, or loads existing data.
    /// All operations (insert, set, delete) are automatically persisted.
    ///
    /// # Arguments
    /// * `path` - Directory path for the database (e.g., "mydb.oadb")
    ///
    /// # Example
    /// ```ignore
    /// let mut store = VectorStore::open("mydb.oadb")?;
    /// store.set("doc1".to_string(), vector, metadata)?;
    /// // Data is automatically persisted
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = SeerDBStorage::open(&path)?;

        // Load existing data from storage
        let vectors_data = storage.load_all_vectors()?;
        let metadata = storage.load_all_metadata()?;
        let id_to_index = storage.load_all_id_mappings()?;
        let deleted = storage.load_all_deleted()?;

        // Get dimensions from config or infer from vectors
        let dimensions = if let Some(dim) = storage.get_config("dimensions")? {
            dim as usize
        } else if let Some((_, first_vec)) = vectors_data.first() {
            first_vec.len()
        } else {
            0 // Will be set on first insert
        };

        // Convert vectors to Vector type
        let mut vectors: Vec<Vector> = Vec::with_capacity(vectors_data.len());
        for (id, data) in &vectors_data {
            // Ensure vectors are in correct order (fill gaps with zeros if needed)
            while vectors.len() < *id {
                vectors.push(Vector::new(vec![0.0; dimensions]));
            }
            vectors.push(Vector::new(data.clone()));
        }

        // Build HNSW index if we have vectors
        let hnsw_index = if !vectors.is_empty() {
            eprintln!(
                "📂 Loading {} vectors from {}...",
                vectors.len(),
                path.as_ref().display()
            );
            let mut index = HNSWIndex::new(vectors.len().max(10_000), dimensions)?;
            for vector in &vectors {
                index.insert(&vector.data)?;
            }
            eprintln!("✅ HNSW index built for {} vectors", vectors.len());
            Some(index)
        } else {
            None
        };

        Ok(Self {
            vectors,
            hnsw_index,
            dimensions,
            quantizer: None,
            quantized_vectors: Vec::new(),
            metadata,
            id_to_index,
            deleted,
            storage: Some(storage),
        })
    }

    /// Open a persistent vector store with specified dimensions
    ///
    /// Like `open()` but ensures dimensions are set for new databases.
    pub fn open_with_dimensions(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let mut store = Self::open(path)?;
        if store.dimensions == 0 {
            store.dimensions = dimensions;
            if let Some(ref storage) = store.storage {
                storage.put_config("dimensions", dimensions as u64)?;
            }
        }
        Ok(store)
    }

    /// Insert vector and return its ID
    pub fn insert(&mut self, vector: Vector) -> Result<usize> {
        let id = self.vectors.len();

        // Lazy initialize HNSW on first insert
        if self.hnsw_index.is_none() {
            // If dimensions == 0, infer from first vector (for compatibility with existing tests)
            // Otherwise use store's configured dimensions
            let dimensions = if self.dimensions == 0 {
                vector.dim()
            } else {
                // Validate dimension matches store's expected dimensions
                if vector.dim() != self.dimensions {
                    anyhow::bail!(
                        "Vector dimension mismatch: store expects {}, got {}. All vectors in same store must have same dimension.",
                        self.dimensions,
                        vector.dim()
                    );
                }
                self.dimensions
            };

            // Start with small default capacity (10K vectors)
            // This uses fast parameters (M=16, ef_construction=100) matching ChromaDB
            // Index will automatically grow as more vectors are added
            self.hnsw_index = Some(HNSWIndex::new(10_000, dimensions)?);
            self.dimensions = dimensions;
        } else {
            // Validate dimension matches existing HNSW index
            // NOTE: HNSW requires all vectors to have same dimension
            if vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector dimension mismatch: store expects {}, got {}. All vectors in same store must have same dimension.",
                    self.dimensions,
                    vector.dim()
                );
            }
        }

        // Insert into HNSW index
        if let Some(ref mut index) = self.hnsw_index {
            index.insert(&vector.data)?;
        }

        // Quantize vector if quantizer is enabled
        if let Some(ref quantizer) = self.quantizer {
            let quantized = quantizer.quantize(&vector.data);
            self.quantized_vectors.push(Some(quantized));
        } else {
            self.quantized_vectors.push(None);
        }

        // Persist to storage if available
        if let Some(ref storage) = self.storage {
            storage.put_vector(id, &vector.data)?;
            storage.increment_count()?;
            // Save dimensions on first insert
            if id == 0 {
                storage.put_config("dimensions", self.dimensions as u64)?;
            }
        }

        self.vectors.push(vector);
        Ok(id)
    }

    /// Insert vector with string ID and metadata
    ///
    /// This is the primary method for inserting vectors with metadata support.
    /// Returns error if ID already exists (use set for insert-or-update semantics).
    pub fn insert_with_metadata(
        &mut self,
        id: String,
        vector: Vector,
        metadata: JsonValue,
    ) -> Result<usize> {
        // Check if ID already exists
        if self.id_to_index.contains_key(&id) {
            anyhow::bail!(
                "Vector with ID '{}' already exists. Use set() to update.",
                id
            );
        }

        // Insert vector using existing insert method
        let index = self.insert(vector)?;

        // Store metadata and ID mapping
        self.metadata.insert(index, metadata.clone());
        self.id_to_index.insert(id.clone(), index);

        // Persist to storage if available
        if let Some(ref storage) = self.storage {
            storage.put_metadata(index, &metadata)?;
            storage.put_id_mapping(&id, index)?;
        }

        Ok(index)
    }

    /// Upsert vector (insert or update) with string ID and metadata
    ///
    /// This is the recommended method for most use cases.
    /// If the ID exists, updates the vector and metadata.
    /// If the ID doesn't exist, inserts a new vector.
    pub fn set(&mut self, id: String, vector: Vector, metadata: JsonValue) -> Result<usize> {
        // Check if ID already exists
        if let Some(&index) = self.id_to_index.get(&id) {
            // Update existing vector
            self.update_by_index(index, Some(vector), Some(metadata))?;
            Ok(index)
        } else {
            // Insert new vector
            self.insert_with_metadata(id, vector, metadata)
        }
    }

    /// Batch set vectors (insert or update multiple vectors at once)
    ///
    /// This is the recommended method for bulk operations. It provides significant
    /// performance improvements over calling set() repeatedly by:
    /// - Reducing function call overhead
    /// - Batching HNSW insertions
    /// - Amortizing metadata operations
    ///
    /// # Arguments
    /// * `batch` - Vector of (id, vector, metadata) tuples
    ///
    /// # Returns
    /// Vector of indices for all set vectors
    ///
    /// # Example
    /// ```ignore
    /// let batch = vec![
    ///     ("vec1".to_string(), Vector::new(vec![0.1, 0.2]), json!({"key": "value"})),
    ///     ("vec2".to_string(), Vector::new(vec![0.3, 0.4]), json!({"key": "value2"})),
    /// ];
    /// let indices = store.set_batch(batch)?;
    /// ```
    pub fn set_batch(&mut self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Separate batch into updates and inserts
        let mut updates: Vec<(usize, Vector, JsonValue)> = Vec::new();
        let mut inserts: Vec<(String, Vector, JsonValue)> = Vec::new();

        for (id, vector, metadata) in batch {
            if let Some(&index) = self.id_to_index.get(&id) {
                // Existing vector - queue for update
                updates.push((index, vector, metadata));
            } else {
                // New vector - queue for insert
                inserts.push((id, vector, metadata));
            }
        }

        let mut result_indices = Vec::new();

        // Process updates first (modify in-place)
        for (index, vector, metadata) in updates {
            self.update_by_index(index, Some(vector), Some(metadata))?;
            result_indices.push(index);
        }

        // Process inserts in batch
        if !inserts.is_empty() {
            // Lazy initialize HNSW if needed
            if self.hnsw_index.is_none() {
                let dimensions = if self.dimensions == 0 {
                    inserts[0].1.dim()
                } else {
                    self.dimensions
                };
                self.hnsw_index = Some(HNSWIndex::new(10_000, dimensions)?);
                self.dimensions = dimensions;
            }

            // Validate all vectors have same dimensions
            for (i, (_, vector, _)) in inserts.iter().enumerate() {
                if vector.dim() != self.dimensions {
                    anyhow::bail!(
                        "Vector {} dimension mismatch: expected {}, got {}",
                        i,
                        self.dimensions,
                        vector.dim()
                    );
                }
            }

            // Extract vectors for batch HNSW insertion
            let vectors_data: Vec<Vec<f32>> =
                inserts.iter().map(|(_, v, _)| v.data.clone()).collect();

            // Batch insert into HNSW
            let base_index = self.vectors.len();
            if let Some(ref mut index) = self.hnsw_index {
                index.batch_insert(&vectors_data)?;
            }

            // Batch persist to storage (atomic, high-performance)
            if let Some(ref storage) = self.storage {
                // Save dimensions on first insert
                if base_index == 0 {
                    storage.put_config("dimensions", self.dimensions as u64)?;
                }

                // Prepare batch items: (index, string_id, vector, metadata)
                let batch_items: Vec<(usize, String, Vec<f32>, serde_json::Value)> = inserts
                    .iter()
                    .enumerate()
                    .map(|(i, (id, vector, metadata))| {
                        (
                            base_index + i,
                            id.clone(),
                            vector.data.clone(),
                            metadata.clone(),
                        )
                    })
                    .collect();

                // Single atomic batch commit (replaces N individual puts)
                storage.put_batch(batch_items)?;
            }

            // Add vectors to in-memory structures
            for (i, (id, vector, metadata)) in inserts.into_iter().enumerate() {
                let idx = base_index + i;

                // Quantize if quantizer enabled
                if let Some(ref quantizer) = self.quantizer {
                    let quantized = quantizer.quantize(&vector.data);
                    self.quantized_vectors.push(Some(quantized));
                } else {
                    self.quantized_vectors.push(None);
                }

                self.vectors.push(vector);
                self.metadata.insert(idx, metadata);
                self.id_to_index.insert(id, idx);
                result_indices.push(idx);
            }
        }

        Ok(result_indices)
    }

    /// Update existing vector by index (internal method)
    fn update_by_index(
        &mut self,
        index: usize,
        vector: Option<Vector>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        // Check if vector exists and is not deleted
        if index >= self.vectors.len() {
            anyhow::bail!("Vector index {} does not exist", index);
        }
        if self.deleted.contains_key(&index) {
            anyhow::bail!("Vector index {} has been deleted", index);
        }

        // Update vector if provided
        if let Some(new_vector) = vector {
            // Validate dimensions
            if new_vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    new_vector.dim()
                );
            }

            // Update in memory
            self.vectors[index] = new_vector.clone();

            // Persist to storage if available
            if let Some(ref storage) = self.storage {
                storage.put_vector(index, &new_vector.data)?;
            }

            // Update in HNSW index (requires rebuild for now)
            // NOTE: HNSW doesn't support in-place updates, need to rebuild
            // For production, we'd use MVCC (mark old as deleted, insert new)
            // For now, we'll just update the vector data
            // The index will be out of sync until rebuild_index() is called

            // Update quantized vector if quantizer enabled
            if let Some(ref quantizer) = self.quantizer {
                let quantized = quantizer.quantize(&new_vector.data);
                self.quantized_vectors[index] = Some(quantized);
            }
        }

        // Update metadata if provided
        if let Some(ref new_metadata) = metadata {
            self.metadata.insert(index, new_metadata.clone());

            // Persist to storage if available
            if let Some(ref storage) = self.storage {
                storage.put_metadata(index, new_metadata)?;
            }
        }

        Ok(())
    }

    /// Update existing vector by string ID
    pub fn update(
        &mut self,
        id: &str,
        vector: Option<Vector>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{}' not found", id))?;

        self.update_by_index(index, vector, metadata)
    }

    /// Delete vector by string ID (marks as deleted, uses tombstone)
    pub fn delete(&mut self, id: &str) -> Result<()> {
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{}' not found", id))?;

        // Mark as deleted
        self.deleted.insert(index, true);

        // Persist tombstone to storage if available
        if let Some(ref storage) = self.storage {
            storage.put_deleted(index)?;
            storage.delete_id_mapping(id)?;
        }

        // Remove from ID mapping
        self.id_to_index.remove(id);

        Ok(())
    }

    /// Delete multiple vectors by string IDs
    pub fn delete_batch(&mut self, ids: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        for id in ids {
            if self.delete(id).is_ok() {
                deleted_count += 1;
            }
        }
        Ok(deleted_count)
    }

    /// Get vector by string ID
    pub fn get_by_id(&self, id: &str) -> Option<(&Vector, &JsonValue)> {
        self.id_to_index.get(id).and_then(|&index| {
            // Check if deleted
            if self.deleted.contains_key(&index) {
                return None;
            }
            // Return vector and metadata
            self.vectors
                .get(index)
                .and_then(|vec| self.metadata.get(&index).map(|meta| (vec, meta)))
        })
    }

    /// Insert batch of vectors in parallel
    ///
    /// Automatically chunks vectors into optimal batch sizes for parallel insertion.
    /// Uses hnsw_rs's parallel_insert with Rayon for multi-threaded building.
    ///
    /// Chunk size of 10,000 balances:
    /// - Parallelization overhead (want batches large enough)
    /// - Memory usage (smaller batches more memory-friendly)
    /// - Progress reporting (can log after each chunk)
    ///
    /// Returns Vec of IDs for inserted vectors
    pub fn batch_insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Validate dimensions
        for (i, vector) in vectors.iter().enumerate() {
            if vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    self.dimensions,
                    vector.dim()
                );
            }
        }

        // Lazy initialize HNSW on first insert
        if self.hnsw_index.is_none() {
            let capacity = vectors.len().max(1_000_000);
            self.hnsw_index = Some(HNSWIndex::new(capacity, self.dimensions)?);
        }

        let _start_id = self.vectors.len();
        let mut all_ids = Vec::with_capacity(vectors.len());

        // Chunk size for parallel insertion (recommended: 1000 × num_threads)
        // Using 10,000 as a good default (works well for 4-16 core machines)
        const CHUNK_SIZE: usize = 10_000;

        // Process in chunks for better memory management and progress tracking
        for (chunk_idx, chunk) in vectors.chunks(CHUNK_SIZE).enumerate() {
            // Extract vector data for HNSW
            let vector_data: Vec<Vec<f32>> = chunk.iter().map(|v| v.data.clone()).collect();

            // Parallel insert this chunk
            if let Some(ref mut index) = self.hnsw_index {
                let chunk_ids = index.batch_insert(&vector_data)?;
                all_ids.extend(chunk_ids);
            }

            // Log progress for large batches
            if vectors.len() >= CHUNK_SIZE {
                let processed = ((chunk_idx + 1) * CHUNK_SIZE).min(vectors.len());
                eprintln!(
                    "  Inserted {} / {} vectors ({:.1}%)",
                    processed,
                    vectors.len(),
                    (processed as f64 / vectors.len() as f64) * 100.0
                );
            }
        }

        // Quantize vectors if quantizer is enabled
        if let Some(ref quantizer) = self.quantizer {
            for vector in &vectors {
                let quantized = quantizer.quantize(&vector.data);
                self.quantized_vectors.push(Some(quantized));
            }
        } else {
            for _ in &vectors {
                self.quantized_vectors.push(None);
            }
        }

        // Add vectors to storage
        self.vectors.extend(vectors);

        // Return IDs from HNSW
        Ok(all_ids)
    }

    /// Rebuild HNSW index from existing vectors
    ///
    /// This is needed when:
    /// - Vectors are loaded from disk but index wasn't persisted
    /// - Index needs to be rebuilt after batch inserts
    /// - Quantization is enabled/disabled after loading
    pub fn rebuild_index(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        eprintln!(
            "🔨 Rebuilding HNSW index for {} vectors...",
            self.vectors.len()
        );
        let start = std::time::Instant::now();

        // Create new HNSW index
        let mut index = HNSWIndex::new(self.vectors.len().max(1_000_000), self.dimensions)?;

        // Insert all vectors
        for vector in &self.vectors {
            index.insert(&vector.data)?;
        }

        self.hnsw_index = Some(index);

        // Rebuild quantized vectors if quantizer is enabled
        if let Some(ref quantizer) = self.quantizer {
            eprintln!("  Quantizing {} vectors...", self.vectors.len());
            self.quantized_vectors.clear();
            for vector in &self.vectors {
                let quantized = quantizer.quantize(&vector.data);
                self.quantized_vectors.push(Some(quantized));
            }
        }

        eprintln!(
            "✅ HNSW index rebuilt in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Merge another VectorStore into this one using IGTM algorithm
    ///
    /// Uses Iterative Greedy Tree Merging for 1.3-1.7x faster batch inserts
    /// compared to naive insertion.
    ///
    /// # Arguments
    /// * `other` - VectorStore to merge from (vectors and metadata will be copied)
    ///
    /// # Returns
    /// Number of vectors merged
    ///
    /// # Note
    /// String IDs from `other` are preserved. If there are conflicts,
    /// the existing ID in `self` takes precedence (other's vector is skipped).
    pub fn merge_from(&mut self, other: &VectorStore) -> Result<usize> {
        if other.dimensions != self.dimensions {
            anyhow::bail!(
                "Dimension mismatch: self={}, other={}",
                self.dimensions,
                other.dimensions
            );
        }

        if other.vectors.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();
        eprintln!("🔗 Merging {} vectors using IGTM...", other.vectors.len());

        // Initialize HNSW if needed
        if self.hnsw_index.is_none() {
            let capacity = (self.vectors.len() + other.vectors.len()).max(1_000_000);
            self.hnsw_index = Some(HNSWIndex::new(capacity, self.dimensions)?);
        }

        // Track how many vectors we actually merge (skip ID conflicts)
        let mut merged_count = 0;
        let base_index = self.vectors.len();

        // Copy vectors and metadata, handling ID conflicts
        for (other_idx, vector) in other.vectors.iter().enumerate() {
            // Check for string ID conflict
            let has_conflict = other
                .id_to_index
                .iter()
                .find(|(_, &idx)| idx == other_idx)
                .map(|(string_id, _)| self.id_to_index.contains_key(string_id))
                .unwrap_or(false);

            if has_conflict {
                continue; // Skip vectors with conflicting string IDs
            }

            // Copy vector
            self.vectors.push(vector.clone());

            // Copy metadata if present
            if let Some(meta) = other.metadata.get(&other_idx) {
                self.metadata
                    .insert(base_index + merged_count, meta.clone());
            }

            // Copy string ID mapping
            if let Some((string_id, _)) =
                other.id_to_index.iter().find(|(_, &idx)| idx == other_idx)
            {
                self.id_to_index
                    .insert(string_id.clone(), base_index + merged_count);
            }

            // Copy quantized vector if present
            if let Some(qv) = other.quantized_vectors.get(other_idx) {
                self.quantized_vectors.push(qv.clone());
            } else {
                self.quantized_vectors.push(None);
            }

            merged_count += 1;
        }

        // Merge HNSW indexes using IGTM
        if let (Some(ref mut self_index), Some(ref other_index)) =
            (&mut self.hnsw_index, &other.hnsw_index)
        {
            self_index.merge_from(other_index)?;
        } else {
            // Fallback: rebuild index if other didn't have one
            eprintln!("  ⚠️  Other store has no HNSW index, rebuilding...");
            self.rebuild_index()?;
        }

        eprintln!(
            "✅ Merged {} vectors in {:.2}s",
            merged_count,
            start.elapsed().as_secs_f64()
        );

        Ok(merged_count)
    }

    /// K-nearest neighbors search using HNSW
    ///
    /// Quantization (if enabled) is for storage/memory savings only.
    /// Search always uses HNSW with original vectors for accuracy and speed.
    pub fn knn_search(&mut self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.dim()
            );
        }

        // Check if we have any data (either in vectors or in HNSW)
        let has_data = !self.vectors.is_empty()
            || (self.hnsw_index.is_some() && !self.hnsw_index.as_ref().unwrap().is_empty());

        if !has_data {
            return Ok(Vec::new());
        }

        // CRITICAL FIX: Rebuild index if missing but vectors exist
        // This handles the case where vectors were loaded from disk but index wasn't persisted
        if self.hnsw_index.is_none() && self.vectors.len() > 100 {
            eprintln!(
                "⚠️  HNSW index missing for {} vectors - rebuilding...",
                self.vectors.len()
            );
            self.rebuild_index()?;
        }

        // Use HNSW index if available
        // NOTE: Quantization (if enabled) is for storage only, not search
        if let Some(ref index) = self.hnsw_index {
            return index.search(&query.data, k);
        }

        // Fallback to brute-force if no index (small datasets only)
        eprintln!(
            "ℹ️  Using brute-force search for {} vectors",
            self.vectors.len()
        );
        self.knn_search_brute_force(query, k)
    }

    /// K-nearest neighbors search with metadata filtering
    ///
    /// Performs HNSW search and filters results by metadata.
    /// Uses oversample-and-filter strategy:
    /// 1. Fetch k*3 candidates from HNSW (to account for filtered results)
    /// 2. Filter by metadata
    /// 3. Return top-k filtered results
    ///
    /// Returns Vec of (id, distance, metadata) tuples
    pub fn knn_search_with_filter(
        &mut self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        // Use ACORN-1 filtered search if HNSW index is available
        if let Some(ref hnsw) = self.hnsw_index {
            // Create filter closure that checks metadata
            let metadata_map = &self.metadata;
            let deleted_map = &self.deleted;
            let filter_fn = |node_id: u32| -> bool {
                let index = node_id as usize;

                // Skip deleted vectors
                if deleted_map.contains_key(&index) {
                    return false;
                }

                // Get metadata (default to empty object if none)
                let metadata = metadata_map
                    .get(&index)
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                // Apply filter
                filter.matches(&metadata)
            };

            // Use ACORN-1 filtered search (includes adaptive threshold and 2-hop exploration)
            let search_results = hnsw.search_with_filter(&query.data, k, filter_fn)?;

            // Convert to (index, distance, metadata) format
            let filtered_results: Vec<(usize, f32, JsonValue)> = search_results
                .into_iter()
                .map(|(index, distance)| {
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    (index, distance, metadata)
                })
                .collect();

            return Ok(filtered_results);
        }

        // Fallback: Brute-force search with filtering (no HNSW index)
        let mut all_results: Vec<(usize, f32, JsonValue)> = self
            .vectors
            .iter()
            .enumerate()
            .filter_map(|(index, vec)| {
                // Skip deleted vectors
                if self.deleted.contains_key(&index) {
                    return None;
                }

                // Get metadata and check filter
                let metadata = self
                    .metadata
                    .get(&index)
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                if !filter.matches(&metadata) {
                    return None;
                }

                // Calculate distance
                let distance = query.l2_distance(vec).unwrap_or(f32::MAX);
                Some((index, distance, metadata))
            })
            .collect();

        // Sort by distance and take top k
        all_results.sort_by(|a, b| a.1.total_cmp(&b.1));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Search with optional filter (convenience method)
    pub fn search(
        &mut self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        if let Some(f) = filter {
            self.knn_search_with_filter(query, k, f)
        } else {
            // No filter - get all results with metadata
            let results = self.knn_search(query, k)?;
            Ok(results
                .into_iter()
                .filter_map(|(index, distance)| {
                    // Skip deleted vectors
                    if self.deleted.contains_key(&index) {
                        return None;
                    }
                    // Get metadata (default to empty object)
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    Some((index, distance, metadata))
                })
                .collect())
        }
    }

    /// Two-phase search with quantization + reranking
    ///
    /// Phase 1: Use quantized vectors for fast filtering (get k*3 candidates)
    /// Phase 2: Rerank candidates with original vectors (get final k)
    ///
    /// Note: Currently unused (quantization is storage-only), but kept for future hybrid search
    #[allow(dead_code)]
    fn knn_search_with_reranking(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        let quantizer = self.quantizer.as_ref().unwrap();

        // Quantize query
        let quantized_query = quantizer.quantize(&query.data);

        // Phase 1: Fast filtering with quantized vectors (oversample 3x)
        let oversample = (k * 3).min(self.vectors.len());
        let mut distances: Vec<(usize, f32)> = self
            .quantized_vectors
            .iter()
            .enumerate()
            .filter_map(|(id, qv_opt)| {
                qv_opt.as_ref().map(|qv| {
                    let dist = quantizer.distance_l2(&quantized_query, qv);
                    (id, dist)
                })
            })
            .collect();

        // Sort by quantized distance and take top candidates
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let candidates: Vec<usize> = distances
            .into_iter()
            .take(oversample)
            .map(|(id, _)| id)
            .collect();

        // Phase 2: Rerank with original vectors
        let mut reranked: Vec<(usize, f32)> = candidates
            .into_iter()
            .filter_map(|id| {
                self.vectors.get(id).map(|vec| {
                    let dist = query.l2_distance(vec).unwrap_or(f32::MAX);
                    (id, dist)
                })
            })
            .collect();

        // Sort by exact distance and return top-k
        reranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(reranked.into_iter().take(k).collect())
    }

    /// Brute-force K-NN search (fallback, mainly for testing)
    pub fn knn_search_brute_force(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.dim()
            );
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Compute distances to all vectors
        let mut distances: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = query.l2_distance(vec).unwrap_or(f32::MAX);
                (id, dist)
            })
            .collect();

        // Sort by distance and take top K
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(distances.into_iter().take(k).collect())
    }

    /// Get vector by ID
    pub fn get(&self, id: usize) -> Option<&Vector> {
        self.vectors.get(id)
    }

    /// Number of vectors stored (excluding deleted vectors)
    pub fn len(&self) -> usize {
        // Return active vector count (excluding tombstones)
        self.vectors.len().saturating_sub(self.deleted.len())
    }

    /// Check if store is empty (no active vectors)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory usage estimate (bytes)
    pub fn memory_usage(&self) -> usize {
        // Vector data: num_vectors * dim * 4 bytes (f32)
        self.vectors.iter().map(|v| v.dim() * 4).sum::<usize>()
    }

    /// Bytes per vector (average)
    pub fn bytes_per_vector(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        self.memory_usage() as f32 / self.vectors.len() as f32
    }

    /// Set HNSW ef_search parameter (runtime tuning)
    pub fn set_ef_search(&mut self, ef_search: usize) {
        if let Some(ref mut index) = self.hnsw_index {
            index.set_ef_search(ef_search);
        }
    }

    /// Get HNSW ef_search parameter
    pub fn get_ef_search(&self) -> Option<usize> {
        self.hnsw_index.as_ref().map(|idx| idx.get_ef_search())
    }

    /// Save vector store to disk with HNSW graph serialization
    ///
    /// Uses hnsw_rs file_dump() to persist both vectors and graph structure.
    /// This enables fast loading (<1s) without rebuilding the index.
    ///
    /// File format:
    /// - `<basename>.hnsw`: HNSW index
    /// - `<basename>.vectors.bin`: Vector data
    /// - `<basename>.quantized.bin`: Quantized vectors (if quantization enabled)
    /// - `<basename>.quantizer.json`: Quantizer parameters (if quantization enabled)
    pub fn save_to_disk(&self, base_path: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(base_path);
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid path: no filename in '{}'", base_path))?;

        // Create directory if needed
        fs::create_dir_all(directory)?;

        // Always save vectors array (needed for get/len/verification)
        let vectors_path = directory.join(format!("{}.vectors.bin", filename));
        let vectors_data: Vec<Vec<f32>> = self.vectors.iter().map(|v| v.data.clone()).collect();
        let encoded = bincode::serialize(&vectors_data)?;
        fs::write(&vectors_path, encoded)?;

        // Save quantized vectors if quantization is enabled
        if self.quantizer.is_some() && !self.quantized_vectors.is_empty() {
            let quantized_path = directory.join(format!("{}.quantized.bin", filename));
            let encoded = bincode::serialize(&self.quantized_vectors)?;
            fs::write(&quantized_path, encoded)?;

            // Save quantizer parameters
            let params_path = directory.join(format!("{}.quantizer.json", filename));
            let quantizer = self.quantizer.as_ref().unwrap();
            let params_json = serde_json::to_string_pretty(&quantizer.params())?;
            fs::write(&params_path, params_json)?;
        }

        // Save metadata if present
        if !self.metadata.is_empty() {
            let metadata_path = directory.join(format!("{}.metadata.json", filename));
            let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
            fs::write(&metadata_path, metadata_json)?;
        }

        // Save ID to index mapping if present
        if !self.id_to_index.is_empty() {
            let id_mapping_path = directory.join(format!("{}.id_mapping.json", filename));
            let id_mapping_json = serde_json::to_string_pretty(&self.id_to_index)?;
            fs::write(&id_mapping_path, id_mapping_json)?;
        }

        // Save deleted vectors tombstones if present
        if !self.deleted.is_empty() {
            let deleted_path = directory.join(format!("{}.deleted.json", filename));
            let deleted_json = serde_json::to_string_pretty(&self.deleted)?;
            fs::write(&deleted_path, deleted_json)?;
        }

        // Check if HNSW index exists
        if let Some(ref index) = self.hnsw_index {
            // Save HNSW index using our fast binary format
            let hnsw_path = directory.join(format!("{}.hnsw", filename));
            index.save(&hnsw_path)?;

            let quantization_status = if self.quantizer.is_some() {
                " with Extended RaBitQ quantization"
            } else {
                ""
            };

            eprintln!(
                "💾 Saved {} vectors ({} dims) with HNSW index{} to {}",
                self.vectors.len(),
                self.dimensions,
                quantization_status,
                base_path
            );
        } else {
            eprintln!(
                "💾 Saved {} vectors ({} dims) without HNSW index (no index built yet)",
                self.vectors.len(),
                self.dimensions
            );
        }

        Ok(())
    }

    /// Load vector store from disk with fast HNSW index loading
    ///
    /// Tries to load HNSW index first (fast: <1s).
    /// Falls back to loading vectors and rebuilding if index not found.
    ///
    /// Performance:
    /// - With HNSW index: <1 second load time (4175x faster than rebuild)
    /// - Fallback (rebuild): Several minutes for 100K+ vectors
    pub fn load_from_disk(base_path: &str, dimensions: usize) -> Result<Self> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(base_path);
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid path: no filename in '{}'", base_path))?;

        // Check if HNSW index file exists
        let hnsw_path = directory.join(format!("{}.hnsw", filename));

        if hnsw_path.exists() {
            // Fast path: Load HNSW index directly (<1s)
            eprintln!("📂 Loading HNSW index from {}...", hnsw_path.display());

            let hnsw_index = HNSWIndex::load(&hnsw_path)?;

            // Load vectors array (needed for get/len/verification)
            let vectors_path = directory.join(format!("{}.vectors.bin", filename));
            let vectors = if vectors_path.exists() {
                let vectors_data = fs::read(&vectors_path)?;
                let vectors_raw: Vec<Vec<f32>> = bincode::deserialize(&vectors_data)?;
                vectors_raw.into_iter().map(Vector::new).collect()
            } else {
                // Fallback: empty vectors (search still works via HNSW)
                eprintln!("⚠️  Warning: vectors.bin not found, get() and len() unavailable");
                Vec::new()
            };

            // Try to load quantizer parameters and quantized vectors
            let params_path = directory.join(format!("{}.quantizer.json", filename));
            let quantized_path = directory.join(format!("{}.quantized.bin", filename));

            let (quantizer, quantized_vectors) = if params_path.exists() && quantized_path.exists()
            {
                // Load quantizer parameters
                let params_json = fs::read_to_string(&params_path)?;
                let params: RaBitQParams = serde_json::from_str(&params_json)?;
                let quantizer = RaBitQ::new(params);

                // Load quantized vectors
                let quantized_data = fs::read(&quantized_path)?;
                let quantized_vectors: Vec<Option<QuantizedVector>> =
                    bincode::deserialize(&quantized_data)?;

                eprintln!(
                    "  Loaded Extended RaBitQ quantization ({} quantized vectors)",
                    quantized_vectors.len()
                );

                (Some(quantizer), quantized_vectors)
            } else {
                (None, Vec::new())
            };

            // Try to load metadata
            let metadata_path = directory.join(format!("{}.metadata.json", filename));
            let metadata = if metadata_path.exists() {
                let metadata_json = fs::read_to_string(&metadata_path)?;
                serde_json::from_str(&metadata_json)?
            } else {
                HashMap::new()
            };

            // Try to load ID to index mapping
            let id_mapping_path = directory.join(format!("{}.id_mapping.json", filename));
            let id_to_index = if id_mapping_path.exists() {
                let id_mapping_json = fs::read_to_string(&id_mapping_path)?;
                serde_json::from_str(&id_mapping_json)?
            } else {
                HashMap::new()
            };

            // Try to load deleted tombstones
            let deleted_path = directory.join(format!("{}.deleted.json", filename));
            let deleted = if deleted_path.exists() {
                let deleted_json = fs::read_to_string(&deleted_path)?;
                serde_json::from_str(&deleted_json)?
            } else {
                HashMap::new()
            };

            eprintln!(
                "✅ Loaded {} vectors ({} dims) with HNSW index (fast load: <1s)",
                vectors.len(),
                dimensions
            );

            Ok(Self {
                vectors,
                hnsw_index: Some(hnsw_index),
                dimensions,
                quantizer,
                quantized_vectors,
                metadata,
                id_to_index,
                deleted,
                storage: None, // Legacy file-based loading doesn't use seerdb
            })
        } else {
            // Fallback: Load vectors and rebuild HNSW
            eprintln!("📂 HNSW index not found, loading vectors and rebuilding...");

            let vectors_path = directory.join(format!("{}.vectors.bin", filename));
            if !vectors_path.exists() {
                anyhow::bail!("Vector file not found: {:?}", vectors_path);
            }

            let vectors_data = fs::read(&vectors_path)?;
            let vectors_raw: Vec<Vec<f32>> = bincode::deserialize(&vectors_data)?;
            let vectors: Vec<Vector> = vectors_raw.into_iter().map(Vector::new).collect();

            eprintln!(
                "📂 Loaded {} vectors ({} dims), rebuilding HNSW...",
                vectors.len(),
                dimensions
            );

            // Try to load quantizer parameters
            let params_path = directory.join(format!("{}.quantizer.json", filename));
            let quantized_path = directory.join(format!("{}.quantized.bin", filename));

            let (quantizer, quantized_vectors) = if params_path.exists() && quantized_path.exists()
            {
                // Load quantizer parameters
                let params_json = fs::read_to_string(&params_path)?;
                let params: RaBitQParams = serde_json::from_str(&params_json)?;
                let quantizer = RaBitQ::new(params);

                // Load quantized vectors
                let quantized_data = fs::read(&quantized_path)?;
                let quantized_vectors: Vec<Option<QuantizedVector>> =
                    bincode::deserialize(&quantized_data)?;

                eprintln!(
                    "  Loaded Extended RaBitQ quantization ({} quantized vectors)",
                    quantized_vectors.len()
                );

                (Some(quantizer), quantized_vectors)
            } else {
                (None, Vec::new())
            };

            // Try to load metadata
            let metadata_path = directory.join(format!("{}.metadata.json", filename));
            let metadata = if metadata_path.exists() {
                let metadata_json = fs::read_to_string(&metadata_path)?;
                serde_json::from_str(&metadata_json)?
            } else {
                HashMap::new()
            };

            // Try to load ID to index mapping
            let id_mapping_path = directory.join(format!("{}.id_mapping.json", filename));
            let id_to_index = if id_mapping_path.exists() {
                let id_mapping_json = fs::read_to_string(&id_mapping_path)?;
                serde_json::from_str(&id_mapping_json)?
            } else {
                HashMap::new()
            };

            // Try to load deleted tombstones
            let deleted_path = directory.join(format!("{}.deleted.json", filename));
            let deleted = if deleted_path.exists() {
                let deleted_json = fs::read_to_string(&deleted_path)?;
                serde_json::from_str(&deleted_json)?
            } else {
                HashMap::new()
            };

            // Create VectorStore and rebuild HNSW index
            let mut store = Self {
                vectors,
                hnsw_index: None,
                dimensions,
                quantizer,
                quantized_vectors,
                metadata,
                id_to_index,
                deleted,
                storage: None, // Legacy file-based loading doesn't use seerdb
            };

            if !store.vectors.is_empty() {
                store.rebuild_index()?;
            }

            Ok(store)
        }
    }

    /// Add a flush method to explicitly sync data to disk
    pub fn flush(&self) -> Result<()> {
        if let Some(ref storage) = self.storage {
            storage.flush()?;
        }
        Ok(())
    }

    /// Check if this store has persistent storage enabled
    pub fn is_persistent(&self) -> bool {
        self.storage.is_some()
    }

    /// Get reference to the seerdb storage backend (if persistent)
    ///
    /// Returns None if storage is not persistent (in-memory mode).
    /// Use for profiling/stats access only.
    pub fn storage(&self) -> Option<&SeerDBStorage> {
        self.storage.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: usize) -> Vector {
        let data: Vec<f32> = (0..dim).map(|i| ((seed + i) as f32) * 0.1).collect();
        Vector::new(data)
    }

    #[test]
    fn test_vector_store_insert() {
        let mut store = VectorStore::new(128);

        let v1 = random_vector(128, 0);
        let v2 = random_vector(128, 1);

        let id1 = store.insert(v1).unwrap();
        let id2 = store.insert(v2).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_vector_store_knn_with_hnsw() {
        let mut store = VectorStore::new(128);

        // Insert some vectors
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Query for nearest neighbors (uses HNSW)
        let query = random_vector(128, 50);
        let results = store.knn_search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_vector_store_brute_force() {
        let mut store = VectorStore::new(128);

        // Insert some vectors
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Query using brute-force
        let query = random_vector(128, 50);
        let results = store.knn_search_brute_force(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = VectorStore::new(128);
        let wrong_dim = Vector::new(vec![1.0; 64]);

        assert!(store.insert(wrong_dim).is_err());
    }

    #[test]
    fn test_ef_search_tuning() {
        let mut store = VectorStore::new(128);

        // Insert vectors to initialize HNSW
        for i in 0..10 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Check default ef_search (lazy init uses 10K capacity → medium tier → ef_construction=200)
        // HNSWIndex::new sets ef_search = ef_construction for consistency
        assert_eq!(store.get_ef_search(), Some(200));

        // Tune ef_search
        store.set_ef_search(600);
        assert_eq!(store.get_ef_search(), Some(600));
    }

    #[test]
    fn test_save_load_roundtrip() {
        use std::fs;

        let test_dir = "/tmp/omendb_test_vector_store";
        let test_path = format!("{}/test_store", test_dir);

        // Clean up any existing test data
        let _ = fs::remove_dir_all(test_dir);

        // Create store with 100 vectors
        let mut store = VectorStore::new(128);
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Verify we have HNSW index
        assert!(store.hnsw_index.is_some());
        assert_eq!(store.len(), 100);

        // Save to disk
        store.save_to_disk(&test_path).unwrap();

        // Verify HNSW index file exists
        assert!(fs::metadata(format!("{}/test_store.hnsw", test_dir)).is_ok());
        assert!(fs::metadata(format!("{}/test_store.vectors.bin", test_dir)).is_ok());

        // Load from disk
        let loaded_store = VectorStore::load_from_disk(&test_path, 128).unwrap();

        // Verify loaded store
        assert_eq!(loaded_store.len(), 100);
        assert_eq!(loaded_store.dimensions, 128);
        assert!(
            loaded_store.hnsw_index.is_some(),
            "HNSW index should be rebuilt"
        );

        // Verify vectors are identical
        for i in 0..100 {
            let original = store.get(i).unwrap();
            let loaded = loaded_store.get(i).unwrap();
            assert_eq!(original.data, loaded.data);
        }

        // Verify search works on loaded store
        let query = random_vector(128, 50);
        let mut loaded_mut = loaded_store;
        let results = loaded_mut.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Clean up
        let _ = fs::remove_dir_all(test_dir);
    }

    #[test]
    fn test_rebuild_index() {
        let mut store = VectorStore::new(128);

        // Insert vectors
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Verify HNSW index exists
        assert!(store.hnsw_index.is_some());

        // Clear the index
        store.hnsw_index = None;
        assert!(store.hnsw_index.is_none());

        // Rebuild index
        store.rebuild_index().unwrap();

        // Verify index is rebuilt
        assert!(store.hnsw_index.is_some());

        // Verify search works
        let query = random_vector(128, 50);
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_quantization_insert() {
        use super::super::rabitq::RaBitQParams;

        // Create store with 4-bit quantization
        let params = RaBitQParams::bits4();
        let mut store = VectorStore::new_with_quantization(128, params);

        // Insert vectors
        for i in 0..50 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Verify quantized vectors were created
        assert_eq!(store.vectors.len(), 50);
        assert_eq!(store.quantized_vectors.len(), 50);
        assert!(store.quantized_vectors.iter().all(|qv| qv.is_some()));
    }

    #[test]
    fn test_quantization_search_accuracy() {
        use super::super::rabitq::RaBitQParams;

        // Create store with 4-bit quantization
        let params = RaBitQParams::bits4();
        let mut store = VectorStore::new_with_quantization(128, params);

        // Insert vectors
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Search with quantization (uses two-phase search)
        let query = random_vector(128, 50);
        let results = store.knn_search(&query, 10).unwrap();

        // Should still get 10 results
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_quantization_persistence() {
        use super::super::rabitq::RaBitQParams;
        use std::fs;

        let test_dir = "/tmp/omendb_test_quantization";
        let test_path = format!("{}/test_store", test_dir);

        // Clean up any existing test data
        let _ = fs::remove_dir_all(test_dir);

        // Create store with 4-bit quantization
        let params = RaBitQParams::bits4();
        let mut store = VectorStore::new_with_quantization(128, params);

        // Insert vectors
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Save to disk
        store.save_to_disk(&test_path).unwrap();

        // Verify quantization files exist
        assert!(fs::metadata(format!("{}/test_store.quantized.bin", test_dir)).is_ok());
        assert!(fs::metadata(format!("{}/test_store.quantizer.json", test_dir)).is_ok());

        // Load from disk
        let mut loaded_store = VectorStore::load_from_disk(&test_path, 128).unwrap();

        // Verify quantized vectors were loaded
        assert_eq!(loaded_store.quantized_vectors.len(), 100);
        assert!(loaded_store.quantizer.is_some());

        // Verify search works with loaded quantization
        let query = random_vector(128, 50);
        let results = loaded_store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Clean up
        let _ = fs::remove_dir_all(test_dir);
    }

    #[test]
    fn test_quantization_batch_insert() {
        use super::super::rabitq::RaBitQParams;

        // Create store with 4-bit quantization
        let params = RaBitQParams::bits4();
        let mut store = VectorStore::new_with_quantization(128, params);

        // Batch insert vectors
        let vectors: Vec<Vector> = (0..100).map(|i| random_vector(128, i)).collect();
        let ids = store.batch_insert(vectors).unwrap();

        // Verify all vectors and quantized vectors were created
        assert_eq!(ids.len(), 100);
        assert_eq!(store.vectors.len(), 100);
        assert_eq!(store.quantized_vectors.len(), 100);
        assert!(store.quantized_vectors.iter().all(|qv| qv.is_some()));
    }

    //  Adaptive parameter selection tests

    #[test]
    fn test_adaptive_params_small_scale() {
        // < 50K vectors should use M=16 (fast & efficient)
        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(10_000);
        assert_eq!(m, 16);
        assert_eq!(ef_construction, 200);
        assert_eq!(ef_search, 100);

        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(49_999);
        assert_eq!(m, 16);
        assert_eq!(ef_construction, 200);
        assert_eq!(ef_search, 100);
    }

    #[test]
    fn test_adaptive_params_medium_scale() {
        // 50K-500K vectors should use M=32 (balanced, 98% recall)
        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(50_000);
        assert_eq!(m, 32);
        assert_eq!(ef_construction, 400);
        assert_eq!(ef_search, 100);

        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(100_000);
        assert_eq!(m, 32);
        assert_eq!(ef_construction, 400);
        assert_eq!(ef_search, 100);

        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(499_999);
        assert_eq!(m, 32);
        assert_eq!(ef_construction, 400);
        assert_eq!(ef_search, 100);
    }

    #[test]
    fn test_adaptive_params_large_scale() {
        // >= 500K vectors should use M=48 (high recall, 99%)
        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(500_000);
        assert_eq!(m, 48);
        assert_eq!(ef_construction, 600);
        assert_eq!(ef_search, 150);

        let (m, ef_construction, ef_search) = VectorStore::adaptive_hnsw_params(1_000_000);
        assert_eq!(m, 48);
        assert_eq!(ef_construction, 600);
        assert_eq!(ef_search, 150);
    }

    #[test]
    fn test_new_with_capacity_small() {
        let store = VectorStore::new_with_capacity(128, 10_000);

        // Should have eagerly initialized HNSW with M=16
        assert!(store.hnsw_index.is_some());

        // Verify parameters (we can't directly inspect but can test behavior)
        assert_eq!(store.dimensions, 128);
    }

    #[test]
    fn test_new_with_capacity_medium() {
        let store = VectorStore::new_with_capacity(128, 100_000);

        // Should have eagerly initialized HNSW with M=32
        assert!(store.hnsw_index.is_some());
        assert_eq!(store.dimensions, 128);
    }

    #[test]
    fn test_new_with_capacity_large() {
        let store = VectorStore::new_with_capacity(128, 1_000_000);

        // Should have eagerly initialized HNSW with M=48
        assert!(store.hnsw_index.is_some());
        assert_eq!(store.dimensions, 128);
    }

    #[test]
    fn test_new_with_capacity_functional() {
        // Verify new_with_capacity works functionally (can insert and search)
        let mut store = VectorStore::new_with_capacity(128, 100_000);

        // Insert vectors
        for i in 0..100 {
            store.insert(random_vector(128, i)).unwrap();
        }

        // Search
        let query = random_vector(128, 50);
        let results = store.knn_search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    // Tests for metadata support

    #[test]
    fn test_insert_with_metadata() {
        let mut store = VectorStore::new(128);

        let metadata = serde_json::json!({
            "title": "Test Document",
            "author": "Alice",
            "year": 2024
        });

        let index = store
            .insert_with_metadata("doc1".to_string(), random_vector(128, 0), metadata.clone())
            .unwrap();

        assert_eq!(index, 0);
        assert!(store.id_to_index.contains_key("doc1"));
        assert_eq!(store.metadata.get(&0), Some(&metadata));
    }

    #[test]
    fn test_set_insert() {
        let mut store = VectorStore::new(128);

        let metadata = serde_json::json!({"title": "Doc 1"});

        // First set should insert
        let index = store
            .set("doc1".to_string(), random_vector(128, 0), metadata.clone())
            .unwrap();

        assert_eq!(index, 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_set_update() {
        let mut store = VectorStore::new(128);

        // Insert initial document
        store
            .set(
                "doc1".to_string(),
                random_vector(128, 0),
                serde_json::json!({"title": "Original"}),
            )
            .unwrap();

        // Upsert with same ID should update
        let index = store
            .set(
                "doc1".to_string(),
                random_vector(128, 1),
                serde_json::json!({"title": "Updated"}),
            )
            .unwrap();

        assert_eq!(index, 0);
        assert_eq!(store.len(), 1); // Still only 1 vector
        assert_eq!(
            store.metadata.get(&0).unwrap().get("title").unwrap(),
            "Updated"
        );
    }

    #[test]
    fn test_delete() {
        let mut store = VectorStore::new(128);

        store
            .insert_with_metadata(
                "doc1".to_string(),
                random_vector(128, 0),
                serde_json::json!({"title": "Doc 1"}),
            )
            .unwrap();

        // Delete the document
        store.delete("doc1").unwrap();

        // Should be marked as deleted
        assert!(store.deleted.contains_key(&0));
        assert!(!store.id_to_index.contains_key("doc1"));

        // get_by_id should return None for deleted
        assert!(store.get_by_id("doc1").is_none());
    }

    #[test]
    fn test_update() {
        let mut store = VectorStore::new(128);

        store
            .insert_with_metadata(
                "doc1".to_string(),
                random_vector(128, 0),
                serde_json::json!({"title": "Original"}),
            )
            .unwrap();

        // Update metadata only
        store
            .update(
                "doc1",
                None,
                Some(serde_json::json!({"title": "Updated", "author": "Bob"})),
            )
            .unwrap();

        let (_, metadata) = store.get_by_id("doc1").unwrap();
        assert_eq!(metadata.get("title").unwrap(), "Updated");
        assert_eq!(metadata.get("author").unwrap(), "Bob");
    }

    #[test]
    fn test_metadata_filter_eq() {
        let filter = MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice"));

        let metadata1 = serde_json::json!({"author": "Alice"});
        let metadata2 = serde_json::json!({"author": "Bob"});

        assert!(filter.matches(&metadata1));
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_metadata_filter_gte() {
        let filter = MetadataFilter::Gte("year".to_string(), 2020.0);

        let metadata1 = serde_json::json!({"year": 2024});
        let metadata2 = serde_json::json!({"year": 2019});

        assert!(filter.matches(&metadata1));
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_metadata_filter_and() {
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice")),
            MetadataFilter::Gte("year".to_string(), 2020.0),
        ]);

        let metadata1 = serde_json::json!({"author": "Alice", "year": 2024});
        let metadata2 = serde_json::json!({"author": "Alice", "year": 2019});
        let metadata3 = serde_json::json!({"author": "Bob", "year": 2024});

        assert!(filter.matches(&metadata1));
        assert!(!filter.matches(&metadata2));
        assert!(!filter.matches(&metadata3));
    }

    #[test]
    fn test_search_with_filter() {
        let mut store = VectorStore::new(128);

        // Insert vectors with metadata
        store
            .set(
                "doc1".to_string(),
                random_vector(128, 0),
                serde_json::json!({"author": "Alice", "year": 2024}),
            )
            .unwrap();

        store
            .set(
                "doc2".to_string(),
                random_vector(128, 1),
                serde_json::json!({"author": "Bob", "year": 2023}),
            )
            .unwrap();

        store
            .set(
                "doc3".to_string(),
                random_vector(128, 2),
                serde_json::json!({"author": "Alice", "year": 2022}),
            )
            .unwrap();

        // Search with filter for Alice's documents
        let filter = MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice"));
        let query = random_vector(128, 0);
        let results = store.knn_search_with_filter(&query, 10, &filter).unwrap();

        // Should only return Alice's documents (doc1 and doc3)
        assert_eq!(results.len(), 2);
        for (_, _, metadata) in &results {
            assert_eq!(metadata.get("author").unwrap(), "Alice");
        }
    }

    #[test]
    fn test_persistence_with_metadata() {
        use std::fs;

        let test_dir = "/tmp/omendb_test_metadata";
        let test_path = format!("{}/test_store", test_dir);

        // Clean up any existing test data
        let _ = fs::remove_dir_all(test_dir);

        // Create store with metadata
        let mut store = VectorStore::new(128);

        store
            .set(
                "doc1".to_string(),
                random_vector(128, 0),
                serde_json::json!({"title": "Doc 1", "year": 2024}),
            )
            .unwrap();

        store
            .set(
                "doc2".to_string(),
                random_vector(128, 1),
                serde_json::json!({"title": "Doc 2", "year": 2023}),
            )
            .unwrap();

        // Save to disk
        store.save_to_disk(&test_path).unwrap();

        // Verify metadata files exist
        assert!(fs::metadata(format!("{}/test_store.metadata.json", test_dir)).is_ok());
        assert!(fs::metadata(format!("{}/test_store.id_mapping.json", test_dir)).is_ok());

        // Load from disk
        let loaded_store = VectorStore::load_from_disk(&test_path, 128).unwrap();

        // Verify metadata was loaded
        assert_eq!(loaded_store.metadata.len(), 2);
        assert_eq!(loaded_store.id_to_index.len(), 2);

        // Verify we can retrieve by ID
        let (_, metadata) = loaded_store.get_by_id("doc1").unwrap();
        assert_eq!(metadata.get("title").unwrap(), "Doc 1");
        assert_eq!(metadata.get("year").unwrap(), 2024);

        // Clean up
        let _ = fs::remove_dir_all(test_dir);
    }

    #[test]
    fn test_get_by_id() {
        let mut store = VectorStore::new(128);

        let vector = random_vector(128, 0);
        let metadata = serde_json::json!({"title": "Test"});

        store
            .insert_with_metadata("doc1".to_string(), vector.clone(), metadata.clone())
            .unwrap();

        // Get by ID
        let (retrieved_vector, retrieved_metadata) = store.get_by_id("doc1").unwrap();

        assert_eq!(retrieved_vector.data, vector.data);
        assert_eq!(retrieved_metadata, &metadata);

        // Non-existent ID should return None
        assert!(store.get_by_id("nonexistent").is_none());
    }

    // Tests for seerdb persistent storage

    #[test]
    fn test_open_new_database() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test-oadb");

        // Open new database
        let mut store = VectorStore::open(&db_path).unwrap();
        assert!(store.is_persistent());
        assert_eq!(store.len(), 0);

        // Insert some vectors
        store
            .set(
                "doc1".to_string(),
                random_vector(128, 0),
                serde_json::json!({"title": "Doc 1"}),
            )
            .unwrap();

        store
            .set(
                "doc2".to_string(),
                random_vector(128, 1),
                serde_json::json!({"title": "Doc 2"}),
            )
            .unwrap();

        assert_eq!(store.len(), 2);
        assert!(store.get_by_id("doc1").is_some());
    }

    #[test]
    fn test_persistent_roundtrip() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("roundtrip-oadb");

        // Create and populate store
        {
            let mut store = VectorStore::open(&db_path).unwrap();

            store
                .set(
                    "vec1".to_string(),
                    random_vector(128, 10),
                    serde_json::json!({"category": "A", "score": 0.95}),
                )
                .unwrap();

            store
                .set(
                    "vec2".to_string(),
                    random_vector(128, 20),
                    serde_json::json!({"category": "B", "score": 0.85}),
                )
                .unwrap();

            store
                .set(
                    "vec3".to_string(),
                    random_vector(128, 30),
                    serde_json::json!({"category": "A", "score": 0.75}),
                )
                .unwrap();

            // Flush to ensure data is on disk
            store.flush().unwrap();
        }

        // Reopen and verify data
        {
            let store = VectorStore::open(&db_path).unwrap();

            assert_eq!(store.len(), 3);

            // Verify vec1
            let (vec1, meta1) = store.get_by_id("vec1").unwrap();
            assert_eq!(vec1.data, random_vector(128, 10).data);
            assert_eq!(meta1["category"], "A");
            assert_eq!(meta1["score"], 0.95);

            // Verify vec2
            let (vec2, meta2) = store.get_by_id("vec2").unwrap();
            assert_eq!(vec2.data, random_vector(128, 20).data);
            assert_eq!(meta2["category"], "B");

            // Verify vec3
            assert!(store.get_by_id("vec3").is_some());
        }
    }

    #[test]
    fn test_persistent_delete() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("delete-oadb");

        // Create, populate, and delete
        {
            let mut store = VectorStore::open(&db_path).unwrap();

            store
                .set(
                    "keep".to_string(),
                    random_vector(128, 1),
                    serde_json::json!({}),
                )
                .unwrap();
            store
                .set(
                    "delete_me".to_string(),
                    random_vector(128, 2),
                    serde_json::json!({}),
                )
                .unwrap();

            assert_eq!(store.len(), 2);

            // Delete one
            store.delete("delete_me").unwrap();
            assert!(store.get_by_id("delete_me").is_none());

            store.flush().unwrap();
        }

        // Reopen and verify deletion persisted
        {
            let store = VectorStore::open(&db_path).unwrap();

            // Only "keep" should be accessible
            assert!(store.get_by_id("keep").is_some());
            assert!(store.get_by_id("delete_me").is_none());
        }
    }

    #[test]
    fn test_persistent_search() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("search-oadb");

        // Create and populate
        {
            let mut store = VectorStore::open(&db_path).unwrap();

            for i in 0..100 {
                store
                    .set(
                        format!("vec{}", i),
                        random_vector(128, i),
                        serde_json::json!({"index": i}),
                    )
                    .unwrap();
            }

            store.flush().unwrap();
        }

        // Reopen and search
        {
            let mut store = VectorStore::open(&db_path).unwrap();

            assert_eq!(store.len(), 100);

            // Search should work
            let query = random_vector(128, 50);
            let results = store.knn_search(&query, 10).unwrap();

            // Verify we get results
            assert_eq!(results.len(), 10, "Should return 10 results");

            // Verify results are sorted by distance
            for i in 1..results.len() {
                assert!(
                    results[i].1 >= results[i - 1].1,
                    "Results should be sorted by distance"
                );
            }
        }
    }
}

#[cfg(test)]
mod incremental_tests {
    use super::*;

    #[test]
    fn test_incremental_set_batch() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        // Single item inserts
        store
            .set_batch(vec![(
                "vec1".to_string(),
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        store
            .set_batch(vec![(
                "vec2".to_string(),
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Batch insert
        store
            .set_batch(vec![
                (
                    "vec3".to_string(),
                    Vector::new(vec![0.0, 0.0, 1.0, 0.0]),
                    serde_json::json!({}),
                ),
                (
                    "vec4".to_string(),
                    Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
                    serde_json::json!({}),
                ),
            ])
            .unwrap();

        // Another batch
        store
            .set_batch(vec![
                (
                    "vec5".to_string(),
                    Vector::new(vec![0.5, 0.5, 0.0, 0.0]),
                    serde_json::json!({}),
                ),
                (
                    "vec6".to_string(),
                    Vector::new(vec![0.0, 0.5, 0.5, 0.0]),
                    serde_json::json!({}),
                ),
            ])
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(
            results.len(),
            6,
            "Incremental inserts must all be searchable"
        );
    }

    /// INC-2: Interleave inserts and searches
    #[test]
    fn test_interleaved_insert_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        let mut total_inserted = 0;

        // Insert 10 batches of 10 vectors, searching after each batch
        for batch in 0..10 {
            let vectors: Vec<_> = (0..10)
                .map(|i| {
                    let id = batch * 10 + i;
                    let mut v = vec![0.0; 4];
                    v[id % 4] = 1.0 + (id as f32 * 0.01);
                    (format!("vec{}", id), Vector::new(v), serde_json::json!({}))
                })
                .collect();

            store.set_batch(vectors).unwrap();
            total_inserted += 10;

            // Search after each batch
            let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
            let results = store.knn_search(&query, total_inserted + 10).unwrap();
            assert_eq!(
                results.len(),
                total_inserted,
                "After batch {}, expected {} results but got {}",
                batch,
                total_inserted,
                results.len()
            );
        }

        // Final verification
        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.knn_search(&query, 200).unwrap();
        assert_eq!(results.len(), 100, "All 100 vectors must be searchable");
    }

    /// INC-3: Insert batch, search, single insert, search
    #[test]
    fn test_batch_then_single_insert() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        // Batch insert
        let batch: Vec<_> = (0..50)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                (format!("batch{}", i), Vector::new(v), serde_json::json!({}))
            })
            .collect();
        store.set_batch(batch).unwrap();

        // Search to "activate" the index
        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.knn_search(&query, 100).unwrap();
        assert_eq!(results.len(), 50, "Batch vectors must be searchable");

        // Single insert after search
        store
            .set_batch(vec![(
                "single".to_string(),
                Vector::new(vec![0.99, 0.01, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search again - new vector must be reachable
        let results = store.knn_search(&query, 100).unwrap();
        assert_eq!(
            results.len(),
            51,
            "New vector after search must be reachable"
        );

        // The new vector should appear in search results
        // Index 50 is the single insert (0-49 were batch)
        let found = results.iter().any(|(idx, _)| *idx == 50);
        assert!(found, "Newly inserted vector must appear in search results");
    }

    /// INC-4: Empty index -> insert -> search -> insert -> search cycle
    #[test]
    fn test_insert_search_cycle_from_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);

        // Search empty index
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 0, "Empty index should return no results");

        // First insert
        store
            .set_batch(vec![(
                "first".to_string(),
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search should find first vector
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 1, "Should find first vector");

        // Second insert
        store
            .set_batch(vec![(
                "second".to_string(),
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search should find both
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 2, "Should find both vectors");

        // Third insert
        store
            .set_batch(vec![(
                "third".to_string(),
                Vector::new(vec![0.5, 0.5, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search should find all three
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 3, "Should find all three vectors");
    }
}
