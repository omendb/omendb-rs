// LSM-Tree Compaction
// Implements leveled compaction to keep read amplification bounded

pub mod filter;
pub mod merge;

use crate::sstable::{SSTable, SSTableBuilder};
use bytes::Bytes;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

pub use filter::{CompactionFilter, FilterDecision};
pub use merge::MergeIterator;

#[derive(Debug, Error)]
pub enum CompactionError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SSTable error: {0}")]
    SSTable(#[from] crate::sstable::SSTableError),

    #[error("No SSTables to compact")]
    NoInput,
}

pub type Result<T> = std::result::Result<T, CompactionError>;

/// Compact multiple SSTables into a single SSTable
///
/// # Arguments
/// * `input_paths` - Paths to SSTables to compact (newer first)
/// * `output_path` - Path for output SSTable
/// * `compaction_level` - The level being compacted TO (passed to filter)
/// * `filter` - Optional compaction filter
///
/// # Returns
/// Path to the new SSTable and its size in bytes
///
/// # Arguments
/// * `input_paths` - Paths to SSTables to compact (newer first)
/// * `output_path` - Path for output SSTable
/// * `compaction_level` - The level being compacted TO (passed to filter)
/// * `filter` - Optional compaction filter
/// * `oldest_snapshot` - Oldest active snapshot seq for MVCC GC (u64::MAX = no GC)
pub fn compact_sstables(
    input_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    compaction_level: usize,
    filter: Option<Arc<dyn CompactionFilter>>,
    oldest_snapshot: u64,
) -> Result<(PathBuf, u64)> {
    if input_paths.is_empty() {
        return Err(CompactionError::NoInput);
    }

    // Open all input SSTables
    let mut sstables = Vec::new();
    for path in input_paths {
        let sstable = SSTable::open(path)?;
        sstables.push(sstable);
    }

    // Create merge iterator with MVCC GC
    let merge = MergeIterator::with_gc(sstables, compaction_level, filter, oldest_snapshot)?;

    // Build new SSTable from merged entries
    let output_path = output_path.as_ref().to_path_buf();
    let mut builder = SSTableBuilder::create(&output_path)?;

    for result in merge {
        let (key, value) = result?;
        // Use add_raw_mvcc to preserve vlog pointers (FLAG_POINTER + data)
        // and extract user key for bloom filter (for MVCC-encoded keys)
        builder.add_raw_mvcc(key, value)?;
    }

    // Finish writing
    builder.finish()?;

    // Get file size
    let metadata = std::fs::metadata(&output_path)?;
    let size = metadata.len();

    Ok((output_path, size))
}

/// Compact multiple SSTables into bytes (for cloud storage support)
///
/// Same as `compact_sstables` but buffers the output in memory and returns
/// the complete SSTable as bytes. This enables:
/// - Single-write local disk operations (fewer syscalls)
/// - Cloud storage uploads (S3/GCS/Azure)
///
/// # Arguments
/// * `input_paths` - Paths to SSTables to compact (newer first)
/// * `compaction_level` - The level being compacted TO (passed to filter)
/// * `filter` - Optional compaction filter
/// * `oldest_snapshot` - Oldest active snapshot seq for MVCC GC (u64::MAX = no GC)
///
/// # Returns
/// Complete SSTable as bytes (caller writes to file and/or uploads to cloud)
pub fn compact_sstables_buffered(
    input_paths: &[PathBuf],
    compaction_level: usize,
    filter: Option<Arc<dyn CompactionFilter>>,
    oldest_snapshot: u64,
) -> Result<Bytes> {
    if input_paths.is_empty() {
        return Err(CompactionError::NoInput);
    }

    // Open all input SSTables
    let mut sstables = Vec::new();
    for path in input_paths {
        let sstable = SSTable::open(path)?;
        sstables.push(sstable);
    }

    // Create merge iterator with MVCC GC
    let merge = MergeIterator::with_gc(sstables, compaction_level, filter, oldest_snapshot)?;

    // Build new SSTable in memory
    let mut builder = SSTableBuilder::new_buffered();

    for result in merge {
        let (key, value) = result?;
        // Use add_raw_mvcc to preserve vlog pointers (FLAG_POINTER + data)
        // and extract user key for bloom filter (for MVCC-encoded keys)
        builder.add_raw_mvcc(key, value)?;
    }

    // Finish building and return bytes
    let bytes = builder.finish_to_bytes()?;

    Ok(bytes)
}

/// Represents a level in the LSM tree
#[derive(Debug, Clone)]
pub struct Level {
    /// Level number (0 = memtable flush target, 1+ = compacted levels)
    level_num: usize,
    /// SSTables in this level (sorted by key range)
    sstables: Vec<PathBuf>,
    /// Current size in bytes
    size: u64,
    /// Size threshold for triggering compaction
    size_threshold: u64,
}

impl Level {
    /// Create a new level with given threshold
    pub fn new(level_num: usize, size_threshold: u64) -> Self {
        Self {
            level_num,
            sstables: Vec::new(),
            size: 0,
            size_threshold,
        }
    }

    /// Add an SSTable to this level
    pub fn add_sstable(&mut self, path: PathBuf, size: u64) {
        self.sstables.push(path);
        self.size += size;
    }

    /// Check if this level needs compaction
    pub fn needs_compaction(&self) -> bool {
        self.size >= self.size_threshold
    }

    /// Get number of SSTables in this level
    pub fn num_sstables(&self) -> usize {
        self.sstables.len()
    }

    /// Get current size
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get level number
    pub fn level_num(&self) -> usize {
        self.level_num
    }

    /// Get SSTables
    pub fn sstables(&self) -> &[PathBuf] {
        &self.sstables
    }
}

/// Workload-aware compaction strategy based on Dostoevsky paper
#[derive(Debug, Clone, Copy)]
pub enum CompactionStrategy {
    /// Fixed size ratio (traditional LSM)
    Fixed(u64),
    /// Adaptive size ratio based on read/write ratio (Dostoevsky)
    Adaptive {
        /// Current size ratio (dynamically adjusted)
        current_ratio: u64,
        /// Minimum allowed ratio
        min_ratio: u64,
        /// Maximum allowed ratio
        max_ratio: u64,
    },
}

impl CompactionStrategy {
    /// Get current size ratio
    pub fn current_ratio(&self) -> u64 {
        match self {
            Self::Fixed(ratio) => *ratio,
            Self::Adaptive { current_ratio, .. } => *current_ratio,
        }
    }

    /// Adjust ratio based on workload (Dostoevsky formula)
    ///
    /// Formula: T = sqrt((Z * W) / R) where:
    /// - T = optimal size ratio
    /// - Z = zero-result penalty (1-2, higher for expensive reads)
    /// - W = write operations
    /// - R = read operations
    ///
    /// Intuition:
    /// - Write-heavy workload → higher T (less compaction overhead)
    /// - Read-heavy workload → lower T (better read performance)
    pub fn adjust_for_workload(&mut self, writes: u64, reads: u64) {
        if let Self::Adaptive {
            current_ratio,
            min_ratio,
            max_ratio,
        } = self
        {
            if reads == 0 || writes == 0 {
                return; // Not enough data
            }

            // Dostoevsky formula with Z=1.5 (moderate penalty for negative lookups)
            let z = 1.5;
            let ratio_f64 = ((z * writes as f64) / reads as f64).sqrt();
            let new_ratio = ratio_f64.round() as u64;

            // Clamp to allowed range
            *current_ratio = new_ratio.max(*min_ratio).min(*max_ratio);
        }
    }
}

/// LSM Tree structure with multiple levels
#[derive(Clone)]
pub struct LSMTree {
    /// Levels (L0, L1, L2, ...)
    levels: Vec<Level>,
    /// Compaction strategy (fixed or adaptive)
    strategy: CompactionStrategy,
    /// Base level size (L1 threshold)
    base_size: u64,
    /// Data directory
    data_dir: PathBuf,
    /// Last workload adjustment (writes, reads)
    last_workload: (u64, u64),
}

impl LSMTree {
    /// Create a new LSM tree with fixed size ratio
    ///
    /// # Arguments
    /// * `data_dir` - Directory for SSTable files
    /// * `base_size` - L1 size threshold (default: 10MB)
    /// * `size_ratio` - Size ratio between levels (default: 10)
    /// * `num_levels` - Number of levels (default: 7)
    pub fn new(
        data_dir: impl AsRef<Path>,
        base_size: u64,
        size_ratio: u64,
        num_levels: usize,
    ) -> Self {
        Self::with_strategy(
            data_dir,
            base_size,
            num_levels,
            CompactionStrategy::Fixed(size_ratio),
        )
    }

    /// Create a new LSM tree with adaptive size ratio (Dostoevsky)
    ///
    /// # Arguments
    /// * `data_dir` - Directory for SSTable files
    /// * `base_size` - L1 size threshold (default: 10MB)
    /// * `num_levels` - Number of levels (default: 7)
    /// * `min_ratio` - Minimum size ratio (default: 4)
    /// * `max_ratio` - Maximum size ratio (default: 20)
    pub fn new_adaptive(
        data_dir: impl AsRef<Path>,
        base_size: u64,
        num_levels: usize,
        min_ratio: u64,
        max_ratio: u64,
    ) -> Self {
        let initial_ratio = (min_ratio + max_ratio) / 2; // Start in middle
        let strategy = CompactionStrategy::Adaptive {
            current_ratio: initial_ratio,
            min_ratio,
            max_ratio,
        };
        Self::with_strategy(data_dir, base_size, num_levels, strategy)
    }

    /// Create LSM tree with custom compaction strategy
    fn with_strategy(
        data_dir: impl AsRef<Path>,
        base_size: u64,
        num_levels: usize,
        strategy: CompactionStrategy,
    ) -> Self {
        let size_ratio = strategy.current_ratio();
        let mut levels = Vec::with_capacity(num_levels);

        // L0 has special handling (memtable flush target, no size limit)
        levels.push(Level::new(0, u64::MAX));

        // L1+ have exponentially increasing thresholds
        // Use saturating math to prevent overflow with extreme configurations
        for i in 1..num_levels {
            let exponent = (i - 1) as u32;
            let multiplier = size_ratio.saturating_pow(exponent);
            let threshold = base_size.saturating_mul(multiplier);
            levels.push(Level::new(i, threshold));
        }

        Self {
            levels,
            strategy,
            base_size,
            data_dir: data_dir.as_ref().to_path_buf(),
            last_workload: (0, 0),
        }
    }

    /// Add an SSTable to L0 (memtable flush)
    pub fn add_l0_sstable(&mut self, path: PathBuf, size: u64) {
        self.levels[0].add_sstable(path, size);
    }

    /// Check if any level needs compaction
    pub fn needs_compaction(&self) -> Option<usize> {
        // Check L0 first (special case: trigger on # of files, not size)
        if self.levels[0].num_sstables() >= 4 {
            return Some(0);
        }

        // Check other levels by size
        for (i, level) in self.levels.iter().enumerate().skip(1) {
            if level.needs_compaction() {
                return Some(i);
            }
        }

        None
    }

    /// Get a level
    pub fn level(&self, level_num: usize) -> Option<&Level> {
        self.levels.get(level_num)
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get data directory
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Adjust compaction strategy based on observed workload (Dostoevsky)
    ///
    /// Call this periodically (e.g., after every N operations) to adapt to workload changes.
    ///
    /// # Arguments
    /// * `writes` - Total write operations since start
    /// * `reads` - Total read operations since start
    ///
    /// # Returns
    /// True if size ratio changed (level thresholds were updated)
    pub fn adjust_for_workload(&mut self, writes: u64, reads: u64) -> bool {
        // Check if workload changed significantly
        let (last_writes, last_reads) = self.last_workload;
        let delta_writes = writes.saturating_sub(last_writes);
        let delta_reads = reads.saturating_sub(last_reads);

        // Only adjust if we have significant new operations (>1000)
        if delta_writes + delta_reads < 1000 {
            return false;
        }

        let old_ratio = self.strategy.current_ratio();
        self.strategy.adjust_for_workload(writes, reads);
        let new_ratio = self.strategy.current_ratio();

        self.last_workload = (writes, reads);

        // If ratio changed, update level thresholds
        if new_ratio != old_ratio {
            self.update_level_thresholds(new_ratio);
            true
        } else {
            false
        }
    }

    /// Update level thresholds based on new size ratio
    fn update_level_thresholds(&mut self, size_ratio: u64) {
        // Skip L0 (always u64::MAX)
        for i in 1..self.levels.len() {
            let exponent = (i - 1) as u32;
            let multiplier = size_ratio.saturating_pow(exponent);
            let threshold = self.base_size.saturating_mul(multiplier);
            self.levels[i].size_threshold = threshold;
        }
    }

    /// Get current compaction strategy
    pub fn strategy(&self) -> &CompactionStrategy {
        &self.strategy
    }

    /// Add an SSTable to a specific level (after compaction)
    pub fn add_to_level(&mut self, level_num: usize, path: PathBuf, size: u64) {
        if let Some(level) = self.levels.get_mut(level_num) {
            level.add_sstable(path, size);
        }
    }

    /// Remove specific SSTables from a level (after compaction)
    pub fn remove_sstables_from_level(&mut self, level_num: usize, paths: &[PathBuf]) {
        if let Some(level) = self.levels.get_mut(level_num) {
            for path in paths {
                if let Some(pos) = level.sstables.iter().position(|p| p == path) {
                    let removed_path = level.sstables.remove(pos);

                    // Update size (need to get file size)
                    if let Ok(metadata) = std::fs::metadata(&removed_path) {
                        let size = metadata.len();
                        level.size = level.size.saturating_sub(size);
                    }
                }
            }
        }
    }

    /// Clear all SSTables from a level (used during compaction)
    pub fn clear_level(&mut self, level_num: usize) -> Vec<PathBuf> {
        if let Some(level) = self.levels.get_mut(level_num) {
            let paths = std::mem::take(&mut level.sstables);
            level.size = 0;
            paths
        } else {
            Vec::new()
        }
    }

    /// Get all SSTable paths across all levels
    ///
    /// Returns paths in level order (L0 first, then L1, L2, etc.)
    /// Used by db.verify() for full database integrity checking.
    pub fn all_sstable_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for level in &self.levels {
            paths.extend(level.sstables.iter().cloned());
        }
        paths
    }

    /// Load existing SSTables from disk into the LSM tree
    ///
    /// Scans the data directory for SSTable files and adds them to L0.
    /// This is called during DB::open() to recover existing data.
    pub fn load_existing_sstables(&mut self) -> std::io::Result<()> {
        use crate::sstable::SSTable;

        // Scan data directory for .sst files
        let entries = std::fs::read_dir(&self.data_dir)?;
        let mut sstable_paths = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            // Only process .sst files
            if path.extension().and_then(|s| s.to_str()) == Some("sst") {
                sstable_paths.push(path);
            }
        }

        // Sort paths lexicographically to ensure L0 order (Oldest -> Newest)
        // Filenames are formatted as "L{level}_{seq:06}.sst", so lexicographical sort works
        sstable_paths.sort();

        for path in sstable_paths {
            // Get file size
            let metadata = std::fs::metadata(&path)?;
            let size = metadata.len();

            // Verify SSTable by opening it and validating all blocks
            // If corrupt, this will return an error
            let mut sstable = SSTable::open(&path).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Corrupt SSTable: {}", e),
                )
            })?;

            // Validate all blocks to detect corruption
            sstable.validate().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Corrupt SSTable: {}", e),
                )
            })?;

            // Add to L0 (all recovered SSTables go to L0)
            self.levels[0].add_sstable(path, size);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use tempfile::tempdir;

    #[test]
    fn test_level_creation() {
        let level = Level::new(1, 10_000_000);
        assert_eq!(level.level_num(), 1);
        assert_eq!(level.size(), 0);
        assert!(!level.needs_compaction());
    }

    #[test]
    fn test_level_compaction_trigger() {
        let mut level = Level::new(1, 1000);
        assert!(!level.needs_compaction());

        level.add_sstable(PathBuf::from("test.sst"), 500);
        assert!(!level.needs_compaction());

        level.add_sstable(PathBuf::from("test2.sst"), 600);
        assert!(level.needs_compaction());
    }

    #[test]
    fn test_lsm_tree_creation() {
        let dir = tempdir().unwrap();
        let lsm = LSMTree::new(dir.path(), 10_000_000, 10, 7);

        assert_eq!(lsm.num_levels(), 7);
        assert_eq!(lsm.level(0).unwrap().level_num(), 0);
        assert_eq!(lsm.level(1).unwrap().size_threshold, 10_000_000);
        assert_eq!(lsm.level(2).unwrap().size_threshold, 100_000_000);
    }

    #[test]
    fn test_l0_compaction_trigger() {
        let dir = tempdir().unwrap();
        let mut lsm = LSMTree::new(dir.path(), 10_000_000, 10, 7);

        assert!(lsm.needs_compaction().is_none());

        // Add 4 SSTables to L0 (triggers compaction)
        for i in 0..4 {
            lsm.add_l0_sstable(PathBuf::from(format!("test{}.sst", i)), 1000);
        }

        assert_eq!(lsm.needs_compaction(), Some(0));
    }

    #[test]
    fn test_level_size_compaction_trigger() {
        let dir = tempdir().unwrap();
        let mut lsm = LSMTree::new(dir.path(), 1000, 10, 7);

        // Add enough data to L1 to trigger compaction
        lsm.levels[1].add_sstable(PathBuf::from("test.sst"), 1200);

        assert_eq!(lsm.needs_compaction(), Some(1));
    }

    #[test]
    fn test_compact_sstables() {
        use crate::sstable::SSTableBuilder;

        let dir = tempdir().unwrap();

        // Build first SSTable
        let path1 = dir.path().join("input1.sst");
        let mut builder1 = SSTableBuilder::create(&path1).unwrap();
        builder1
            .add(Bytes::from("key1"), Bytes::from("value1"))
            .unwrap();
        builder1
            .add(Bytes::from("key3"), Bytes::from("value3"))
            .unwrap();
        builder1.finish().unwrap();

        // Build second SSTable
        let path2 = dir.path().join("input2.sst");
        let mut builder2 = SSTableBuilder::create(&path2).unwrap();
        builder2
            .add(Bytes::from("key2"), Bytes::from("value2"))
            .unwrap();
        builder2
            .add(Bytes::from("key4"), Bytes::from("value4"))
            .unwrap();
        builder2.finish().unwrap();

        // Compact (no active snapshots = u64::MAX means GC everything)
        let output_path = dir.path().join("output.sst");
        let (result_path, size) =
            compact_sstables(&[path1, path2], &output_path, 0, None, u64::MAX).unwrap();

        assert_eq!(result_path, output_path);
        assert!(size > 0);

        // Verify output SSTable has merged data
        let mut output_sst = SSTable::open(&output_path).unwrap();
        assert_eq!(output_sst.len(), 4);

        assert_eq!(
            output_sst.get(b"key1").unwrap(),
            Some(Bytes::from("value1"))
        );
        assert_eq!(
            output_sst.get(b"key2").unwrap(),
            Some(Bytes::from("value2"))
        );
        assert_eq!(
            output_sst.get(b"key3").unwrap(),
            Some(Bytes::from("value3"))
        );
        assert_eq!(
            output_sst.get(b"key4").unwrap(),
            Some(Bytes::from("value4"))
        );
    }

    #[test]
    fn test_compact_with_duplicates() {
        use crate::sstable::SSTableBuilder;

        let dir = tempdir().unwrap();

        // Build older SSTable (like first in L0)
        let path1 = dir.path().join("input1.sst");
        let mut builder1 = SSTableBuilder::create(&path1).unwrap();
        builder1
            .add(Bytes::from("key1"), Bytes::from("old_value"))
            .unwrap();
        builder1
            .add(Bytes::from("key2"), Bytes::from("value2"))
            .unwrap();
        builder1.finish().unwrap();

        // Build newer SSTable (like second in L0)
        let path2 = dir.path().join("input2.sst");
        let mut builder2 = SSTableBuilder::create(&path2).unwrap();
        builder2
            .add(Bytes::from("key1"), Bytes::from("new_value"))
            .unwrap();
        builder2.finish().unwrap();

        // Compact (older first, newer second - like L0 order)
        let output_path = dir.path().join("output.sst");
        compact_sstables(&[path1, path2], &output_path, 0, None, u64::MAX).unwrap();

        // Verify output has newer value (from higher source_id)
        let mut output_sst = SSTable::open(&output_path).unwrap();
        assert_eq!(output_sst.len(), 2);

        assert_eq!(
            output_sst.get(b"key1").unwrap(),
            Some(Bytes::from("new_value"))
        );
        assert_eq!(
            output_sst.get(b"key2").unwrap(),
            Some(Bytes::from("value2"))
        );
    }

    #[test]
    fn test_adaptive_compaction_strategy() {
        let mut strategy = CompactionStrategy::Adaptive {
            current_ratio: 8,
            min_ratio: 4,
            max_ratio: 20,
        };

        // Write-heavy workload (100:1 write:read) → higher ratio
        // Formula: T = sqrt((1.5 * 100000) / 1000) = sqrt(150) ≈ 12
        strategy.adjust_for_workload(100000, 1000);
        let ratio = strategy.current_ratio();
        assert!(
            ratio > 8,
            "Write-heavy should increase ratio, got {}",
            ratio
        );
        assert!(ratio <= 20, "Should respect max ratio");
        assert_eq!(ratio, 12, "Expected ratio ~12 for 100:1 w:r");

        // Read-heavy workload (1:100 write:read) → lower ratio
        // Formula: T = sqrt((1.5 * 101000) / 10100000) ≈ 0.12 → clamped to min_ratio
        strategy.adjust_for_workload(101000, 10100000);
        let ratio = strategy.current_ratio();
        assert!(ratio < 8, "Read-heavy should decrease ratio, got {}", ratio);
        assert_eq!(ratio, 4, "Should clamp to min_ratio for read-heavy");
    }

    #[test]
    fn test_lsm_adaptive_workload_adjustment() {
        let dir = tempdir().unwrap();
        let mut lsm = LSMTree::new_adaptive(dir.path(), 1000, 7, 4, 20);

        // Check initial state
        assert_eq!(lsm.strategy().current_ratio(), 12); // (4+20)/2

        // Simulate write-heavy workload (200:1 write:read)
        // Formula: T = sqrt((1.5 * 200000) / 1000) ≈ 17
        let adjusted = lsm.adjust_for_workload(200000, 1000);
        assert!(adjusted, "Should adjust on significant workload change");

        // Ratio should increase for write-heavy
        let new_ratio = lsm.strategy().current_ratio();
        assert!(new_ratio > 12, "Write-heavy should increase ratio");
        assert_eq!(new_ratio, 17, "Expected ratio ~17 for 200:1 w:r");

        // Verify level thresholds updated
        let l1_threshold = lsm.level(1).unwrap().size_threshold;
        assert_eq!(l1_threshold, 1000); // base_size * ratio^0

        let l2_threshold = lsm.level(2).unwrap().size_threshold;
        assert_eq!(l2_threshold, 1000 * new_ratio); // base_size * ratio^1
    }

    #[test]
    fn test_fixed_strategy_doesnt_adapt() {
        let strategy = CompactionStrategy::Fixed(10);
        assert_eq!(strategy.current_ratio(), 10);

        // Fixed strategy shouldn't change
        let mut strategy_mut = strategy;
        strategy_mut.adjust_for_workload(10000, 1000);
        assert_eq!(strategy_mut.current_ratio(), 10);
    }
}
