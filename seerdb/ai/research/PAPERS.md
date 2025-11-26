# PAPERS - Research Paper Summaries

**Reading Protocol**:
1. Read abstract and intro
2. Study key figures
3. Read evaluation section
4. Summarize: Key idea, Application to seerdb, Complexity, Priority
5. Add references

---

## Phase 1: Foundational Papers

### ✅ "The Case for Learned Index Structures" (Kraska et al., MIT 2018)

**Key Idea**: Replace traditional indexes (B-trees, hash tables, bloom filters) with ML models

**Core Insight**: Data structures are models of data distribution
- B-tree: Approximates cumulative distribution function (CDF) of keys
- Hash table: Assumes uniform distribution
- Bloom filter: Models set membership

If data has patterns → ML models can exploit them better than generic structures

**Results**:
- 10-100x space reduction vs B-trees
- Similar or better lookup speed
- Training cost amortized over queries

**Application to seerdb**:
- Learned bloom filters: Replace traditional blooms in SSTables
- Learned index: Replace binary search in SSTable index blocks
- Workload-aware: Train models on actual data patterns

**Complexity**: Medium (need ML framework, model training pipeline)

**Priority**: Must-have (foundational concept)

**References**:
- ALEX paper (updatable learned indexes)
- Learned Bloom Filters (Kraska et al., 2018)
- Neural Data Structures (broader ML + DS research)

**Notes**:
- Focus on B-trees (not LSM trees directly)
- Read-only indexes (static data)
- Need updatable version (ALEX paper)

---

## Phase 1: Remaining Papers

### ✅ "ALEX: An Updatable Adaptive Learned Index" (Ding et al., MIT/Columbia 2020)

**Published**: SIGMOD 2020
**Authors**: Jialin Ding, Umar Farooq Minhas, Jia Yu, Chi Wang, Jaeyoung Do, Yinan Li, Hantian Zhang, Badrish Chandramouli, Johannes Gehrke, Donald Kossmann, David Lomet, Tim Kraska
**Paper**: https://arxiv.org/abs/1905.08898

**Key Idea**: Practical learned index that handles dynamic workloads (inserts, updates, deletes) using gapped arrays

**Core Innovation**: Extends learned indexes beyond read-only workloads by combining ML model predictions with proven storage techniques
- Uses **gapped array structure** for efficient in-place insertions
- Models predict key positions in dataset (like CDF approximation)
- Dynamically adapts as data distribution changes
- Hierarchical structure (internal nodes + data nodes with gaps)

**Results**:
- **Read-only**: 2.2x faster than original learned index, 15x smaller index size
- **Read-write**: Up to 4.1x faster than B+trees across all workloads
- **Space**: Up to 2000x smaller memory footprint vs B+trees
- **Guarantee**: Never performs worse than B+trees (robust worst-case)

**How it Works**:
1. Internal nodes use linear regression models to route searches
2. Data nodes store keys in gapped arrays (allows inserts without full reorganization)
3. When gaps fill up, nodes split or expand
4. Models retrain periodically as data distribution shifts

**Application to seerdb**:
- **SSTable index**: Use ALEX-style learned index instead of binary search
- **Memtable**: Potential alternative to skiplist (need to benchmark)
- **Adaptive**: Models adapt to changing workload patterns (aligns with seerdb goals)
- **Code availability**: Archived implementation in organization/ can be adapted for seerdb

**Implementation Complexity**: Medium-High
- Gapped array management
- Model training and retraining logic
- Node splitting/expansion strategies
- Need to handle worst-case degradation gracefully

**Priority**: Must-have (practical, proven in mixed workloads)

**Trade-offs**:
- ✅ Handles updates/deletes (critical for seerdb)
- ✅ Better than B+trees in all cases
- ✅ Massive space savings
- ❌ More complex than traditional B+tree
- ❌ Model retraining overhead (need to measure)
- ❌ Gap management adds complexity

**Notes**:
- Code archived in organization/ (can use when needed for seerdb)
- Microsoft Research backing (production-quality implementation exists)
- Good fit for SSTable index blocks (sorted, mostly immutable after compaction)
- Not currently in database production (available for seerdb implementation)

---

### ✅ "Learned Bloom Filters" (Mitzenmacher 2018, Kraska et al. 2018)

**Published**: NeurIPS 2018 (Mitzenmacher), SIGMOD 2018 (Kraska et al.)
**Papers**:
- "A Model for Learned Bloom Filters, and Optimizing by Sandwiching" (Mitzenmacher)
- "The Case for Learned Index Structures" (Kraska et al. - includes LBF section)
**ArXiv**: https://arxiv.org/abs/1901.00902

**Key Idea**: Use ML classifier to predict set membership, backed by small traditional bloom filter

**Core Innovation**: Membership queries as binary classification problem
- ML model classifies element as "definitely in set" or "uncertain"
- High-confidence predictions (score > threshold) → return immediately
- Low-confidence predictions → check backup bloom filter
- Backup filter only stores keys model is uncertain about (much smaller)

**Architecture Variants**:

1. **Basic Learned Bloom Filter (LBF)**:
   ```
   Query → ML Model → Score > threshold?
                     ↓ Yes: Return "in set"
                     ↓ No: Check backup bloom filter
   ```

2. **Sandwiched LBF** (Mitzenmacher optimization):
   ```
   Query → Initial Bloom Filter → ML Model → Backup Bloom Filter
   ```
   - Initial filter removes most non-members (cheap hash checks)
   - Model removes false positives from initial filter
   - Backup filter handles model false negatives

3. **Partitioned LBF (PLBF)**:
   - Multiple thresholds and separate backup filters per region
   - 4-6 partitions sufficient for best performance

**Results**:
- **Space savings**: 17-97% reduction vs traditional bloom filters (depends on variant and data)
- **Original claim**: ~90% space reduction (Kraska et al.)
- **False positive rate**: Same or better than traditional bloom filters
- **Query time**: Model inference + backup filter lookup (tradeoff)

**ML Models Used**:
- **Simple**: Decision trees, logistic regression (fast inference)
- **Medium**: Gradient boosting trees (GBT) - good balance
- **Complex**: Small neural networks (higher accuracy, slower)
- **Recommendation**: Start with GBT or decision trees for seerdb

**Training Process**:
1. Positive examples: Keys in the set (SSTable)
2. Negative examples: Random keys NOT in set (need to generate)
3. Train binary classifier on features of keys
4. Choose threshold to balance FPR and backup filter size
5. Build backup bloom filter for keys below threshold

**Application to seerdb**:
- **SSTable bloom filters**: Replace traditional blooms in each SSTable
- **Training time**: During compaction (keys known, can sample negatives)
- **Model per SSTable**: Each SSTable gets own trained model
- **Fallback**: If model fails or too expensive, use traditional bloom

**Implementation Complexity**: Medium
- Binary classifier (use existing ML library)
- Negative sampling (generate realistic non-member keys)
- Backup bloom filter (already need traditional implementation)
- Threshold tuning (optimize FPR vs space)

**Priority**: Must-have (Week 1 prototype target, high impact)

**Trade-offs**:
- ✅ 90%+ space savings (huge for large databases)
- ✅ Same FPR guarantees as traditional blooms
- ✅ Simple models (decision trees) sufficient
- ❌ Model inference slower than hash functions (need to benchmark)
- ❌ Training overhead during compaction
- ❌ Need negative samples (don't have real non-members)

**Practical Considerations**:
- **Negative sampling**: Generate random keys or use Zipf distribution
- **Model size**: Must fit in memory alongside backup filter
- **Inference speed**: Critical path for point queries (measure carefully)
- **Retraining**: Model trained once per SSTable, doesn't change after

**Notes**:
- Good first prototype (simpler than learned index)
- Can validate space savings claim quickly
- SSTable blooms are immutable (don't need update handling)
- Even 50% space savings is significant (90% may be optimistic)

---

## Phase 2: LSM Tree Papers

### ✅ "WiscKey: Separating Keys from Values in SSD-Conscious Storage" (Lu et al., Wisconsin 2016)

**Published**: USENIX FAST 2016
**Authors**: Lanyue Lu, Thanumalayan Sankaranarayana Pillai, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau
**Institution**: University of Wisconsin—Madison
**Paper**: https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf

**Key Idea**: Separate keys from values in LSM-tree to minimize write amplification, optimized for SSD characteristics

**Core Innovation**: Key-Value Separation Architecture
- **LSM Tree**: Stores only keys + value references (offset + length in vLog)
- **Value Log (vLog)**: Append-only log storing actual values sequentially
- During compaction: Only keys rewritten, values stay in vLog
- Reduces compaction overhead from full KV pairs to just keys

**Architecture**:
```
Write: Key → LSM-tree (key + vLog pointer)
       Value → vLog (append-only, sequential)

Read:  LSM-tree lookup → Get vLog pointer → Read value from vLog

Compaction: Only compact keys in LSM-tree (values untouched)
```

**Results**:
- **Database loading**: 2.5x - 111x faster than LevelDB
- **Random lookups**: 1.6x - 14x faster than LevelDB
- **Write amplification**: Reduced by ~2 orders of magnitude (10-100x reduction)
- **LSM size reduction**: 100GB dataset → 2GB LSM tree (16B keys, 1KB values)
- **YCSB workloads**: Outperforms LevelDB and RocksDB across all 6 workloads

**Size Reduction Example**:
- Key: 16 bytes
- Value: 1KB
- Dataset: 100GB
- LSM tree after separation: ~2GB (50x smaller)
- Benefit: More keys fit in memory, fewer levels to traverse

**Garbage Collection**:
- **Mechanism**: Head-tail tracking on vLog
- Values between head and tail are valid range
- GC reads from tail, checks LSM for validity, writes valid entries to head
- **Overhead**: Not a bottleneck (70x faster than LevelDB even with GC running)
- **Trigger**: When vLog space threshold reached or old entries accumulate

**Range Query Optimization**:
- **Problem**: Scattered values in vLog cause random I/O during range scans
- **Solution**: Parallel prefetching
  - Background threads detect access patterns
  - Queue value addresses sequentially
  - Fetch values concurrently (exploit SSD parallelism)
- **Result**: Maintains range query performance despite value separation

**Application to seerdb**:
- **database vectors**: Large embeddings (512-4096 bytes) → perfect for KV separation
- **Threshold**: Values >4KB separated (need to tune for workload)
- **LSM tree**: Stays small (more fits in cache, faster compaction)
- **vLog**: Sequential writes (SSD-friendly), parallel reads

**Implementation Complexity**: Medium
- Append-only vLog (simple writes)
- Garbage collection (head-tail tracking, validity checks)
- Parallel prefetching for range queries
- Crash recovery (vLog head pointer persistence)

**Priority**: Must-have (critical for database large value workload)

**Trade-offs**:
- ✅ Write amplification: 10-100x reduction
- ✅ Compaction: Much faster (only keys)
- ✅ LSM size: 50x smaller (more in cache)
- ✅ SSD-optimized: Sequential vLog writes, parallel reads
- ❌ Space amplification: Higher (garbage in vLog until GC)
- ❌ Crash recovery: Slower (2.6s vs 0.7s for LevelDB)
- ❌ Range scans: Random I/O without prefetching
- ❌ Small values: No benefit (separation overhead not worth it)

**When to Use KV Separation**:
- ✅ Large values (>1KB) that dominate storage
- ✅ Write-heavy workloads (compaction bottleneck)
- ✅ SSDs (can exploit parallel reads)
- ❌ Small values (<256B) - separation overhead too high
- ❌ Range-scan heavy workloads (unless prefetching implemented)

**Industrial Adoption**:
- **BlobDB** (RocksDB component)
- **Titan** (RocksDB 6.x plugin)
- **TerarkDB** (RocksDB 5.x fork with v-SST index)
- **BadgerDB** (Go-based, closest to original WiscKey)

**Variants & Improvements**:
- **TerarkDB**: Maintains dependency relationships, avoids LSM rewrites during GC
- **Titan**: Level merge for last 2 levels
- **BadgerDB**: DISCARD file for garbage tracking, vLog rewriting

**Value Size Thresholds in Practice**:
- BadgerDB: 4KB default
- TerarkDB: 512B default
- seerdb target: 4KB (tune based on database workload analysis)

**Notes**:
- Fundamental trade-off: Write amplification vs space amplification
- Perfect fit for database vectors (large embeddings, append-heavy)
- Need careful tuning: threshold, GC frequency, prefetch strategy
- Should benchmark with real database workload before committing

---

### ✅ "Dostoevsky: Better Space-Time Trade-Offs for LSM-Tree Based Key-Value Stores" (Dayan & Idreos, Harvard 2018)

**Published**: ACM SIGMOD 2018
**Authors**: Niv Dayan, Stratos Idreos
**Institution**: Harvard University
**Paper**: https://scholar.harvard.edu/files/stratos/files/dostoevskykv.pdf

**Key Idea**: Optimize LSM-tree compaction by removing superfluous merging operations through Lazy Leveling

**Core Problem**: Mainstream LSM-trees suboptimally trade between:
- Update I/O cost (writes)
- Lookup I/O cost (reads)
- Storage space amplification

Traditional LSMs perform equally expensive merge operations across ALL levels, but most merges provide negligible benefit while significantly increasing update costs.

**Core Innovation: Lazy Leveling**

Remove merge operations from all levels **except the largest level**

**Strategy Comparison**:

1. **Tiered Compaction**:
   - Merges entire levels when size threshold reached
   - Multiple overlapping runs per level
   - **Write amp**: Low (good for write-heavy workloads)
   - **Read amp**: High (must check all runs per level)
   - **Space amp**: O(T) - very high (at T=4, 1.2GB → 9.3GB)
   - **Use case**: Update-intensive workloads

2. **Leveled Compaction** (RocksDB default):
   - Each level is T times larger than previous
   - Disjoint (non-overlapping) key ranges per level (except L0)
   - **Write amp**: High (at T=10, writes 11x original data size)
   - **Read amp**: Low (worst case: L0 + level_count - 1 I/Os)
   - **Space amp**: ~11% (at T=10) - very good
   - **Use case**: Lookup-intensive workloads, range queries

3. **Lazy Leveling** (Dostoevsky):
   - **Largest level**: Leveled compaction (merge everything)
   - **All other levels**: Tiered compaction (allow multiple runs)
   - **Write amp**: Intermediate (better than leveled)
   - **Read amp**: Intermediate (better than tiered)
   - **Space amp**: Better than tiered, similar to leveled
   - **Use case**: Mixed workloads (balanced read/write)

**Fluid LSM-Tree Design Space**

Generalization of entire LSM-tree design space:
- Can parameterize to assume **any** existing design
- Navigate design space based on workload and hardware
- Optimize more for updates (merge less at largest level)
- Optimize more for range queries (merge more at other levels)
- **Adaptive tuning**: Dynamically calculate optimal configuration during execution

**Level Ratio (T) Tuning**

- **T**: Size ratio between consecutive levels (typically 4-10)
- **T=10**: Common in practice (RocksDB default)
- **High T**: Faster reads, higher write amplification
- **Low T**: Slower reads, lower write amplification
- **Dostoevsky approach**: Calculate optimal T based on workload mix

**Write Amplification Formula** (Leveled):
- At T=10: Write amplification ≈ 11x
- When level overflows: Must rewrite all overlapping data in next level
- Example: L1→L2 compaction with 128MB and fanout=8 rewrites 320MB

**Performance Results**:
- Implemented on top of RocksDB
- **Strictly dominates** state-of-the-art designs in performance and storage
- Adaptive: Removes superfluous merging based on workload
- Closed-form performance model for throughput maximization

**Application to seerdb**:

**Workload Analysis**:
- **database vectors**: Append-heavy writes + range scans (vector search top-K)
  - → **Lazy Leveling** (balance writes and range queries)
- **queue applications**: High write throughput, FIFO access
  - → **Tiered** (optimize for writes)
- **database time series**: Append-only + time-range queries
  - → **Lazy Leveling** (range queries important)

**Configuration Strategy**:
- Start with **Lazy Leveling** (best for mixed workloads)
- Level ratio **T=10** (standard, tune later)
- Largest level: Leveled compaction (disjoint for range queries)
- Other levels: Tiered (reduce write amplification)
- **Adaptive tuning**: Implement Dostoevsky's workload-aware optimization later (Phase 3)

**Implementation Complexity**: Medium-High
- Need both tiered and leveled compaction logic
- Level size tracking and overflow detection
- Adaptive tuning requires workload profiling
- Closed-form performance model (math heavy)

**Priority**: Must-have (critical for performance tuning)

**Trade-offs**:
- ✅ Better write amp than leveled (fewer merges in upper levels)
- ✅ Better read amp than tiered (largest level is disjoint)
- ✅ Better space amp than tiered (largest level compact)
- ✅ Adaptive to workload changes
- ❌ More complex than pure leveled or tiered
- ❌ Need to implement both compaction strategies
- ❌ Optimal tuning requires workload profiling

**Key Formulas**:

- **Point lookup cost**: O(L0_segments + level_count - 1) with bloom filters
- **Range query cost**: Lower with disjoint largest level
- **Update cost**: Reduced by avoiding merges in upper levels
- **Space amplification**: ~11% (similar to leveled)

**Real-World Implementations**:
- **Leveled**: RocksDB, CockroachDB Pebble, BadgerDB, Fjall (default)
- **Tiered**: Cassandra, ScyllaDB
- **Lazy Leveling**: Dostoevsky (research prototype on RocksDB)

**Notes**:
- Dostoevsky provides mathematical framework for LSM tuning
- Lazy Leveling is sweet spot for most workloads
- database workload (append-heavy + range scans) fits Lazy Leveling perfectly
- Should implement adaptive tuning in Phase 3 (workload-aware)
- Start simple (fixed T=10, Lazy Leveling), optimize later

---

### ✅ "PebblesDB: Building Key-Value Stores using Fragmented Log-Structured Merge Trees" (Raju et al., Wisconsin 2017)

**Published**: ACM SOSP 2017
**Authors**: Pandian Raju, Rohan Kadekodi, Vijay Chidambaram, Ittai Abraham
**Institution**: University of Texas at Austin, VMware Research
**Paper**: https://www.semanticscholar.org/paper/PebblesDB-Raju-Kadekodi/6370a252951f5bdbf7a313528cc8a46b02d05825
**Code**: https://github.com/utsaslab/pebblesdb

**Key Idea**: Reduce write amplification by fragmenting LSM-tree levels using guards (skip-list inspired)

**Core Problem**: Traditional LSM-trees suffer high write amplification due to:
- Maintaining disjoint key ranges within each level
- Rewriting all overlapping sstables during compaction
- Same data rewritten multiple times as it moves down levels

**Core Innovation: Fragmented LSM (FLSM)**

Discard the disjoint key-range invariant within levels. Instead:
- Use **guards** to divide key space into disjoint units
- Allow multiple overlapping sstables within each guard
- Append fragments during compaction (no rewriting within level)

**How Guards Work**:

1. **Key Space Division**:
   - Each guard Gi has associated key Ki
   - Guards divide level into disjoint boundaries
   - SStables cannot cross guard boundaries
   - SStables within same guard CAN overlap

2. **Skip-List Inspiration**:
   - Guards increase in granularity at deeper levels
   - If key is guard at level i, it's guard in all higher levels
   - Similar to skip list pointers across levels

3. **Compaction Process**:
   ```
   Level i → Level i+1 compaction:
   1. Merge-sort sstables in guard
   2. Partition by next level's guards
   3. Append new sstables to correct guards
   4. NO rewriting of existing level i+1 sstables
   ```

**Traditional LSM vs FLSM**:

| Aspect | Traditional LSM | FLSM (PebblesDB) |
|--------|-----------------|------------------|
| Key ranges | Disjoint within level | Overlapping within guards |
| Compaction | Rewrite overlapping sstables | Append new fragments |
| Write amp | High (2.4-3x more) | Low (baseline) |
| Read lookup | 1 sstable per level | All sstables in guard |
| Read latency | Lower | Higher |

**Performance Results**:
- **Write amplification**: 2.4-3x reduction vs RocksDB
- **Write throughput**: 6.7x faster than RocksDB
- **MongoDB integration**: 18-105% higher throughput, 35-55% less write I/O
- **HyperDex integration**: Similar improvements with YCSB benchmark

**Write Amplification Reduction Mechanism**:

Traditional LSM (at T=10): Writes 11x original data
- Level 0→1: Write 1x, read/write 10x = 11x total
- Same data rewritten at each level

FLSM: Only append, no same-level rewrites
- Level 0→1: Write 1x (split by guards)
- Guards accept fragments without merging existing sstables
- Result: 2.4-3x less total I/O

**Read Performance Impact**:

**Trade-off**: More sstables to check per level
- Traditional: 1 sstable per level (disjoint)
- FLSM: Multiple sstables per guard

**Mitigation**:
- Bloom filters on each sstable (quick key presence test)
- Seek-based compaction (merge guards with many sstables)
- Parallel seek for range queries

**Application to seerdb**:

**When PebblesDB Fits**:
- ✅ Write-heavy workloads (queue applications: high enqueue rate)
- ✅ Point lookups with bloom filters
- ❌ NOT for range-scan heavy workloads (database vectors)
- ❌ Read latency critical applications

**Comparison**:
- **WiscKey**: Better for large values (KV separation)
- **Dostoevsky**: Better for mixed workloads (Lazy Leveling)
- **PebblesDB**: Better for pure write-heavy workloads

**For seerdb**:
- **database vectors**: Dostoevsky Lazy Leveling (range scans important)
- **queue applications**: PebblesDB OR Tiered (write throughput critical, small values)
- Likely choose Lazy Leveling (more general, similar write amp benefits)

**Implementation Complexity**: Medium
- Guard management (similar to skip list)
- Fragment appending during compaction
- Bloom filters per sstable (already needed)
- Seek-based compaction for guard merging
- Simpler than full adaptive tuning (Dostoevsky)

**Priority**: Nice-to-have (alternative to Lazy Leveling)

**Trade-offs**:
- ✅ Write amplification: 2.4-3x reduction vs RocksDB
- ✅ Write throughput: 6.7x faster
- ✅ Simpler than Dostoevsky adaptive tuning
- ❌ Read latency: Higher (multiple sstables per guard)
- ❌ Range queries: Slower than leveled compaction
- ❌ More sstables = more file handles, memory overhead

**Key Insight**: Discarding disjoint key-range invariant is the key to reducing write amplification

**Notes**:
- Complementary to WiscKey (can combine both techniques)
- Alternative to Dostoevsky Lazy Leveling
- Better for pure write workloads, worse for reads
- database workload has range queries (vector search) → Lazy Leveling better fit
- Consider for queue applications (write-heavy, no range scans)
- Implementation on HyperLevelDB shows it's production-ready

---

## Phase 3: Workload-Aware Papers

### ✅ "From WiscKey to Bourbon: A Learned Index for Log-Structured Merge Trees" (Dai et al., Wisconsin/Microsoft 2020)

**Published**: USENIX OSDI 2020
**Authors**: Yifan Dai, Yien Xu, Aishwarya Ganesan, Ramnatthan Alagappan, Brian Kroth, Andrea Arpaci-Dusseau, Remzi Arpaci-Dusseau
**Institution**: University of Wisconsin-Madison, Microsoft Gray Systems Lab
**Paper**: https://www.usenix.org/conference/osdi20/presentation/dai
**ArXiv**: https://arxiv.org/abs/2005.14213

**Key Idea**: Use piecewise linear regression to learn key distributions in immutable LSM files for faster lookups

**Core Innovation**: File-level learned index for LSM-trees
- **Piecewise Linear Regression (PLR)**: Lightweight models for key distribution
- **Cost-Benefit Analyzer (CBA)**: Decides when learning is worthwhile
- **File-level learning**: Train models on immutable SSTables after compaction
- **Adaptive**: Only learns on files that will live long enough to benefit

**How It Works**:
1. SSTable becomes immutable after compaction
2. CBA waits 50ms to ensure file won't be quickly deleted
3. Background thread trains PLR model on key distribution
4. Model predicts key position, reduces search space
5. Fallback to traditional binary search if model fails

**Performance Results**:
- **Lookup improvement**: 1.23x - 1.78x vs production LSMs
- **Read-heavy workloads**: Best performance gains
- **Fast storage**: Better on in-memory/Optane vs SATA SSDs
- **Low-cardinality data**: Models compress well

**Piecewise Linear Regression Benefits**:
- **Low training overhead**: Fast model construction
- **Low inference cost**: Simple linear prediction
- **Small space overhead**: Compact model representation
- **Good fit for sorted data**: LSM SSTables are sorted

**Cost-Benefit Analyzer (CBA)**:
- **Training cost (C_model)**: Time to build PLR model
- **Expected benefit**: Based on file size, access patterns, storage speed
- **Decision**: Only learn if benefit > cost
- **Adaptive**: Skips learning on short-lived files (high write workloads)

**Comparison to ALEX**:
- **ALEX**: Updatable, gapped arrays, handles inserts/deletes
- **Bourbon**: Immutable only, file-level, simpler model
- **ALEX use case**: In-memory indexes, mutable data structures
- **Bourbon use case**: LSM SSTable files (immutable after write)

**Application to seerdb**:

**When Bourbon Fits**:
- ✅ Immutable SSTables (LSM files after compaction)
- ✅ Read-heavy workloads (lookups dominate)
- ✅ Fast storage (NVMe, Optane)
- ❌ Write-heavy workloads (files short-lived, wasted learning)
- ❌ Frequent compaction (models discarded)

**For seerdb Decision**:
- **ALEX preferred** for SSTable index (code available, handles updates)
- **Bourbon lessons**: Cost-Benefit Analyzer concept useful
- **Adaptive learning**: Don't train models on every SSTable (waste)
- **Apply CBA logic**: Only index long-lived SSTables in lower levels

**Implementation Complexity**: Medium
- Piecewise linear regression (simpler than ALEX)
- Cost-benefit analysis (workload profiling)
- Background training threads
- Model serialization/deserialization

**Priority**: Nice-to-have (alternative approach, ALEX better for seerdb)

**Trade-offs**:
- ✅ Simpler than ALEX (immutable only)
- ✅ 1.23-1.78x lookup improvement
- ✅ Adaptive learning (CBA avoids waste)
- ✅ Low overhead (lightweight models)
- ❌ Immutable data only (no updates)
- ❌ Write-heavy workloads: minimal benefit
- ❌ Short-lived files: wasted learning effort
- ❌ Random workloads: models help less

**Key Insight**: Cost-Benefit Analyzer is the innovation - knowing **when NOT to learn** is as important as the learning itself

**Notes**:
- Bourbon builds on WiscKey (key-value separation)
- CBA concept applicable to any learned component
- For seerdb: Use ALEX, apply Bourbon's CBA logic
- Don't train learned bloom filters or indexes on upper LSM levels (short-lived)
- Focus learning effort on largest level (long-lived files)

---

### ~~"Tucana" (Liu et al., Tsinghua 2020)~~ - **DOES NOT EXIST**

**Note**: The original research list referenced "Tucana (Liu et al., Tsinghua 2020)" about learned LSM trees. This paper does not exist.

**Actual Tucana Paper**:
- **Title**: "Tucana: Design and Implementation of a Fast and Efficient Scale-up Key-value Store"
- **Authors**: Papagiannis et al., FORTH Greece (not Liu et al., Tsinghua)
- **Published**: USENIX ATC 2016 (not 2020)
- **Focus**: Bε-tree, CPU efficiency, NOT learned data structures
- **Not relevant** to seerdb (different focus than research goals)

**Correction**: The workload-aware/learned LSM paper for Phase 3 is **Bourbon** (above), not "Tucana 2020"

---

## Phase 4: Modern Hardware Papers

### "FASTER: A Concurrent Key-Value Store with In-Place Updates" (Chandramouli et al., Microsoft 2018)

**Status**: Not yet read
**Priority**: Nice-to-have (concurrency patterns)

---

### io_uring Documentation

**Status**: ~~Not needed - Security decision made~~
**Priority**: ~~Nice-to-have~~ **DECIDED: Not using by default**

**Decision Made**: Use tokio async I/O by default, io_uring opt-in only
- **Security concerns**: 77 CVEs, 60% of 2022 kernel exploits
- **Risk**: Privilege escalation, use-after-free, reference counting bugs
- **Implementation**: tokio default, io_uring behind feature flag (opt-in)
- See ai/DECISIONS.md for full rationale

---

## Additional Papers (Discovered While Reading)

### "Partitioned Learned Bloom Filter" (Dai & Shrivastava, 2020)

**Published**: ICLR 2021
**Status**: Discovered during LBF research
**Priority**: Nice-to-have (optimization of basic LBF)

**Key Idea**: Use multiple thresholds and separate backup filters for different regions
- 4-6 partitions optimal
- Better space/FPR trade-off than basic LBF
- Worth exploring after basic LBF works

---

### "Stable Learned Bloom Filters for Data Streams" (Liu et al., 2020)

**Published**: VLDB 2020
**Status**: Discovered during LBF research
**Priority**: Future (for streaming workloads)

**Key Idea**: Adapt learned bloom filters to dynamic data streams
- Handles changing data distributions over time
- Relevant for queue workloads (high throughput, dynamic)
- Consider for queue applications integration

---

*Update as papers are read - maintain checklist at top*

**Papers Read**: 7/9 (78% complete) - Note: 1 paper removed (fake reference)
- Phase 1 Foundational: 3/3 ✅ Complete (Learned Indexes, ALEX, Learned Bloom Filters)
- Phase 2 LSM Trees: 3/3 ✅ Complete (WiscKey, Dostoevsky, PebblesDB)
- Phase 3 Workload-Aware: 1/1 ✅ Complete (Bourbon) - Tucana 2020 removed (doesn't exist)
- Phase 4 Modern Hardware: 0/1 (FASTER remaining) - io_uring decided against
