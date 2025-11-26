// Metrics collection for production observability
// Tracks throughput, latency, and resource usage with minimal overhead

use hdrhistogram::Histogram;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Type alias for latency percentile tuples (p50, p95, p99, p999)
type LatencyPercentiles = ((u64, u64, u64, u64), (u64, u64, u64, u64), (u64, u64, u64));

/// Database statistics for monitoring and observability
#[derive(Debug, Clone)]
pub struct DBStats {
    // Throughput metrics
    pub writes_per_sec: f64,
    pub reads_per_sec: f64,
    pub deletes_per_sec: f64,

    // Operation counts (lifetime)
    pub total_puts: u64,
    pub total_gets: u64,
    pub total_deletes: u64,
    pub total_flushes: u64,
    pub total_compactions: u64,

    // Latency percentiles (microseconds)
    pub put_latency_p50_us: u64,
    pub put_latency_p95_us: u64,
    pub put_latency_p99_us: u64,
    pub put_latency_p999_us: u64,

    pub get_latency_p50_us: u64,
    pub get_latency_p95_us: u64,
    pub get_latency_p99_us: u64,
    pub get_latency_p999_us: u64,

    pub delete_latency_p50_us: u64,
    pub delete_latency_p95_us: u64,
    pub delete_latency_p99_us: u64,

    // Resource usage
    pub memtable_size_bytes: usize,
    pub memtable_capacity_bytes: usize,
    pub memtable_utilization_pct: f64,
    pub wal_size_bytes: u64,
    pub total_disk_bytes: u64,

    // Block cache performance
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub block_cache_size: usize,     // Current number of cached blocks
    pub block_cache_capacity: usize, // Maximum blocks in cache

    // LSM tree structure
    pub sstables_per_level: Vec<usize>,
    pub level_sizes_bytes: Vec<u64>,
    pub total_sstables: usize,

    // Write amplification tracking
    pub logical_bytes_written: u64, // Bytes written by user (put values)
    pub physical_bytes_written: u64, // Bytes written to disk (WAL, SSTable, vLog, compaction)
    pub write_amplification: f64,   // physical / logical

    // Uptime
    pub uptime_seconds: u64,
}

/// Internal metrics collector with atomic counters
pub(crate) struct MetricsCollector {
    // Operation counters
    pub(crate) total_puts: AtomicU64,
    pub(crate) total_gets: AtomicU64,
    pub(crate) total_deletes: AtomicU64,
    pub(crate) total_flushes: AtomicU64,
    pub(crate) total_compactions: AtomicU64,

    // Write amplification tracking
    pub(crate) logical_bytes_written: AtomicU64, // User data bytes
    pub(crate) physical_bytes_written: AtomicU64, // Disk bytes

    // Latency histograms (require locking)
    pub(crate) put_latencies: std::sync::Mutex<Histogram<u64>>,
    pub(crate) get_latencies: std::sync::Mutex<Histogram<u64>>,
    pub(crate) delete_latencies: std::sync::Mutex<Histogram<u64>>,

    // Start time for uptime calculation
    pub(crate) start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            total_puts: AtomicU64::new(0),
            total_gets: AtomicU64::new(0),
            total_deletes: AtomicU64::new(0),
            total_flushes: AtomicU64::new(0),
            total_compactions: AtomicU64::new(0),

            logical_bytes_written: AtomicU64::new(0),
            physical_bytes_written: AtomicU64::new(0),

            // High dynamic range histograms: 1us to 1 minute, 3 sig figs
            put_latencies: std::sync::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3).expect("Invalid histogram config"),
            ),
            get_latencies: std::sync::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3).expect("Invalid histogram config"),
            ),
            delete_latencies: std::sync::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3).expect("Invalid histogram config"),
            ),

            start_time: Instant::now(),
        }
    }

    /// Record a put operation
    #[inline]
    pub fn record_put(&self, latency: Duration) {
        self.total_puts.fetch_add(1, Ordering::Relaxed);

        // Record latency in microseconds
        let latency_us = latency.as_micros() as u64;
        if let Ok(mut hist) = self.put_latencies.lock() {
            let _ = hist.record(latency_us); // Ignore overflow (clamps to max)
        }
    }

    /// Record a get operation
    #[inline]
    pub fn record_get(&self, latency: Duration) {
        self.total_gets.fetch_add(1, Ordering::Relaxed);

        let latency_us = latency.as_micros() as u64;
        if let Ok(mut hist) = self.get_latencies.lock() {
            let _ = hist.record(latency_us);
        }
    }

    /// Record a delete operation
    #[inline]
    pub fn record_delete(&self, latency: Duration) {
        self.total_deletes.fetch_add(1, Ordering::Relaxed);

        let latency_us = latency.as_micros() as u64;
        if let Ok(mut hist) = self.delete_latencies.lock() {
            let _ = hist.record(latency_us);
        }
    }

    /// Record a flush operation
    #[inline]
    pub fn record_flush(&self) {
        self.total_flushes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a compaction operation
    #[inline]
    #[allow(dead_code)] // Will be used when compaction metrics are added
    pub fn record_compaction(&self) {
        self.total_compactions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record logical bytes written (user data)
    #[inline]
    pub fn record_logical_bytes(&self, bytes: u64) {
        self.logical_bytes_written
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record physical bytes written to disk (WAL, SSTable, vLog, compaction)
    #[inline]
    pub fn record_physical_bytes(&self, bytes: u64) {
        self.physical_bytes_written
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get current operation counts
    pub fn get_counts(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.total_puts.load(Ordering::Relaxed),
            self.total_gets.load(Ordering::Relaxed),
            self.total_deletes.load(Ordering::Relaxed),
            self.total_flushes.load(Ordering::Relaxed),
            self.total_compactions.load(Ordering::Relaxed),
        )
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Calculate throughput (ops/sec)
    pub fn calculate_throughput(&self) -> (f64, f64, f64) {
        let uptime_secs = self.uptime_seconds() as f64;
        if uptime_secs < 0.001 {
            return (0.0, 0.0, 0.0);
        }

        let puts = self.total_puts.load(Ordering::Relaxed) as f64;
        let gets = self.total_gets.load(Ordering::Relaxed) as f64;
        let deletes = self.total_deletes.load(Ordering::Relaxed) as f64;

        (
            puts / uptime_secs,
            gets / uptime_secs,
            deletes / uptime_secs,
        )
    }

    /// Get latency percentiles (microseconds)
    pub fn get_latency_percentiles(&self) -> LatencyPercentiles {
        let put_hist = self.put_latencies.lock().expect("mutex poisoned");
        let get_hist = self.get_latencies.lock().expect("mutex poisoned");
        let delete_hist = self.delete_latencies.lock().expect("mutex poisoned");

        let put_stats = (
            put_hist.value_at_percentile(50.0),
            put_hist.value_at_percentile(95.0),
            put_hist.value_at_percentile(99.0),
            put_hist.value_at_percentile(99.9),
        );

        let get_stats = (
            get_hist.value_at_percentile(50.0),
            get_hist.value_at_percentile(95.0),
            get_hist.value_at_percentile(99.0),
            get_hist.value_at_percentile(99.9),
        );

        let delete_stats = (
            delete_hist.value_at_percentile(50.0),
            delete_hist.value_at_percentile(95.0),
            delete_hist.value_at_percentile(99.0),
        );

        (put_stats, get_stats, delete_stats)
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector_basic() {
        let collector = MetricsCollector::new();

        // Record some operations
        collector.record_put(Duration::from_micros(100));
        collector.record_put(Duration::from_micros(200));
        collector.record_get(Duration::from_micros(50));
        collector.record_delete(Duration::from_micros(150));

        let (puts, gets, deletes, flushes, compactions) = collector.get_counts();
        assert_eq!(puts, 2);
        assert_eq!(gets, 1);
        assert_eq!(deletes, 1);
        assert_eq!(flushes, 0);
        assert_eq!(compactions, 0);
    }

    #[test]
    fn test_metrics_latency_percentiles() {
        let collector = MetricsCollector::new();

        // Record 100 operations with increasing latencies
        for i in 1..=100 {
            collector.record_put(Duration::from_micros(i * 10));
        }

        let (put_stats, _, _) = collector.get_latency_percentiles();
        let (p50, p95, p99, _p999) = put_stats;

        // p50 should be around 500us (50th value is 500us)
        assert!((400..=600).contains(&p50), "p50: {}", p50);

        // p95 should be around 950us
        assert!((900..=1000).contains(&p95), "p95: {}", p95);

        // p99 should be around 990us
        assert!((980..=1010).contains(&p99), "p99: {}", p99);
    }

    #[test]
    fn test_metrics_throughput() {
        let collector = MetricsCollector::new();

        // Record operations
        for _ in 0..100 {
            collector.record_put(Duration::from_micros(10));
        }

        // Wait after operations to ensure uptime > 0
        thread::sleep(Duration::from_secs(1));

        let (writes_per_sec, _, _) = collector.calculate_throughput();

        // Should be less than 200 ops/sec (100 ops in >1s)
        // We just want to ensure it calculates properly, not test timing precision
        assert!(
            writes_per_sec > 0.0 && writes_per_sec <= 200.0,
            "writes_per_sec: {}",
            writes_per_sec
        );
    }

    #[test]
    fn test_metrics_concurrent() {
        let collector = std::sync::Arc::new(MetricsCollector::new());
        let mut handles = vec![];

        // Spawn 10 threads recording operations
        for _ in 0..10 {
            let c = collector.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    c.record_put(Duration::from_micros(50));
                    c.record_get(Duration::from_micros(25));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let (puts, gets, _, _, _) = collector.get_counts();
        assert_eq!(puts, 1000);
        assert_eq!(gets, 1000);
    }

    #[test]
    fn test_metrics_uptime() {
        let collector = MetricsCollector::new();

        thread::sleep(Duration::from_millis(100));

        let uptime = collector.uptime_seconds();
        assert!(uptime == 0); // Less than 1 second

        thread::sleep(Duration::from_secs(1));

        let uptime = collector.uptime_seconds();
        assert!((1..=2).contains(&uptime));
    }
}
