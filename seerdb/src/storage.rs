// Storage backend abstraction for pluggable storage tiers
//
// Feature-gated: Cloud storage backends only compiled when --features object-store
// Default (no feature): Uses concrete LocalStorage (zero overhead)
// With feature: Generic Storage trait + ObjectStoreBackend (cloud support)

use crate::db::Result;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[cfg(feature = "object-store")]
use std::sync::Arc;

#[cfg(feature = "object-store")]
use rand::Rng;

/// Retry configuration for object store operations
///
/// Controls retry behavior for transient failures (network timeouts, 503, etc.)
#[cfg(feature = "object-store")]
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, fail immediately)
    pub max_attempts: u32,
    /// Base delay in milliseconds for exponential backoff
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds (prevents excessive waits)
    pub max_delay_ms: u64,
    /// Enable jitter to prevent thundering herd (adds random 0-50% to delay)
    pub jitter: bool,
}

#[cfg(feature = "object-store")]
impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            jitter: true,
        }
    }
}

#[cfg(feature = "object-store")]
impl RetryConfig {
    /// No retries - fail immediately on any error
    pub fn none() -> Self {
        Self {
            max_attempts: 0,
            ..Default::default()
        }
    }

    /// Aggressive retries for unreliable networks
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            base_delay_ms: 50,
            max_delay_ms: 10000,
            jitter: true,
        }
    }

    /// Calculate delay for attempt N with exponential backoff and optional jitter
    fn delay_for_attempt(&self, attempt: u32) -> std::time::Duration {
        let base_delay = self.base_delay_ms * 2u64.pow(attempt);
        let delay = std::cmp::min(base_delay, self.max_delay_ms);

        if self.jitter {
            let jitter_range = delay / 2; // 0-50% jitter
            let jitter = rand::thread_rng().gen_range(0..=jitter_range);
            std::time::Duration::from_millis(delay + jitter)
        } else {
            std::time::Duration::from_millis(delay)
        }
    }
}

/// Classify object_store errors into retryable vs permanent
#[cfg(feature = "object-store")]
fn is_retryable_error(err: &object_store::Error) -> bool {
    use object_store::Error;

    match err {
        // Transient network errors - should retry
        Error::Generic { source, .. } => {
            let msg = source.to_string().to_lowercase();
            msg.contains("timeout")
                || msg.contains("connection reset")
                || msg.contains("connection refused")
                || msg.contains("temporarily unavailable")
        }
        // Permanent errors - fail fast
        Error::NotFound { .. } => false,
        Error::AlreadyExists { .. } => false,
        Error::Precondition { .. } => false,
        Error::NotModified { .. } => false,
        Error::InvalidPath { .. } => false,
        Error::UnknownConfigurationKey { .. } => false,
        // Conservative: treat unknown errors as non-retryable
        _ => false,
    }
}

/// Storage trait for pluggable storage implementations
///
/// Enables:
/// - Local disk (default, always available)
/// - Object storage: S3, GCS, Azure, R2 (requires --features s3-backend)
/// - Custom backends (encryption, compression, tiering)
///
/// Performance: Monomorphized generics ensure zero runtime overhead
///
/// # Zero-Cost Abstraction
///
/// When `object-store` feature is disabled (default):
/// - No trait overhead, direct function calls
/// - LocalStorage methods are inlined by compiler
/// - Binary size identical to hand-written file I/O
///
/// When `object-store` feature is enabled:
/// - Trait uses monomorphized generics (static dispatch)
/// - Compiler optimizes as if no trait exists
/// - Zero runtime cost vs direct implementation
#[cfg(feature = "object-store")]
pub trait Storage: Send + Sync {
    /// Read a block from an SSTable at the given offset
    ///
    /// Returns the raw bytes (caller handles decompression/parsing)
    fn read_block(&self, path: &Path, offset: u64, size: u32) -> Result<Vec<u8>>;

    /// Write an SSTable to storage
    ///
    /// Data should be fully buffered by caller before calling this
    fn write_sstable(&self, path: &Path, data: &[u8]) -> Result<()>;

    /// Delete an SSTable from storage
    fn delete_sstable(&self, path: &Path) -> Result<()>;

    /// Fsync an SSTable to ensure durability
    fn sync(&self, path: &Path) -> Result<()>;

    /// Check if an SSTable exists
    fn exists(&self, path: &Path) -> Result<bool>;

    /// List all SSTables in a directory
    fn list_sstables(&self, dir: &Path) -> Result<Vec<PathBuf>>;
}

/// Local disk storage implementation
///
/// Direct file system operations with zero abstraction overhead.
///
/// # Performance
///
/// All methods are inlined by the compiler when possible, resulting in
/// performance identical to hand-written file I/O code.
pub struct LocalStorage {
    base_path: PathBuf,
}

impl LocalStorage {
    /// Create a new local storage instance
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base directory for all storage operations
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    /// Get the full path for a relative path
    fn full_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_path.join(path)
        }
    }
}

/// Implement Storage trait when object-store feature is enabled
#[cfg(feature = "object-store")]
impl Storage for LocalStorage {
    fn read_block(&self, path: &Path, offset: u64, size: u32) -> Result<Vec<u8>> {
        let full_path = self.full_path(path);
        let mut file = File::open(&full_path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0u8; size as usize];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn write_sstable(&self, path: &Path, data: &[u8]) -> Result<()> {
        let full_path = self.full_path(path);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&full_path)?;

        file.write_all(data)?;
        file.sync_all()?;
        Ok(())
    }

    fn delete_sstable(&self, path: &Path) -> Result<()> {
        let full_path = self.full_path(path);
        std::fs::remove_file(full_path)?;
        Ok(())
    }

    fn sync(&self, path: &Path) -> Result<()> {
        let full_path = self.full_path(path);
        let file = OpenOptions::new().write(true).open(full_path)?;
        file.sync_all()?;
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool> {
        let full_path = self.full_path(path);
        Ok(full_path.exists())
    }

    fn list_sstables(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let full_dir = self.full_path(dir);
        let mut sstables = Vec::new();

        if full_dir.exists() && full_dir.is_dir() {
            for entry in std::fs::read_dir(&full_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("sst") {
                    sstables.push(path);
                }
            }
        }

        Ok(sstables)
    }
}

// When object-store feature is disabled: LocalStorage is standalone (no trait)
// This avoids trait overhead and keeps binary size minimal
#[cfg(not(feature = "object-store"))]
impl LocalStorage {
    /// Read a block from an SSTable (direct implementation, no trait)
    pub fn read_block(&self, path: &Path, offset: u64, size: u32) -> Result<Vec<u8>> {
        let full_path = self.full_path(path);
        let mut file = File::open(&full_path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0u8; size as usize];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Write an SSTable (direct implementation, no trait)
    pub fn write_sstable(&self, path: &Path, data: &[u8]) -> Result<()> {
        let full_path = self.full_path(path);

        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&full_path)?;

        file.write_all(data)?;
        file.sync_all()?;
        Ok(())
    }

    /// Delete an SSTable (direct implementation, no trait)
    pub fn delete_sstable(&self, path: &Path) -> Result<()> {
        let full_path = self.full_path(path);
        std::fs::remove_file(full_path)?;
        Ok(())
    }

    /// Fsync an SSTable (direct implementation, no trait)
    pub fn sync(&self, path: &Path) -> Result<()> {
        let full_path = self.full_path(path);
        let file = OpenOptions::new().write(true).open(full_path)?;
        file.sync_all()?;
        Ok(())
    }

    /// Check if an SSTable exists (direct implementation, no trait)
    pub fn exists(&self, path: &Path) -> Result<bool> {
        let full_path = self.full_path(path);
        Ok(full_path.exists())
    }

    /// List all SSTables (direct implementation, no trait)
    pub fn list_sstables(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let full_dir = self.full_path(dir);
        let mut sstables = Vec::new();

        if full_dir.exists() && full_dir.is_dir() {
            for entry in std::fs::read_dir(&full_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("sst") {
                    sstables.push(path);
                }
            }
        }

        Ok(sstables)
    }
}

/// Cloud storage backend using object_store crate
///
/// Supports S3, GCS, Azure Blob Storage, and S3-compatible services (MinIO, R2)
///
/// # Performance Notes
///
/// - Writes buffer entire SSTable in memory before upload (64MB max)
/// - Reads use range requests (ideal for block-based access)
/// - Uses tokio runtime internally (sync wrapper over async)
/// - Includes retry logic with exponential backoff and jitter
#[cfg(feature = "object-store")]
pub struct ObjectStoreBackend {
    store: Arc<dyn object_store::ObjectStore>,
    runtime: tokio::runtime::Handle,
    prefix: String,
    retry_config: RetryConfig,
}

#[cfg(feature = "object-store")]
impl ObjectStoreBackend {
    /// Create a new ObjectStoreBackend with default retry configuration
    ///
    /// # Arguments
    ///
    /// * `store` - The object_store implementation (S3, GCS, Azure, etc.)
    /// * `prefix` - Optional path prefix for all objects (e.g., "seerdb/data/")
    pub fn new(store: Arc<dyn object_store::ObjectStore>, prefix: String) -> Self {
        Self::with_retry_config(store, prefix, RetryConfig::default())
    }

    /// Create a new ObjectStoreBackend with custom retry configuration
    ///
    /// # Arguments
    ///
    /// * `store` - The object_store implementation (S3, GCS, Azure, etc.)
    /// * `prefix` - Optional path prefix for all objects (e.g., "seerdb/data/")
    /// * `retry_config` - Retry behavior configuration
    pub fn with_retry_config(
        store: Arc<dyn object_store::ObjectStore>,
        prefix: String,
        retry_config: RetryConfig,
    ) -> Self {
        Self {
            store,
            runtime: tokio::runtime::Handle::current(),
            prefix,
            retry_config,
        }
    }

    /// Create an S3-compatible backend
    ///
    /// # Arguments
    ///
    /// * `bucket` - S3 bucket name
    /// * `region` - AWS region (e.g., "us-west-2")
    /// * `endpoint` - Optional custom endpoint for MinIO, R2, etc.
    /// * `prefix` - Optional path prefix within bucket
    pub fn s3(bucket: &str, region: &str, endpoint: Option<&str>, prefix: String) -> Result<Self> {
        use object_store::aws::AmazonS3Builder;

        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_region(region);

        if let Some(ep) = endpoint {
            builder = builder.with_endpoint(ep).with_allow_http(true);
        }

        let store = builder
            .build()
            .map_err(|e| crate::db::DBError::ObjectStore(e.to_string()))?;

        Ok(Self {
            store: Arc::new(store),
            runtime: tokio::runtime::Handle::current(),
            prefix,
            retry_config: RetryConfig::default(),
        })
    }

    /// Create a Google Cloud Storage backend
    ///
    /// # Arguments
    ///
    /// * `bucket` - GCS bucket name
    /// * `service_account_path` - Optional path to service account JSON
    /// * `prefix` - Optional path prefix within bucket
    pub fn gcs(bucket: &str, service_account_path: Option<&Path>, prefix: String) -> Result<Self> {
        use object_store::gcp::GoogleCloudStorageBuilder;

        let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(bucket);

        if let Some(path) = service_account_path {
            builder = builder.with_service_account_path(path.to_string_lossy());
        }

        let store = builder
            .build()
            .map_err(|e| crate::db::DBError::ObjectStore(e.to_string()))?;

        Ok(Self {
            store: Arc::new(store),
            runtime: tokio::runtime::Handle::current(),
            prefix,
            retry_config: RetryConfig::default(),
        })
    }

    /// Create an Azure Blob Storage backend
    ///
    /// # Arguments
    ///
    /// * `container` - Azure container name
    /// * `account` - Azure storage account name
    /// * `prefix` - Optional path prefix within container
    pub fn azure(container: &str, account: &str, prefix: String) -> Result<Self> {
        use object_store::azure::MicrosoftAzureBuilder;

        let store = MicrosoftAzureBuilder::new()
            .with_container_name(container)
            .with_account(account)
            .build()
            .map_err(|e| crate::db::DBError::ObjectStore(e.to_string()))?;

        Ok(Self {
            store: Arc::new(store),
            runtime: tokio::runtime::Handle::current(),
            prefix,
            retry_config: RetryConfig::default(),
        })
    }

    /// Convert a local path to object store path
    fn to_object_path(&self, path: &Path) -> object_store::path::Path {
        let path_str = if self.prefix.is_empty() {
            path.to_string_lossy().to_string()
        } else {
            format!("{}/{}", self.prefix.trim_end_matches('/'), path.display())
        };
        object_store::path::Path::from(path_str)
    }

    /// Execute an async operation with retry logic
    ///
    /// Automatically retries on transient errors with exponential backoff + jitter.
    /// Returns immediately on permanent errors (NotFound, auth failures, etc.)
    async fn retry_async<F, T, Fut>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, object_store::Error>>,
    {
        let mut attempt = 0;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Check if we should retry
                    if attempt >= self.retry_config.max_attempts || !is_retryable_error(&e) {
                        return Err(crate::db::DBError::ObjectStore(e.to_string()));
                    }

                    // Sleep with exponential backoff + jitter
                    let delay = self.retry_config.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }
}

#[cfg(feature = "object-store")]
impl Storage for ObjectStoreBackend {
    fn read_block(&self, path: &Path, offset: u64, size: u32) -> Result<Vec<u8>> {
        let object_path = self.to_object_path(path);
        let range = offset as usize..(offset as usize + size as usize);

        self.runtime.block_on(async {
            self.retry_async(|| async {
                let bytes = self.store.get_range(&object_path, range.clone()).await?;
                Ok(bytes.to_vec())
            })
            .await
        })
    }

    fn write_sstable(&self, path: &Path, data: &[u8]) -> Result<()> {
        let object_path = self.to_object_path(path);
        let data = data.to_vec(); // Clone for async closure

        self.runtime.block_on(async {
            self.retry_async(|| async {
                self.store.put(&object_path, data.clone().into()).await?;
                Ok(())
            })
            .await
        })
    }

    fn delete_sstable(&self, path: &Path) -> Result<()> {
        let object_path = self.to_object_path(path);

        self.runtime
            .block_on(async { self.retry_async(|| self.store.delete(&object_path)).await })
    }

    fn sync(&self, _path: &Path) -> Result<()> {
        // Object stores are immediately durable after successful PUT
        // No explicit sync needed
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool> {
        let object_path = self.to_object_path(path);

        self.runtime.block_on(async {
            // Note: We don't retry NotFound errors (they're permanent)
            // but retry_async will handle this correctly via is_retryable_error
            match self
                .retry_async(|| async { self.store.head(&object_path).await })
                .await
            {
                Ok(_) => Ok(true),
                Err(crate::db::DBError::ObjectStore(msg)) if msg.contains("not found") => Ok(false),
                Err(e) => Err(e),
            }
        })
    }

    fn list_sstables(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let prefix = self.to_object_path(dir);

        self.runtime.block_on(async {
            use futures::TryStreamExt;

            // Retry the entire list operation (stream creation + consumption)
            self.retry_async(|| async {
                let mut sstables = Vec::new();
                let mut stream = self.store.list(Some(&prefix));

                while let Some(meta) = stream.try_next().await? {
                    let path_str = meta.location.to_string();
                    if path_str.ends_with(".sst") {
                        sstables.push(PathBuf::from(path_str));
                    }
                }

                Ok(sstables)
            })
            .await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_local_storage_write_read() {
        let dir = tempdir().unwrap();
        let storage = LocalStorage::new(dir.path().to_path_buf());

        // Write data
        let path = Path::new("test.sst");
        let data = b"hello world";
        storage.write_sstable(path, data).unwrap();

        // Read back
        let read_data = storage.read_block(path, 0, data.len() as u32).unwrap();
        assert_eq!(&read_data, data);
    }

    #[test]
    fn test_local_storage_exists() {
        let dir = tempdir().unwrap();
        let storage = LocalStorage::new(dir.path().to_path_buf());

        let path = Path::new("test.sst");
        assert!(!storage.exists(path).unwrap());

        storage.write_sstable(path, b"data").unwrap();
        assert!(storage.exists(path).unwrap());
    }

    #[test]
    fn test_local_storage_delete() {
        let dir = tempdir().unwrap();
        let storage = LocalStorage::new(dir.path().to_path_buf());

        let path = Path::new("test.sst");
        storage.write_sstable(path, b"data").unwrap();
        assert!(storage.exists(path).unwrap());

        storage.delete_sstable(path).unwrap();
        assert!(!storage.exists(path).unwrap());
    }

    #[test]
    fn test_local_storage_list() {
        let dir = tempdir().unwrap();
        let storage = LocalStorage::new(dir.path().to_path_buf());

        // Write multiple SSTables
        storage
            .write_sstable(Path::new("L0_001.sst"), b"data1")
            .unwrap();
        storage
            .write_sstable(Path::new("L0_002.sst"), b"data2")
            .unwrap();
        storage
            .write_sstable(Path::new("L1_001.sst"), b"data3")
            .unwrap();

        // List all
        let sstables = storage.list_sstables(Path::new(".")).unwrap();
        assert_eq!(sstables.len(), 3);
    }

    #[cfg(feature = "object-store")]
    mod object_store_tests {
        use super::*;
        use object_store::memory::InMemory;
        use object_store::ObjectStore as _;

        // Helper to create a runtime and backend for tests
        fn create_test_backend() -> (tokio::runtime::Runtime, ObjectStoreBackend) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let store = Arc::new(InMemory::new());
            let backend = rt.block_on(async {
                ObjectStoreBackend {
                    store,
                    runtime: tokio::runtime::Handle::current(),
                    prefix: String::new(),
                    retry_config: RetryConfig::default(),
                }
            });
            (rt, backend)
        }

        #[test]
        fn test_object_store_backend_write_read() {
            let (rt, backend) = create_test_backend();
            let _guard = rt.enter();

            // Write data
            let path = Path::new("test.sst");
            let data = b"hello world from cloud";
            backend.write_sstable(path, data).unwrap();

            // Read back
            let read_data = backend.read_block(path, 0, data.len() as u32).unwrap();
            assert_eq!(&read_data, data);

            // Read partial (range request)
            let partial = backend.read_block(path, 6, 5).unwrap();
            assert_eq!(&partial, b"world");
        }

        #[test]
        fn test_object_store_backend_exists() {
            let (rt, backend) = create_test_backend();
            let _guard = rt.enter();

            let path = Path::new("test.sst");
            assert!(!backend.exists(path).unwrap());

            backend.write_sstable(path, b"data").unwrap();
            assert!(backend.exists(path).unwrap());
        }

        #[test]
        fn test_object_store_backend_delete() {
            let (rt, backend) = create_test_backend();
            let _guard = rt.enter();

            let path = Path::new("test.sst");
            backend.write_sstable(path, b"data").unwrap();
            assert!(backend.exists(path).unwrap());

            backend.delete_sstable(path).unwrap();
            assert!(!backend.exists(path).unwrap());
        }

        #[test]
        fn test_object_store_backend_list() {
            let (rt, backend) = create_test_backend();
            let _guard = rt.enter();

            // Write multiple SSTables
            backend
                .write_sstable(Path::new("L0_001.sst"), b"data1")
                .unwrap();
            backend
                .write_sstable(Path::new("L0_002.sst"), b"data2")
                .unwrap();
            backend
                .write_sstable(Path::new("L1_001.sst"), b"data3")
                .unwrap();

            // List all
            let sstables = backend.list_sstables(Path::new("")).unwrap();
            assert_eq!(sstables.len(), 3);
        }

        #[test]
        fn test_object_store_backend_with_prefix() {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let store = Arc::new(InMemory::new());
            let backend = rt.block_on(async {
                ObjectStoreBackend {
                    store: store.clone(),
                    runtime: tokio::runtime::Handle::current(),
                    prefix: "seerdb/data".to_string(),
                    retry_config: RetryConfig::default(),
                }
            });
            let _guard = rt.enter();

            // Write with prefix
            let path = Path::new("L0_001.sst");
            backend.write_sstable(path, b"data").unwrap();

            // Verify it's stored with prefix
            let object_path = object_store::path::Path::from("seerdb/data/L0_001.sst");
            let exists = rt.block_on(async { store.head(&object_path).await.is_ok() });
            assert!(exists);

            // Read back should work
            let data = backend.read_block(path, 0, 4).unwrap();
            assert_eq!(&data, b"data");
        }

        #[test]
        fn test_object_store_backend_sync_is_noop() {
            let (rt, backend) = create_test_backend();
            let _guard = rt.enter();

            // sync should be a no-op for object stores
            let path = Path::new("test.sst");
            backend.sync(path).unwrap();
        }
    }
}
