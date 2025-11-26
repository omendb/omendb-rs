use bytes::Bytes;
use seerdb::buffer::{BufferPool, BufferPoolOptions};
use seerdb::sstable::{SSTable, SSTableBuilder};
use tempfile::tempdir;

#[test]
fn test_buffer_pool_integration() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.sst");

    // 1. Create SSTable
    let mut builder = SSTableBuilder::create(&path).unwrap();
    builder
        .add(Bytes::from("key1"), Bytes::from("value1"))
        .unwrap();
    builder
        .add(Bytes::from("key2"), Bytes::from("value2"))
        .unwrap();
    builder
        .add(Bytes::from("key3"), Bytes::from("value3"))
        .unwrap();
    builder.finish().unwrap();

    // 2. Open with BufferPool
    let pool_options = BufferPoolOptions {
        capacity_bytes: 1024 * 1024, // 1MB
        frame_size: 4096,            // 4KB
        num_shards: 4,               // 4 shards
    };
    let buffer_pool = BufferPool::new(pool_options);

    let mut sst = SSTable::open_with_buffer_pool(&path, Some(buffer_pool.clone())).unwrap();

    // 3. Read data
    // First read should be a miss (load from disk into pool)
    assert_eq!(sst.get(b"key1").unwrap().unwrap(), Bytes::from("value1"));

    // Second read should be a hit (if we re-read same block)
    // Note: SSTable also has block_cache, so we might hit that.
    // But BufferPool is checked on miss.

    // Let's verify correctness first
    assert_eq!(sst.get(b"key2").unwrap().unwrap(), Bytes::from("value2"));
    assert_eq!(sst.get(b"key3").unwrap().unwrap(), Bytes::from("value3"));

    // 4. Verify BufferPool usage
    // We can't easily inspect internal stats of BufferPool from here without public API
    // But if this passes, at least the new get_page API works correctly.
}

#[test]
fn test_buffer_pool_large_values() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.sst");

    // 1. Create SSTable with large value (larger than frame size)
    let mut builder = SSTableBuilder::create(&path).unwrap();
    let large_val = vec![b'x'; 10000]; // 10KB > 4KB frame
    builder
        .add(Bytes::from("large"), Bytes::from(large_val.clone()))
        .unwrap();
    builder.finish().unwrap();

    // 2. Open with BufferPool (Small frames)
    let pool_options = BufferPoolOptions {
        capacity_bytes: 1024 * 1024,
        frame_size: 4096, // 4KB frames
        num_shards: 4,    // 4 shards
    };
    let buffer_pool = BufferPool::new(pool_options);

    let mut sst = SSTable::open_with_buffer_pool(&path, Some(buffer_pool)).unwrap();

    // 3. Read data
    // This should trigger the resize logic in get_page loader
    let val = sst.get(b"large").unwrap().unwrap();
    assert_eq!(val.len(), 10000);
    assert_eq!(val.as_ref(), &large_val);
}
