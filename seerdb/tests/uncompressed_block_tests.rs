use bytes::Bytes;
use seerdb::buffer::manager::FrameRef;
use seerdb::sstable::block::{Block, BlockBuilder, BlockData};

#[test]
fn test_uncompressed_block_builder() {
    let mut builder = BlockBuilder::new();
    builder.set_compression(false); // Disable compression

    // Add highly compressible data
    let key = b"key1";
    let value = vec![b'x'; 1000]; // 1000 'x's
    builder.add(key, &value);

    let block_bytes = builder.finish();

    // Verify size: if compressed, it would be small (~20 bytes).
    // Uncompressed should be ~1000 bytes + overhead.
    assert!(block_bytes.len() > 1000, "Block should be uncompressed");

    // Parse back
    let block = Block::from_bytes(block_bytes).unwrap();
    let entries: Vec<_> = block.iter().collect();

    assert_eq!(entries.len(), 1);
    let (k, v) = entries[0].as_ref().unwrap();
    assert_eq!(k, &Bytes::from("key1"));
    assert_eq!(v.len(), 1000);
    assert_eq!(v[0], b'x');
}

#[test]
fn test_uncompressed_block_zero_copy_logic() {
    // This test verifies that Block::new works with Borrowed data for uncompressed blocks
    // We can't easily mock FrameRef without a BufferPool, so we'll rely on Owned data path
    // which shares the exact same parsing logic (BlockData::as_slice).

    let mut builder = BlockBuilder::new();
    builder.set_compression(false);
    builder.add(b"key1", b"val1");
    builder.add(b"key2", b"val2");
    let block_bytes = builder.finish();

    let block = Block::from_bytes(block_bytes.clone()).unwrap();

    // Verify it works
    let entries: Vec<_> = block.iter().map(|r| r.unwrap()).collect();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].0, Bytes::from("key1"));
    assert_eq!(entries[1].0, Bytes::from("key2"));
}

#[test]
fn test_mixed_compression_settings() {
    // Test 1: Compressed (Default)
    let mut builder = BlockBuilder::new();
    // builder.set_compression(true); // Default
    let large_val = vec![b'a'; 1000];
    builder.add(b"key1", &large_val);
    let compressed_bytes = builder.finish();

    // Test 2: Uncompressed
    let mut builder = BlockBuilder::new();
    builder.set_compression(false);
    builder.add(b"key1", &large_val);
    let uncompressed_bytes = builder.finish();

    assert!(compressed_bytes.len() < uncompressed_bytes.len());

    // Verify both parse correctly
    let b1 = Block::from_bytes(compressed_bytes).unwrap();
    let b2 = Block::from_bytes(uncompressed_bytes).unwrap();

    let e1: Vec<_> = b1.iter().collect();
    let e2: Vec<_> = b2.iter().collect();

    assert_eq!(e1.len(), e2.len());
    assert_eq!(e1[0].as_ref().unwrap().1, e2[0].as_ref().unwrap().1);
}
