use bytes::Bytes;
use seerdb::{DBOptions, MergeOperator, StringAppendOperator, DB};
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_merge_resolution_in_range_scan() {
    let dir = tempdir().unwrap();
    let mut opts = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    // Configure Merge Operator (append with comma)
    opts.merge_operator = Some(Arc::new(StringAppendOperator::new(',')));

    let db = DB::open(opts).unwrap();

    // 1. Put base value
    db.put(b"key1", b"val").unwrap();

    // 2. Merge operand
    db.merge(b"key1", b"op1").unwrap();

    // 3. Range Scan
    // Should return "val,op1"
    // Currently fails because RangeIterator treats Merge as Tombstone (None)
    let mut iter = db.range(b"key1", Some(b"key2")).unwrap();

    match iter.next() {
        Some(Ok((key, value))) => {
            assert_eq!(key, Bytes::from("key1"));
            assert_eq!(
                value,
                Bytes::from("val,op1"),
                "Range scan failed to resolve merge"
            );
        }
        Some(Err(e)) => panic!("Iterator error: {}", e),
        None => panic!("Expected one item, found none (Merge treated as Tombstone?)"),
    }
}

#[test]
fn test_merge_stacking_in_range_scan() {
    let dir = tempdir().unwrap();
    let mut opts = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    opts.merge_operator = Some(Arc::new(StringAppendOperator::new(',')));

    let db = DB::open(opts).unwrap();

    // 1. Merge (no base)
    db.merge(b"list", b"item1").unwrap();

    // 2. Merge again
    db.merge(b"list", b"item2").unwrap();

    // 3. Range Scan
    let mut iter = db.prefix(b"list").unwrap();

    match iter.next() {
        Some(Ok((key, value))) => {
            assert_eq!(key, Bytes::from("list"));
            assert_eq!(
                value,
                Bytes::from("item1,item2"),
                "Range scan failed to resolve stacked merges"
            );
        }
        _ => panic!("Expected item 'list' with value 'item1,item2'"),
    }
}
