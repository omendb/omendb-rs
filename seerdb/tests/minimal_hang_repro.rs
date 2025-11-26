// Minimal reproduction test to isolate DB::open() hang

use seerdb::{DBOptions, DB};
use tempfile::TempDir;

#[test]
fn test_minimal_db_open() {
    eprintln!("TEST START");

    let temp_dir = TempDir::new().unwrap();
    eprintln!("TempDir created");

    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        background_flush: false,
        background_compaction: false,
        ..Default::default()
    };
    eprintln!("Options created");

    eprintln!("Calling DB::open...");
    let _db = DB::open(opts).unwrap();
    eprintln!("DB opened!");

    eprintln!("TEST COMPLETE");
}
