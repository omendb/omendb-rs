#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    // Skip empty inputs
    if data.is_empty() {
        return;
    }

    // Write fuzzed data to a temporary file
    let mut temp_file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };

    if temp_file.write_all(data).is_err() {
        return;
    }

    let path = temp_file.path().to_path_buf();

    // Try to open the fuzzed SSTable
    // We expect this to fail gracefully with an error, not panic
    let _ = seerdb::sstable::SSTable::open(&path);

    // File is automatically cleaned up when temp_file is dropped
});
