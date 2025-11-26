#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    // Skip empty inputs
    if data.is_empty() {
        return;
    }

    // Write fuzzed data to a temporary WAL file
    let mut temp_file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };

    if temp_file.write_all(data).is_err() {
        return;
    }

    let path = temp_file.path();

    // Try to open and read the fuzzed WAL
    // We expect this to fail gracefully with an error, not panic
    if let Ok(mut reader) = seerdb::wal::WALReader::open(path) {
        // Try to read all records
        // Should handle corruption/truncation gracefully
        while let Ok(Some(_record)) = reader.read_next() {
            // Continue reading until error or end
        }
    }

    // File is automatically cleaned up when temp_file is dropped
});
