#![no_main]

use libfuzzer_sys::fuzz_target;
use seerdb::vlog::ValuePointer;
use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    // Skip inputs too small to construct a ValuePointer
    if data.len() < 12 {
        return;
    }

    // Write fuzzed data to a temporary vLog file
    let mut temp_file = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };

    if temp_file.write_all(data).is_err() {
        return;
    }

    let path = temp_file.path();

    // Try to open and read the fuzzed vLog
    // We expect this to fail gracefully with an error, not panic
    if let Ok(mut vlog) = seerdb::vlog::VLog::open(path) {
        // Construct a ValuePointer from fuzzed data
        let offset = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        let length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

        let pointer = ValuePointer { offset, length };

        // Try to read - should handle invalid pointers gracefully
        let _ = vlog.read(pointer);
    }

    // File is automatically cleaned up when temp_file is dropped
});
