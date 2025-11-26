use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecordError {
    #[error("Buffer too short")]
    BufferTooShort,
    #[error("Invalid record type: {0}")]
    InvalidType(u8),
    #[error("Varint decoding error")]
    VarintError,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchOp {
    Put { key: Bytes, value: Bytes },
    Delete { key: Bytes },
    Merge { key: Bytes, operand: Bytes },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Put {
        key: Bytes,
        value: Bytes,
        seq: u64,
    },
    Delete {
        key: Bytes,
        seq: u64,
    },
    Merge {
        key: Bytes,
        operand: Bytes,
        seq: u64,
    },
    Batch {
        base_seq: u64,
        operations: Vec<BatchOp>,
    },
}

impl Record {
    /// Compute encoded length without allocating (avoids double-encode)
    #[inline]
    pub fn encoded_len(&self) -> usize {
        match self {
            Record::Put { key, value, .. } => {
                // type(1) + seq(8) + key_len(4) + key + value_len(4) + value
                1 + 8 + 4 + key.len() + 4 + value.len()
            }
            Record::Delete { key, .. } => {
                // type(1) + seq(8) + key_len(4) + key
                1 + 8 + 4 + key.len()
            }
            Record::Merge { key, operand, .. } => {
                // type(1) + seq(8) + key_len(4) + key + operand_len(4) + operand
                1 + 8 + 4 + key.len() + 4 + operand.len()
            }
            Record::Batch { operations, .. } => {
                // type(1) + base_seq(8) + count(4) + ops
                let ops_len: usize = operations
                    .iter()
                    .map(|op| match op {
                        BatchOp::Put { key, value } => 1 + 4 + key.len() + 4 + value.len(),
                        BatchOp::Delete { key } => 1 + 4 + key.len(),
                        BatchOp::Merge { key, operand } => 1 + 4 + key.len() + 4 + operand.len(),
                    })
                    .sum();
                1 + 8 + 4 + ops_len
            }
        }
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        match self {
            Record::Put { key, value, seq } => {
                buf.put_u8(1); // Type 1 = Put
                buf.put_u64_le(*seq); // Sequence number
                put_bytes(&mut buf, key);
                put_bytes(&mut buf, value);
            }
            Record::Delete { key, seq } => {
                buf.put_u8(2); // Type 2 = Delete
                buf.put_u64_le(*seq); // Sequence number
                put_bytes(&mut buf, key);
            }
            Record::Batch {
                base_seq,
                operations,
            } => {
                buf.put_u8(3); // Type 3 = Batch
                buf.put_u64_le(*base_seq); // Base sequence number
                buf.put_u32_le(operations.len() as u32);
                for op in operations {
                    match op {
                        BatchOp::Put { key, value } => {
                            buf.put_u8(1);
                            put_bytes(&mut buf, key);
                            put_bytes(&mut buf, value);
                        }
                        BatchOp::Delete { key } => {
                            buf.put_u8(2);
                            put_bytes(&mut buf, key);
                        }
                        BatchOp::Merge { key, operand } => {
                            buf.put_u8(4);
                            put_bytes(&mut buf, key);
                            put_bytes(&mut buf, operand);
                        }
                    }
                }
            }
            Record::Merge { key, operand, seq } => {
                buf.put_u8(4); // Type 4 = Merge
                buf.put_u64_le(*seq); // Sequence number
                put_bytes(&mut buf, key);
                put_bytes(&mut buf, operand);
            }
        }

        buf.freeze()
    }

    pub fn decode(mut buf: Bytes) -> Result<Self, RecordError> {
        if buf.remaining() < 1 {
            return Err(RecordError::BufferTooShort);
        }

        let record_type = buf.get_u8();

        match record_type {
            1 => {
                // Put
                if buf.remaining() < 8 {
                    return Err(RecordError::BufferTooShort);
                }
                let seq = buf.get_u64_le();
                let key = get_bytes(&mut buf)?;
                let value = get_bytes(&mut buf)?;
                Ok(Record::Put { key, value, seq })
            }
            2 => {
                // Delete
                if buf.remaining() < 8 {
                    return Err(RecordError::BufferTooShort);
                }
                let seq = buf.get_u64_le();
                let key = get_bytes(&mut buf)?;
                Ok(Record::Delete { key, seq })
            }
            3 => {
                // Batch
                if buf.remaining() < 8 {
                    return Err(RecordError::BufferTooShort);
                }
                let base_seq = buf.get_u64_le();

                if buf.remaining() < 4 {
                    return Err(RecordError::BufferTooShort);
                }
                let count = buf.get_u32_le();
                let mut operations = Vec::with_capacity(count as usize);

                for _ in 0..count {
                    if buf.remaining() < 1 {
                        return Err(RecordError::BufferTooShort);
                    }
                    let op_type = buf.get_u8();
                    match op_type {
                        1 => {
                            let key = get_bytes(&mut buf)?;
                            let value = get_bytes(&mut buf)?;
                            operations.push(BatchOp::Put { key, value });
                        }
                        2 => {
                            let key = get_bytes(&mut buf)?;
                            operations.push(BatchOp::Delete { key });
                        }
                        4 => {
                            let key = get_bytes(&mut buf)?;
                            let operand = get_bytes(&mut buf)?;
                            operations.push(BatchOp::Merge { key, operand });
                        }
                        _ => return Err(RecordError::InvalidType(op_type)),
                    }
                }
                Ok(Record::Batch {
                    base_seq,
                    operations,
                })
            }
            4 => {
                // Merge
                if buf.remaining() < 8 {
                    return Err(RecordError::BufferTooShort);
                }
                let seq = buf.get_u64_le();
                let key = get_bytes(&mut buf)?;
                let operand = get_bytes(&mut buf)?;
                Ok(Record::Merge { key, operand, seq })
            }
            _ => Err(RecordError::InvalidType(record_type)),
        }
    }
}

fn put_bytes(buf: &mut BytesMut, bytes: &Bytes) {
    buf.put_u32_le(bytes.len() as u32);
    buf.put(bytes.as_ref());
}

fn get_bytes(buf: &mut Bytes) -> Result<Bytes, RecordError> {
    if buf.remaining() < 4 {
        return Err(RecordError::BufferTooShort);
    }
    let len = buf.get_u32_le() as usize;
    if buf.remaining() < len {
        return Err(RecordError::BufferTooShort);
    }
    Ok(buf.split_to(len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_encoding() {
        let record = Record::Put {
            key: Bytes::from("key"),
            value: Bytes::from("value"),
            seq: 42,
        };

        let encoded = record.encode();
        let decoded = Record::decode(encoded).unwrap();

        assert_eq!(record, decoded);
    }

    #[test]
    fn test_batch_encoding() {
        let operations = vec![
            BatchOp::Put {
                key: Bytes::from("k1"),
                value: Bytes::from("v1"),
            },
            BatchOp::Delete {
                key: Bytes::from("k2"),
            },
        ];

        let record = Record::Batch {
            base_seq: 100,
            operations,
        };

        let encoded = record.encode();
        let decoded = Record::decode(encoded).unwrap();

        assert_eq!(record, decoded);
    }
}
