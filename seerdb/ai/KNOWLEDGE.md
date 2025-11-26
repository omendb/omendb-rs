# KNOWLEDGE - seerdb

Permanent codebase quirks and gotchas discovered during development.

---

## InternalKey Encoding

| Area | Knowledge | Impact | Discovered |
|------|-----------|--------|------------|
| SSTable lookup | InternalKey = `[user_key][8-byte-inverted-trailer]` | Prefix keys sort counterintuitively | Nov 2025 |

**Detail**: When one user_key is a prefix of another (e.g., "key1" vs "key10"):
- "key10" encodes as `[k,e,y,1,0,trailer...]`
- "key1" encodes as `[k,e,y,1,trailer...]`
- Since `'0'` (0x30) < `0xFF` (first trailer byte), **key10 < key1** in encoded order
- But in user_key order: **"key1" < "key10"** (shorter string is smaller)

**Fix**: Use `Block::find_mvcc()` which scans forward from binary search position to find matching user_key. Never use simple binary search + user_key verification for MVCC lookups.

**Commit**: `092f3ec` - `fix(sstable): correct MVCC lookup for prefix keys`

---

## Trailer Format

| Area | Knowledge | Impact | Discovered |
|------|-----------|--------|------------|
| InternalKey | Trailer = `!((seq << 8) \| type)` (inverted) | Higher seq numbers sort first | Nov 2025 |

**Why inverted**: For MVCC, we want the newest version (highest seq) to sort first for any given user_key. Inverting the trailer achieves this with standard lexicographic comparison.
