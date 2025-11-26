#!/usr/bin/env python3
"""Debug persistence issue"""

import omendb
import os

# Use fixed path for debugging
db_path = "/tmp/test_persist.db"

# Clean up if exists
if os.path.exists(db_path):
    import shutil
    for f in os.listdir(os.path.dirname(db_path)):
        if f.startswith(os.path.basename(db_path)):
            os.remove(os.path.join(os.path.dirname(db_path), f))

print("Creating database...")
db = omendb.open(db_path, dimensions=4)

print("Inserting vectors...")
vectors = [
    {"id": "v1", "embedding": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    {"id": "v2", "embedding": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
]
db.set(vectors)
print(f"  Inserted {len(vectors)} vectors")

print("Searching before save...")
results = db.search([1.0, 0.0, 0.0, 0.0], k=2)
print(f"  Found {len(results)} results")
for r in results:
    print(f"    - id={r['id']}, distance={r['distance']:.4f}")

print("Saving...")
db.save()
print(f"  Files created:")
for f in sorted(os.listdir(os.path.dirname(db_path))):
    if f.startswith(os.path.basename(db_path)):
        full_path = os.path.join(os.path.dirname(db_path), f)
        size = os.path.getsize(full_path)
        print(f"    - {f} ({size} bytes)")

print("Deleting db object...")
del db

print("\nReloading database...")
db2 = omendb.open(db_path, dimensions=4)
print(f"  Loaded, length = {len(db2)}")

print("Searching after reload...")
results = db2.search([1.0, 0.0, 0.0, 0.0], k=2)
print(f"  Found {len(results)} results")
for r in results:
    print(f"    - id={r['id']}, distance={r['distance']:.4f}")
