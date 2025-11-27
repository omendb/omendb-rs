#!/usr/bin/env python3
"""Test ID mapping after delete and reload"""

import omendb
import os
import tempfile
import json

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, "test.db")

    # Create and populate
    print("Creating database...")
    db = omendb.open(db_path, dimensions=4)

    vectors = [
        {"id": "v1", "embedding": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "v2", "embedding": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
        {"id": "v3", "embedding": [0.0, 0.0, 1.0, 0.0], "metadata": {}},
    ]
    db.set(vectors)
    print(f"Inserted {len(vectors)} vectors")

    # Delete v2
    print("Deleting v2...")
    db.delete(["v2"])

    # Save
    print("Saving...")
    db.save()

    # Check what was saved
    directory = os.path.dirname(db_path)
    filename = os.path.basename(db_path)
    id_mapping_file = os.path.join(directory, f"{filename}.id_mapping.json")

    print("\nid_mapping.json contents:")
    with open(id_mapping_file) as f:
        id_mapping = json.load(f)
        print(json.dumps(id_mapping, indent=2))

    deleted_file = os.path.join(directory, f"{filename}.deleted.json")
    if os.path.exists(deleted_file):
        print("\ndeleted.json contents:")
        with open(deleted_file) as f:
            deleted = json.load(f)
            print(json.dumps(deleted, indent=2))

    del db

    # Reload
    print("\nReloading...")
    db2 = omendb.open(db_path, dimensions=4)
    print(f"Loaded {len(db2)} vectors")

    # Search
    print("\nSearching...")
    results = db2.search([0.5, 0.5, 0.5, 0.5], k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - id={r['id']}, distance={r['distance']:.4f}")
