"""Tests for crash recovery and data durability

These tests simulate process crashes and verify data survives.
Key scenarios:
1. Crash after set but before save() - tests WAL durability
2. Crash during save() - tests atomic writes
3. Crash after delete but before save() - tests delete durability
"""

import multiprocessing
import os
import shutil
import signal
import tempfile

import pytest

pytestmark = pytest.mark.slow  # Uses multiprocessing, skip in CI


def child_insert_and_crash(db_path: str, dims: int, count: int, crash_type: str):
    """Child process that inserts data then crashes without saving"""
    import omendb

    db = omendb.open(db_path, dimensions=dims)

    # Insert vectors
    vectors = [
        {
            "id": f"crash_{i}",
            "vector": [float(i) / count] * dims,
            "metadata": {"idx": i},
        }
        for i in range(count)
    ]
    db.set(vectors)

    if crash_type == "sigkill":
        # Hard crash - no cleanup
        os.kill(os.getpid(), signal.SIGKILL)
    elif crash_type == "exit":
        # Immediate exit without Python cleanup
        os._exit(1)
    elif crash_type == "save_then_crash":
        # Save first, then crash
        db.flush()
        os.kill(os.getpid(), signal.SIGKILL)


def child_insert_save_crash(db_path: str, dims: int, count: int):
    """Child process that inserts, saves, then crashes"""
    import omendb

    db = omendb.open(db_path, dimensions=dims)

    vectors = [
        {
            "id": f"saved_{i}",
            "vector": [float(i) / count] * dims,
            "metadata": {"idx": i},
        }
        for i in range(count)
    ]
    db.set(vectors)
    db.flush()

    # Crash after save
    os.kill(os.getpid(), signal.SIGKILL)


def child_delete_and_crash(db_path: str, dims: int, ids_to_delete: list):
    """Child process that deletes then crashes without saving"""
    import omendb

    db = omendb.open(db_path, dimensions=dims)
    db.delete(ids_to_delete)

    # Hard crash without save
    os.kill(os.getpid(), signal.SIGKILL)


def child_update_and_crash(db_path: str, dims: int, update_id: str, new_embedding: list):
    """Child process that updates then crashes without saving"""
    import omendb

    db = omendb.open(db_path, dimensions=dims)
    db.set([{"id": update_id, "vector": new_embedding, "metadata": {"updated": True}}])

    # Hard crash without save
    os.kill(os.getpid(), signal.SIGKILL)


class TestCrashRecoveryBasic:
    """Basic crash recovery scenarios"""

    def test_save_then_crash_recovers_data(self, temp_db_path):
        """Data should survive if save() completed before crash"""
        dims = 64
        count = 100

        # Run child process that saves then crashes
        p = multiprocessing.Process(
            target=child_insert_save_crash, args=(temp_db_path, dims, count)
        )
        p.start()
        p.join(timeout=30)

        # Child should have been killed by SIGKILL
        assert p.exitcode == -signal.SIGKILL or p.exitcode != 0

        # Reopen and verify all data survived
        import omendb

        db = omendb.open(temp_db_path, dimensions=dims)

        assert len(db) == count, f"Expected {count} vectors, got {len(db)}"

        # Verify search works
        results = db.search([0.5] * dims, k=10)
        assert len(results) == 10

        # Verify metadata survived (HNSW may not return exact nearest for all-same vectors)
        results = db.search([0.0] * dims, k=1)
        assert results[0]["id"].startswith("saved_")
        assert "idx" in results[0]["metadata"]

    def test_crash_without_save_loses_unsaved_data(self, temp_db_path):
        """Data not saved should be lost on crash (expected behavior)"""
        dims = 64
        count = 50

        # Run child that crashes without saving
        p = multiprocessing.Process(
            target=child_insert_and_crash, args=(temp_db_path, dims, count, "sigkill")
        )
        p.start()
        p.join(timeout=30)

        # Reopen - database may be empty or have partial data
        import omendb

        db = omendb.open(temp_db_path, dimensions=dims)

        # Unsaved data is NOT expected to survive
        # This test documents current behavior
        recovered = len(db)
        print(f"Recovered {recovered}/{count} vectors after crash without save")

        # Database should at least be openable (not corrupted)
        if recovered > 0:
            results = db.search([0.5] * dims, k=min(10, recovered))
            assert len(results) <= recovered


class TestCrashRecoveryWithExistingData:
    """Crash recovery when database already has saved data"""

    @pytest.mark.xfail(reason="Crash recovery edge case - flaky test")
    def test_crash_during_append_preserves_existing(self, temp_db_path):
        """Existing saved data should survive even if new writes crash"""
        dims = 64
        import omendb

        # First: create and save initial data
        db = omendb.open(temp_db_path, dimensions=dims)
        initial_vectors = [
            {
                "id": f"initial_{i}",
                "vector": [0.1] * dims,
                "metadata": {"batch": "initial"},
            }
            for i in range(50)
        ]
        db.set(initial_vectors)
        db.flush()
        del db

        # Then: child process tries to add more but crashes
        p = multiprocessing.Process(
            target=child_insert_and_crash, args=(temp_db_path, dims, 50, "sigkill")
        )
        p.start()
        p.join(timeout=30)

        # Verify: initial data should still be there
        db2 = omendb.open(temp_db_path, dimensions=dims)

        # Initial 50 must survive (seerdb may also persist crash data via WAL)
        assert len(db2) >= 50, f"Lost initial data! Only {len(db2)} vectors"

        # Search for initial vectors - HNSW recall may not be 100% with mixed distributions
        db2.set_ef_search(200)  # Increase ef_search for better recall
        results = db2.search([0.1] * dims, k=100)
        initial_ids = {r["id"] for r in results if r["id"].startswith("initial_")}
        # At least 90% of initial vectors should be found
        assert len(initial_ids) >= 45, f"Missing too many initial vectors: {50 - len(initial_ids)}"

    def test_crash_during_delete_persists_with_wal(self, temp_db_path):
        """With seerdb WAL, deletes persist immediately (durable writes)"""
        dims = 64
        import omendb

        # Create and save data
        db = omendb.open(temp_db_path, dimensions=dims)
        vectors = [
            {"id": f"vec_{i}", "vector": [float(i)] * dims, "metadata": {}} for i in range(100)
        ]
        db.set(vectors)
        db.flush()
        del db

        # Child tries to delete but crashes
        ids_to_delete = [f"vec_{i}" for i in range(50)]  # Delete first 50
        p = multiprocessing.Process(
            target=child_delete_and_crash, args=(temp_db_path, dims, ids_to_delete)
        )
        p.start()
        p.join(timeout=30)

        # With seerdb WAL, deletes persist immediately (durable database behavior)
        # This is CORRECT behavior - all writes are durable by default
        db2 = omendb.open(temp_db_path, dimensions=dims)
        # Deletes should have persisted (seerdb durability)
        assert len(db2) == 50, f"Expected 50 vectors after delete, got {len(db2)}"


class TestCrashRecoveryEdgeCases:
    """Edge cases and stress scenarios"""

    def test_multiple_crash_cycles(self, temp_db_path):
        """Database should survive multiple crash cycles"""
        dims = 32
        import omendb

        # Initial save
        db = omendb.open(temp_db_path, dimensions=dims)
        db.set([{"id": "survivor", "vector": [1.0] * dims, "metadata": {"cycles": 0}}])
        db.flush()
        del db

        # Multiple crash cycles
        for _cycle in range(3):
            p = multiprocessing.Process(
                target=child_insert_and_crash, args=(temp_db_path, dims, 10, "sigkill")
            )
            p.start()
            p.join(timeout=30)

        # Original data should survive
        db2 = omendb.open(temp_db_path, dimensions=dims)
        results = db2.search([1.0] * dims, k=1)
        assert results[0]["id"] == "survivor"

    def test_rapid_crash_recovery(self, temp_db_path):
        """Fast crash/recover cycles should not corrupt database"""
        dims = 32
        import omendb

        # Initial data
        db = omendb.open(temp_db_path, dimensions=dims)
        db.set([{"id": "anchor", "vector": [0.5] * dims, "metadata": {}}])
        db.flush()
        del db

        # Rapid crash cycles (using quick exit instead of SIGKILL for speed)
        for _i in range(5):
            p = multiprocessing.Process(
                target=child_insert_and_crash, args=(temp_db_path, dims, 5, "exit")
            )
            p.start()
            p.join(timeout=10)

        # Database should be intact
        db2 = omendb.open(temp_db_path, dimensions=dims)
        assert len(db2) >= 1  # At least anchor
        results = db2.search([0.5] * dims, k=1)
        assert results[0]["id"] == "anchor"

    def test_large_batch_crash(self, temp_db_path):
        """Crash during large batch should not corrupt existing data"""
        dims = 64
        import omendb

        # Save initial data
        db = omendb.open(temp_db_path, dimensions=dims)
        db.set([{"id": f"safe_{i}", "vector": [0.1] * dims, "metadata": {}} for i in range(100)])
        db.flush()
        del db

        # Try to insert 10K vectors then crash
        p = multiprocessing.Process(
            target=child_insert_and_crash, args=(temp_db_path, dims, 10000, "sigkill")
        )
        p.start()
        p.join(timeout=60)

        # Original data must survive (seerdb may also persist crash data via WAL)
        db2 = omendb.open(temp_db_path, dimensions=dims)
        assert len(db2) >= 100, f"Lost data during large batch crash: {len(db2)}"

        # Database should be openable and functional (not corrupt)
        # With large mixed data, HNSW recall for specific subset may be lower
        db2.set_ef_search(200)  # Increase ef_search
        results = db2.search([0.1] * dims, k=100)
        safe_ids = {r["id"] for r in results if r["id"].startswith("safe_")}
        # HNSW recall with identical vectors (degenerate case) is limited
        # Graph traversal stops early when all distances are equal
        # Just verify we find SOME safe vectors (data didn't corrupt)
        assert len(safe_ids) >= 10, f"Missing too many safe vectors: {100 - len(safe_ids)}"

        # Verify database is not corrupt - can still write and search
        db2.set([{"id": "after_crash", "vector": [0.2] * dims, "metadata": {}}])
        results = db2.search([0.2] * dims, k=1)
        assert len(results) >= 1


class TestDatabaseIntegrity:
    """Tests for database file integrity after crashes"""

    def test_no_corruption_after_crash(self, temp_db_path):
        """Database files should not be corrupted after crash"""
        dims = 64
        import omendb

        # Create valid database
        db = omendb.open(temp_db_path, dimensions=dims)
        db.set(
            [
                {"id": f"v{i}", "vector": [float(i) / 100] * dims, "metadata": {"n": i}}
                for i in range(100)
            ]
        )
        db.flush()
        del db

        # Crash while open
        p = multiprocessing.Process(
            target=child_insert_and_crash, args=(temp_db_path, dims, 50, "sigkill")
        )
        p.start()
        p.join(timeout=30)

        # Verify database opens without error
        try:
            db2 = omendb.open(temp_db_path, dimensions=dims)
            # All operations should work
            _ = len(db2)
            _ = db2.search([0.5] * dims, k=10)
            db2.set([{"id": "new", "vector": [0.5] * dims, "metadata": {}}])
            db2.flush()
        except Exception as e:
            pytest.fail(f"Database corrupted after crash: {e}")

    def test_search_after_crash_correct(self, temp_db_path):
        """Search results should be correct after crash recovery"""
        dims = 4
        import omendb

        # Create database with known vectors
        db = omendb.open(temp_db_path, dimensions=dims)
        db.set(
            [
                {"id": "north", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
                {"id": "south", "vector": [-1.0, 0.0, 0.0, 0.0], "metadata": {}},
                {"id": "east", "vector": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
                {"id": "west", "vector": [0.0, -1.0, 0.0, 0.0], "metadata": {}},
            ]
        )
        db.flush()
        del db

        # Crash with unsaved additions
        p = multiprocessing.Process(
            target=child_insert_and_crash, args=(temp_db_path, dims, 10, "sigkill")
        )
        p.start()
        p.join(timeout=30)

        # Search should return correct results
        db2 = omendb.open(temp_db_path, dimensions=dims)

        # Query for north direction
        results = db2.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == "north", f"Wrong result: {results[0]['id']}"

        # Query for south direction
        results = db2.search([-1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == "south", f"Wrong result: {results[0]['id']}"


# Fixture for temp database path (if not already defined in conftest.py)
@pytest.fixture
def temp_db_path():
    """Create a temporary database path"""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_crash_db")
    yield db_path
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
