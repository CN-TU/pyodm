"""Tests for the exact kNN indexes (M-Tree, KD-Tree, brute force)."""

from __future__ import annotations

import numpy as np
import pytest

from pyodm.trees import BruteForceIndex, KDTreeIndex, MTree, make_index

METRICS = ["euclidean", "manhattan", "chebyshev"]
CORES = ["m-tree", "kd-tree", "brute"]


def build(core, points, metric="euclidean"):
    index = make_index(core, dim=points.shape[1], metric=metric)
    for row, point in enumerate(points):
        index.insert(point, row)
    return index


def assert_same_query(reference, index, queries, k):
    for query in queries:
        expected_ids, expected_dists = reference.query(query, k)
        ids, dists = index.query(query, k)
        assert np.array_equal(ids, expected_ids), (ids, expected_ids)
        np.testing.assert_allclose(dists, expected_dists, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("metric", METRICS)
def test_matches_brute_force_after_updates_and_removals(core, metric):
    rng = np.random.RandomState(0)
    points = rng.normal(size=(250, 3))
    queries = rng.normal(size=(40, 3))
    k = 4

    reference = build("brute", points, metric)
    index = build(core, points, metric)
    assert_same_query(reference, index, queries, k)

    # Move a subset of the objects (ODM moves observers all the time).
    for obj_id in rng.choice(len(points), size=120, replace=False):
        new_point = rng.normal(size=3)
        reference.update(int(obj_id), new_point)
        index.update(int(obj_id), new_point)
    assert_same_query(reference, index, queries, k)

    # Remove a subset of the objects.
    for obj_id in rng.choice(len(points), size=80, replace=False):
        reference.remove(int(obj_id))
        index.remove(int(obj_id))
    assert_same_query(reference, index, queries, k)


def test_insert_move_order_does_not_change_observer_state():
    """The M-Tree must store its own copy of every inserted point.

    ``base.check_point`` used to return a *view*; ODM updates its observer
    array in place, so the tree silently kept stale/aliased coordinates.
    """
    rng = np.random.RandomState(1)
    points = rng.normal(size=(20, 2))
    tree = MTree(dim=2)
    for row, point in enumerate(points):
        tree.insert(point, row)
    pointers = [tree._points[row] for row in range(len(points))]
    for pointer, original in zip(pointers, points):
        np.testing.assert_allclose(pointer, original)
        assert not np.shares_memory(pointer, original)


@pytest.mark.parametrize("core", CORES)
def test_index_maintenance_operations(core):
    rng = np.random.RandomState(2)
    index = make_index(core, dim=2)
    reference = BruteForceIndex(dim=2)
    points = rng.normal(size=(30, 2))
    for row, point in enumerate(points):
        index.insert(point, row)
        reference.insert(point, row)
    assert len(index) == 30

    index.remove(5)
    reference.remove(5)
    assert 5 not in index
    with pytest.raises(KeyError):
        index.remove(5)
    with pytest.raises(KeyError):
        index.insert(points[0], 6)  # duplicate id (6 is still alive)

    # Update keeps the object alive and the results consistent.
    index.update(7, np.array([100.0, 100.0]))
    reference.update(7, np.array([100.0, 100.0]))
    assert 7 in index
    assert np.array_equal(index.query(np.array([0.0, 0.0]), 1)[0], reference.query(np.array([0.0, 0.0]), 1)[0])

    # k larger than the number of objects is clamped.
    ids, dists = index.query(np.array([0.0, 0.0]), len(index) + 10)
    assert ids.size == len(index)


def test_kdtree_rebuilds_and_stays_exact():
    rng = np.random.RandomState(3)
    index = KDTreeIndex(dim=2, rebuild_ratio=0.05)
    reference = BruteForceIndex(dim=2)
    points = rng.normal(size=(200, 2))
    ids = np.arange(points.shape[0])
    index.bulk_load(points, ids)
    for point, obj_id in zip(points, ids):
        reference.insert(point, int(obj_id))

    for row in range(100):
        obj_id = int(rng.randint(0, 200))
        new_point = rng.normal(size=2)
        index.update(obj_id, new_point)
        reference.update(obj_id, new_point)
    stats = index.stats()
    assert stats["n_rebuilds"] >= 1
    assert stats["n_pending"] <= len(index)

    for _ in range(30):
        query = rng.normal(size=2)
        assert np.array_equal(index.query(query, 3)[0], reference.query(query, 3)[0])


@pytest.mark.parametrize("core", ["m-tree", "kd-tree"])
def test_query_many_agrees_with_query(core):
    rng = np.random.RandomState(4)
    points = rng.normal(size=(150, 2))
    index = build(core, points)
    queries = rng.normal(size=(20, 2))
    k = 3
    bulk_ids, bulk_dists = index.query_many(queries, k=k)
    for row, query in enumerate(queries):
        ids, dists = index.query(query, k)
        assert np.array_equal(bulk_ids[row], ids)
        np.testing.assert_allclose(bulk_dists[row], dists)


def test_empty_and_invalid_queries():
    index = MTree(dim=2)
    with pytest.raises(ValueError):
        index.query(np.zeros(2))
    with pytest.raises(ValueError):
        MTree(dim=2).insert(np.zeros(0), 0)
    with pytest.raises(ValueError):
        MTree(dim=2).insert(np.array([1.0, np.nan]), 0)
    with pytest.raises(ValueError):
        make_index("unknown-core")


def test_metric_aliases_and_custom_metric():
    index = MTree(dim=2, metric="l1").__class__(dim=2, metric="manhattan")
    for row, point in enumerate(np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])):
        index.insert(point, row)
    assert index.query(np.array([0.0, 0.0]), 1)[1][0] == pytest.approx(0.0)

    # custom callable metric works on the M-Tree and brute force cores
    def euclid(a, b):
        return float(np.sqrt(((a - b) ** 2).sum()))

    tree = make_index("m-tree", dim=2, metric=euclid)
    for row, point in enumerate(np.array([[0.0, 0.0], [3.0, 4.0]])):
        tree.insert(point, row)
    ids, dists = tree.query(np.array([0.0, 0.0]), 2)
    np.testing.assert_allclose(dists, [0.0, 5.0])
    with pytest.raises(ValueError):
        make_index("kd-tree", dim=2, metric=euclid)
