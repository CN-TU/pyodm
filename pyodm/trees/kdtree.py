# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : scipy cKDTree wrapper with lazy updates for ODM.
#
# ****************************************************************************

"""Static KD-tree core with lazy deletions and amortised rebuilds.

``scipy.spatial.cKDTree`` is a very fast exact kNN structure for Euclidean
data, but it is immutable: objects cannot be removed or moved.  ODM moves
observers on almost every sample, so this wrapper keeps

* a ``cKDTree`` built from a snapshot of the objects,
* a ``pending`` set of ids changed since that snapshot,
* a rebuild once ``len(pending) > rebuild_ratio * n_objects``.

A query combines the tree results (stale entries filtered out) with a linear
scan of the (small) pending set, so the answer is always exact.  The amortised
cost stays ``O(log n)`` per operation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .base import BaseIndex, check_point

__all__ = ["KDTreeIndex"]


class KDTreeIndex(BaseIndex):
    """Exact dynamic kNN over a ``cKDTree`` with periodic rebuilds.

    Parameters
    ----------
    dim : int, optional
        Dimensionality of the points; inferred from the first insertion.
    metric : str, optional (default='euclidean')
        One of ``'euclidean'``, ``'manhattan'``, ``'chebyshev'``.
    leaf_size : int, optional (default=16)
        Leaf size handed to ``cKDTree``.
    rebuild_ratio : float, optional (default=0.1)
        Rebuild once more than this fraction of the objects changed.
    """

    def __init__(self, dim=None, metric="euclidean", leaf_size=16, rebuild_ratio=0.1):
        super().__init__(dim=dim, metric=metric)
        if not self.metric.is_builtin:
            raise ValueError("the kd-tree core only supports built-in Minkowski metrics")
        self.leaf_size = int(leaf_size)
        self.rebuild_ratio = float(rebuild_ratio)
        if not 0.0 < self.rebuild_ratio <= 1.0:
            raise ValueError("rebuild_ratio must be within (0, 1]")
        self._tree = None
        self._ids = np.empty(0, dtype=np.int64)
        self._snapshot = {}  # id -> point indexed by self._tree
        self._pending = set()  # ids changed/added/removed since the snapshot
        self._n_rebuilds = 0

    # ------------------------------------------------------------------
    # tree maintenance
    # ------------------------------------------------------------------
    def _rebuild(self):
        if not self._snapshot:
            self._tree = None
            self._ids = np.empty(0, dtype=np.int64)
            self._pending.clear()
            return
        self._ids = np.fromiter(self._snapshot.keys(), dtype=np.int64, count=len(self._snapshot))
        points = np.vstack([self._snapshot[int(obj_id)] for obj_id in self._ids])
        self._tree = cKDTree(points, leafsize=self.leaf_size)
        self._pending.clear()
        self._n_rebuilds += 1

    def _maybe_rebuild(self):
        if self._tree is None or len(self._pending) > self.rebuild_ratio * max(len(self), 1):
            self._rebuild()

    # ------------------------------------------------------------------
    def insert(self, point, obj_id):
        point, obj_id = self._check_insert(point, obj_id)
        self._register(obj_id, point)
        self._snapshot[obj_id] = point
        self._pending.add(obj_id)
        self._maybe_rebuild()

    def remove(self, obj_id):
        obj_id = int(obj_id)
        if obj_id not in self._points:
            raise KeyError("unknown object id {!r}".format(obj_id))
        self._unregister(obj_id)
        self._snapshot.pop(obj_id, None)
        self._pending.add(obj_id)
        self._maybe_rebuild()

    def update(self, obj_id, point):
        point = check_point(point, self.dim)
        if obj_id not in self._points:
            raise KeyError("unknown object id {!r}".format(obj_id))
        self._register(obj_id, point)
        self._snapshot[obj_id] = point
        self._pending.add(obj_id)
        self._maybe_rebuild()

    def bulk_load(self, points, ids=None):
        """Build the index from scratch; much faster than ``n`` insertions."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("points must be a non-empty 2-D array")
        if self.dim is None:
            self.dim = int(points.shape[1])
        if ids is None:
            ids = np.arange(points.shape[0], dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape[0] != points.shape[0]:
            raise ValueError("ids and points must have the same length")
        self._points = {int(obj_id): points[row] for row, obj_id in enumerate(ids)}
        self._size = len(self._points)
        self._snapshot = dict(self._points)
        self._pending.clear()
        self._rebuild()

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def query(self, point, k=1):
        point = check_point(point, self.dim)
        if not self._points:
            raise ValueError("the index is empty")
        self._maybe_rebuild()
        k = min(int(k), len(self._points))
        if k < 1:
            raise ValueError("k must be >= 1, got {}".format(k))
        ids, dists = self._query_live(point, k)
        order = np.lexsort((ids, dists))[:k]
        return ids[order], dists[order]

    def _query_live(self, point, k):
        """Exact kNN = tree hits with stale ids removed + scan of pending ids."""
        size = int(self._ids.size)
        request = min(size, k + len(self._pending)) if size else 0
        while True:
            found = []
            if request > 0:
                raw_dists, raw_indices = self._tree.query(point, k=request, p=self.metric.p)
                raw_dists = np.atleast_1d(np.asarray(raw_dists, dtype=np.float64))
                raw_indices = np.atleast_1d(np.asarray(raw_indices, dtype=np.int64))
                valid = raw_indices < size
                for dist, index in zip(raw_dists[valid], raw_indices[valid]):
                    obj_id = int(self._ids[index])
                    if obj_id in self._points and obj_id not in self._pending:
                        found.append((float(dist), obj_id))
            # Objects that moved/arrived since the snapshot are not in the tree.
            if self._pending:
                pending_ids = sorted(
                    obj_id for obj_id in self._pending if obj_id in self._points
                )
                if pending_ids:
                    pending_points = np.vstack([self._points[obj_id] for obj_id in pending_ids])
                    pending_dists = self.metric.many(pending_points, point)
                    found.extend(
                        (float(dist), int(obj_id)) for dist, obj_id in zip(pending_dists, pending_ids)
                    )
            found.sort()
            if len(found) >= k or request >= size:
                if not found:  # pragma: no cover - guarded by callers
                    raise ValueError("the index is empty")
                top = found[:k]
                return (
                    np.fromiter((obj_id for _, obj_id in top), dtype=np.int64, count=len(top)),
                    np.fromiter((dist for dist, _ in top), dtype=np.float64, count=len(top)),
                )
            request = min(size, request * 2)

    def query_many(self, points, k=1):
        """Vectorised kNN for a batch of points (rebuilds first)."""
        points = np.asarray(points, dtype=np.float64)
        points = points.reshape(1, -1) if points.ndim == 1 else points
        if not self._points:
            raise ValueError("the index is empty")
        if self._pending:
            self._rebuild()
        k = min(int(k), len(self._points))
        if k < 1:
            raise ValueError("k must be >= 1, got {}".format(k))
        raw_dists, raw_indices = self._tree.query(points, k=k, p=self.metric.p)
        raw_dists = np.asarray(raw_dists, dtype=np.float64)
        raw_indices = np.asarray(raw_indices, dtype=np.int64)
        if raw_dists.ndim == 1:
            raw_dists = raw_dists.reshape(-1, 1)
            raw_indices = raw_indices.reshape(-1, 1)
        ids = self._ids[raw_indices]
        for row in range(ids.shape[0]):
            order = np.lexsort((ids[row], raw_dists[row]))
            ids[row] = ids[row][order]
            raw_dists[row] = raw_dists[row][order]
        return ids, raw_dists

    # ------------------------------------------------------------------
    def to_arrays(self):
        """Return ``(ids, points)`` of the live objects as arrays."""
        if not self._points:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, self.dim or 0), dtype=np.float64),
            )
        ids = np.fromiter(self._points.keys(), dtype=np.int64, count=len(self._points))
        points = np.vstack([self._points[int(obj_id)] for obj_id in ids])
        return ids, points

    def stats(self):
        return {
            "n_objects": len(self),
            "n_pending": len(self._pending),
            "n_rebuilds": self._n_rebuilds,
            "leaf_size": self.leaf_size,
        }

    def __repr__(self):  # pragma: no cover - cosmetic
        return "KDTreeIndex(n_objects={}, n_pending={}, dim={}, metric={!r})".format(
            len(self), len(self._pending), self.dim, self.metric.name
        )
