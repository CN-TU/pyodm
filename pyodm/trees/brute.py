# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Exact brute-force kNN index (reference implementation).
#
# ****************************************************************************

"""Exact linear-scan kNN index.

This is the reference implementation used by the test-suite to validate the
M-Tree and KD-Tree cores.  It is also the fallback when a custom (user
supplied) distance function is requested.  Complexity is ``O(n)`` per query.
"""

from __future__ import annotations

import numpy as np

from .base import BaseIndex, check_point

__all__ = ["BruteForceIndex"]


class BruteForceIndex(BaseIndex):
    """Exact kNN by scanning every stored object.

    Parameters
    ----------
    dim : int, optional
        Dimensionality of the points; inferred from the first insertion.
    metric : str or callable, optional (default='euclidean')
        Any metric accepted by :class:`pyodm.trees.base.DistanceMetric`.
    """

    def __init__(self, dim=None, metric="euclidean"):
        super().__init__(dim=dim, metric=metric)
        self._order = []  # insertion ordered ids, used for vectorised scans
        self._matrix = None  # cached (n_objects, dim) array, invalidated on change

    # ------------------------------------------------------------------
    def insert(self, point, obj_id):
        point, obj_id = self._check_insert(point, obj_id)
        self._register(obj_id, point)
        self._order.append(obj_id)
        self._matrix = None

    def remove(self, obj_id):
        obj_id = int(obj_id)
        self._unregister(obj_id)
        self._order.remove(obj_id)
        self._matrix = None

    def update(self, obj_id, point):
        point = check_point(point, self.dim)
        self._unregister(obj_id)
        self._register(obj_id, point)
        if not self._order or self._order[-1] != obj_id:
            self._order.remove(obj_id)
            self._order.append(obj_id)
        self._matrix = None

    # ------------------------------------------------------------------
    def _matrix_view(self):
        if self._matrix is None:
            if not self._order:  # pragma: no cover - guarded by query
                raise ValueError("the index is empty")
            self._matrix = np.vstack([self._points[obj_id] for obj_id in self._order])
        return self._matrix

    def query(self, point, k=1):
        point = check_point(point, self.dim)
        if not self._order:
            raise ValueError("the index is empty")
        k = min(int(k), len(self._order))
        if k < 1:
            raise ValueError("k must be >= 1, got {}".format(k))
        matrix = self._matrix_view()
        dists = self.metric.many(matrix, point)
        ids = np.fromiter(self._order, dtype=np.int64, count=len(self._order))
        # Lexicographic (distance, id) ordering keeps results deterministic.
        order = np.lexsort((ids, dists))[:k]
        return ids[order], dists[order]

    # ------------------------------------------------------------------
    def to_arrays(self):
        """Return ``(ids, points)`` as contiguous arrays."""
        ids = np.fromiter(self._order, dtype=np.int64, count=len(self._order))
        return ids, self._matrix_view().copy()

    def __repr__(self):  # pragma: no cover - cosmetic
        return "BruteForceIndex(n_objects={}, dim={}, metric={!r})".format(
            len(self), self.dim, self.metric.name
        )
