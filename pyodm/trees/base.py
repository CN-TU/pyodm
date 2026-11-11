# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Distance helpers and the common interface of the kNN indexes.
#
# ****************************************************************************

"""Distance helpers and the common kNN index interface.

ODM only needs three operations from a spatial index:

* ``insert(point, id)``
* ``remove(id)``
* ``query(point, k)``  -> the ``k`` nearest observers, sorted by distance

The original implementation relied on the private ``dSalmon`` M-Tree.  The
replacements in this package are self contained (numpy + scipy only) and exact:

``m-tree``
    A dynamic metric tree (Ciaccia et al., 1997) implemented in
    :mod:`pyodm.trees.mtree`.  This is the structure the paper uses and it
    supports moving/deleted objects without rebuilding.

``kd-tree``
    A scipy ``cKDTree`` wrapper with lazy deletions and amortised rebuilds
    (:mod:`pyodm.trees.kdtree`).  Fast for Euclidean data, but a static tree
    cannot delete objects, hence the periodic rebuild.

``brute``
    Exact linear scan.  Useful as a reference in tests and for very small
    observer sets.

All indexes return results ordered by ``(distance, object_id)`` so the
outcome is deterministic and consistent between cores whenever distances
are not exactly tied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "DistanceMetric",
    "BaseIndex",
    "SUPPORTED_METRICS",
    "check_point",
    "validate_k",
]

_METRIC_ALIASES = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "euclid": "euclidean",
    "manhattan": "manhattan",
    "l1": "manhattan",
    "cityblock": "manhattan",
    "city-block": "manhattan",
    "chebyshev": "chebyshev",
    "linf": "chebyshev",
    "inf": "chebyshev",
    "maximum": "chebyshev",
}

#: Metric name -> ``p`` parameter understood by ``scipy.spatial.cKDTree``.
SCIPY_P = {"euclidean": 2.0, "manhattan": 1.0, "chebyshev": np.inf}

#: Metric name -> ``scipy.spatial.distance.pdist`` keyword.
PDIST_METRIC = {"euclidean": "euclidean", "manhattan": "cityblock", "chebyshev": "chebyshev"}

SUPPORTED_METRICS = tuple(sorted(SCIPY_P))


class DistanceMetric(object):
    """Batched distance computation for the metrics understood by ODM.

    Parameters
    ----------
    metric : str or callable, optional (default='euclidean')
        One of ``'euclidean'``, ``'manhattan'``, ``'chebyshev'`` (aliases such
        as ``'l1'``/``'l2'``/``'linf'`` are accepted) or a callable
        ``metric(a, b) -> float``.  A callable is only supported by the
        ``'m-tree'`` and ``'brute'`` cores.
    """

    __slots__ = ("name", "p", "fn")

    def __init__(self, metric="euclidean"):
        if isinstance(metric, DistanceMetric):
            self.name, self.p, self.fn = metric.name, metric.p, metric.fn
            return
        if callable(metric):
            self.name = getattr(metric, "__name__", "custom")
            self.p = None
            self.fn = metric
            return
        if not isinstance(metric, str):
            raise TypeError("metric must be a string or a callable, got {!r}".format(type(metric)))
        key = metric.strip().lower()
        if key not in _METRIC_ALIASES:
            raise ValueError(
                "unsupported metric {!r}; choose from {}".format(metric, ", ".join(SUPPORTED_METRICS))
            )
        self.name = _METRIC_ALIASES[key]
        self.p = SCIPY_P[self.name]
        self.fn = None

    # ------------------------------------------------------------------
    @property
    def is_builtin(self):
        """True when the metric is one of the built-in Minkowski metrics."""
        return self.fn is None

    def many(self, points, point):
        """Distance between every row of ``points`` and a single ``point``."""
        points = np.asarray(points, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        if points.size == 0:
            return np.empty(0, dtype=np.float64)
        points = points.reshape(-1, point.size) if points.ndim == 1 else points
        if self.fn is not None:
            return np.fromiter(
                (self.fn(row, point) for row in points), dtype=np.float64, count=points.shape[0]
            )
        diff = points - point
        if self.name == "euclidean":
            return np.sqrt(np.einsum("ij,ij->i", diff, diff))
        if self.name == "manhattan":
            return np.abs(diff).sum(axis=1)
        return np.abs(diff).max(axis=1)

    def one(self, a, b):
        """Distance between two single points as a float.

        This is the hottest function of the M-Tree (millions of calls during a
        large fit), so the built-in metrics use dedicated scalar paths instead
        of the generic ``many`` broadcasting.
        """
        if self.fn is not None:
            return float(self.fn(a, b))
        diff = a - b
        if self.name == "euclidean":
            return float(np.sqrt(np.dot(diff, diff)))
        if self.name == "manhattan":
            return float(np.abs(diff).sum())
        return float(np.abs(diff).max())

    def matrix(self, points):
        """Full pairwise distance matrix of ``points`` (as ``float64``)."""
        points = np.asarray(points, dtype=np.float64)
        count = points.shape[0]
        if self.fn is not None:
            out = np.empty((count, count), dtype=np.float64)
            for i in range(count):
                out[i] = self.many(points, points[i])
            return out
        from scipy.spatial.distance import cdist

        return cdist(points, points, metric=PDIST_METRIC[self.name])

    def __repr__(self):  # pragma: no cover - cosmetic
        return "DistanceMetric({!r})".format(self.name)


def check_point(point, dim=None, name="point"):
    """Return ``point`` as an independent 1-D ``float64`` vector, validated.

    A copy is always made: ODM (and users) update their observer arrays in
    place, and an index that kept a reference to those arrays would see its
    stored coordinates change without any structural update (this used to
    silently corrupt the M-Tree covering radii).
    """
    point = np.array(point, dtype=np.float64, copy=True).ravel()
    if point.size == 0:
        raise ValueError("{} must not be empty".format(name))
    if not np.all(np.isfinite(point)):
        raise ValueError("{} contains NaN or infinite values".format(name))
    if dim is not None and point.size != dim:
        raise ValueError("expected a point with {} dimensions, got {}".format(dim, point.size))
    return point


def validate_k(k, size):
    """Validate a request for ``k`` neighbours against an index of ``size``."""
    try:
        k = int(k)
    except (TypeError, ValueError):
        raise ValueError("k must be an integer, got {!r}".format(k))
    if k < 1:
        raise ValueError("k must be >= 1, got {}".format(k))
    if size == 0:
        raise ValueError("the index is empty")
    return min(k, int(size))


class BaseIndex(ABC):
    """Abstract exact k-nearest-neighbour index over a dynamic set of objects."""

    def __init__(self, dim=None, metric="euclidean"):
        self.dim = None if dim is None else int(dim)
        self.metric = DistanceMetric(metric)
        self._points = {}
        self._size = 0

    # -- construction ---------------------------------------------------
    def _check_insert(self, point, obj_id):
        point = check_point(point, self.dim)
        if self.dim is None:
            self.dim = point.size
        try:
            obj_id = int(obj_id)
        except (TypeError, ValueError):
            raise TypeError("object ids must be integers, got {!r}".format(obj_id))
        if obj_id in self._points:
            raise KeyError("duplicate object id {!r}".format(obj_id))
        return point, obj_id

    def _register(self, obj_id, point):
        self._points[obj_id] = point
        self._size = len(self._points)

    def _unregister(self, obj_id):
        try:
            del self._points[obj_id]
        except KeyError:
            raise KeyError("unknown object id {!r}".format(obj_id))
        self._size = len(self._points)

    # -- required API ---------------------------------------------------
    @abstractmethod
    def insert(self, point, obj_id):
        """Insert ``point`` under the (unique) identifier ``obj_id``."""

    @abstractmethod
    def remove(self, obj_id):
        """Remove the object stored under ``obj_id``."""

    @abstractmethod
    def query(self, point, k=1):
        """Return ``(ids, distances)`` of the ``k`` nearest objects.

        Both arrays are sorted by increasing distance (ties are broken by
        object id) and have length ``min(k, len(self))``.
        """

    # -- common helpers -------------------------------------------------
    def update(self, obj_id, point):
        """Move the object ``obj_id`` to ``point`` (remove + insert)."""
        self.remove(obj_id)
        self.insert(point, obj_id)

    def query_many(self, points, k=1):
        """kNN for a batch of queries; returns ``(ids, distances)`` arrays.

        The default implementation loops over :meth:`query`.  Indexes that
        support vectorised queries (the KD-Tree) override this method.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points must be 2-D")
        k = min(int(k), len(self))
        if k < 1:
            raise ValueError("k must be >= 1, got {}".format(k))
        ids = np.empty((points.shape[0], k), dtype=np.int64)
        dists = np.empty((points.shape[0], k), dtype=np.float64)
        for row, point in enumerate(points):
            row_ids, row_dists = self.query(point, k)
            ids[row] = row_ids
            dists[row] = row_dists
        return ids, dists

    def __len__(self):
        return len(self._points)

    def __contains__(self, obj_id):
        return obj_id in self._points

    def __repr__(self):  # pragma: no cover - cosmetic
        return "{}(n_objects={}, dim={}, metric={!r})".format(
            type(self).__name__, len(self), self.dim, self.metric.name
        )

    @staticmethod
    def _finalise(ranked, k):
        """Turn an iterable of ``(distance, id)`` pairs into sorted arrays."""
        ranked = sorted(ranked, key=lambda item: (item[0], item[1]))[:k]
        if not ranked:
            raise ValueError("the index is empty")
        ids = np.fromiter((int(item[1]) for item in ranked), dtype=np.int64, count=len(ranked))
        dists = np.fromiter((float(item[0]) for item in ranked), dtype=np.float64, count=len(ranked))
        return ids, dists
