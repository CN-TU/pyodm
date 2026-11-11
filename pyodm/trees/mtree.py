# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Self contained M-Tree used as the default ODM kNN core.
#
# Reference   : P. Ciaccia, M. Patella and P. Zezula, "M-tree: An efficient
#               access method for similarity search in metric spaces",
#               Proc. of the 23rd VLDB Conference, 1997.
#
# ****************************************************************************

"""Exact dynamic metric tree (M-Tree).

The M-Tree is the index the ODM paper describes and the structure the original
implementation obtained from the private ``dSalmon`` package.  This module
re-implements the parts ODM needs (dynamic insertion, deletion and exact
k-nearest-neighbour search) in plain Python/numpy, so no external dependency is
required any more.

Design
------
* An **internal entry** stores a *routing object* by value, the covering
  radius ``r`` of the subtree below it and a pointer to the child node.
* A **leaf entry** stores the object itself plus its identifier.
* Covering radii are kept as tight as possible and are recomputed exactly
  after every structural change.
* kNN search is best-first with branch and bound: a subtree is skipped as soon
  as its lower bound ``max(0, d(q, routing_object) - r)`` cannot improve the
  current k-th distance, hence results are exact.
* Ties are broken by object id so the output is deterministic and matches the
  brute-force reference (:class:`pyodm.trees.brute.BruteForceIndex`).
"""

from __future__ import annotations

import heapq

import numpy as np

from .base import BaseIndex, check_point

__all__ = ["MTree"]


class _Entry(object):
    """A node entry: either an object (leaf) or a pointer to a child (internal)."""

    __slots__ = ("obj", "obj_id", "child", "r")

    def __init__(self, obj, obj_id=None, child=None, r=0.0):
        self.obj = obj
        self.obj_id = obj_id
        self.child = child
        self.r = float(r)

    @property
    def is_object(self):
        return self.child is None

    def __repr__(self):  # pragma: no cover - cosmetic
        if self.is_object:
            return "_Entry(id={}, r={:.3f})".format(self.obj_id, self.r)
        return "_Entry(internal, r={:.3f})".format(self.r)


class _Node(object):
    """An M-Tree node; ``entries`` holds its routing/object entries."""

    __slots__ = ("entries", "leaf")

    def __init__(self, leaf=True, entries=None):
        self.entries = list(entries) if entries else []
        self.leaf = bool(leaf)

    def __repr__(self):  # pragma: no cover - cosmetic
        return "_Node(leaf={}, n_entries={})".format(self.leaf, len(self.entries))


class MTree(BaseIndex):
    """Exact k-NN search over a dynamic set of metric-space objects.

    Parameters
    ----------
    dim : int, optional
        Dimensionality of the points; inferred from the first insertion.
    metric : str or callable, optional (default='euclidean')
        Any metric accepted by :class:`pyodm.trees.base.DistanceMetric`.
    max_entries : int, optional (default=16)
        Node capacity; nodes split when they exceed it.
    """

    def __init__(self, dim=None, metric="euclidean", max_entries=16, min_entries=None):
        super().__init__(dim=dim, metric=metric)
        self.max_entries = int(max_entries)
        if self.max_entries < 4:
            raise ValueError("max_entries must be >= 4, got {}".format(self.max_entries))
        self.min_entries = (
            int(min_entries) if min_entries is not None else max(2, self.max_entries // 2)
        )
        if not 1 <= self.min_entries <= self.max_entries:
            raise ValueError("min_entries must be within [1, max_entries]")
        self._root = None
        self._n_queries = 0
        self._n_metric_calls = 0

    # ------------------------------------------------------------------
    # insertion
    # ------------------------------------------------------------------
    def insert(self, point, obj_id):
        point, obj_id = self._check_insert(point, obj_id)
        entry = _Entry(point, obj_id=obj_id)
        if self._root is None:
            self._root = _Node(leaf=True, entries=[entry])
        else:
            promotion = self._insert_rec(self._root, entry)
            if promotion is not None:
                # The root split: grow the tree by one level.
                self._root = _Node(leaf=False, entries=promotion)
        self._register(obj_id, point)

    def _insert_rec(self, node, entry):
        """Insert ``entry`` below ``node``; return split promotions or None.

        A promotion is a list of two entries that must replace ``node`` inside
        its parent: one keeps the old node, the other points at the sibling
        created by the split.
        """
        if node.leaf:
            node.entries.append(entry)
            return self._split(node) if len(node.entries) > self.max_entries else None

        # ChooseSubtree: minimal radius enlargement, then minimal distance.
        target = None
        best_cost = None
        for candidate in node.entries:
            dist = self.metric.one(candidate.obj, entry.obj)
            cost = (max(dist, candidate.r) - candidate.r, dist)
            if best_cost is None or cost < best_cost:
                target, best_cost = candidate, cost

        promotion = self._insert_rec(target.child, entry)
        if promotion is None:
            # Radius may only grow here; recompute it exactly.
            target.r = self._cover(target.obj, target.child.entries)
            return None
        # The child split: ``promotion[0]`` is the replacement entry for the
        # old child (tight radius + new routing object), ``promotion[1]`` is
        # the new sibling.  Both are attached here.
        target.obj = promotion[0].obj
        target.r = promotion[0].r
        node.entries.append(promotion[1])
        return self._split(node) if len(node.entries) > self.max_entries else None

    # ------------------------------------------------------------------
    # splitting
    # ------------------------------------------------------------------
    def _pick_pivots(self, entries):
        """Choose the two most distant entries (PickPivots2 heuristic)."""
        best_pair = (entries[0], entries[1])
        best_dist = -1.0
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                dist = self.metric.one(entries[i].obj, entries[j].obj)
                self._n_metric_calls += 1
                if dist > best_dist:
                    best_dist, best_pair = dist, (entries[i], entries[j])
        return best_pair

    def _partition(self, entries, pivot_a, pivot_b):
        """Assign entries to the closer pivot; never leaves a side empty."""
        group_a, group_b = [], []
        for entry in entries:
            dist_a = self.metric.one(pivot_a.obj, entry.obj)
            dist_b = self.metric.one(pivot_b.obj, entry.obj)
            self._n_metric_calls += 2
            (group_a if dist_a <= dist_b else group_b).append(entry)
        if not group_a or not group_b:
            empty_is_a = not group_a
            donor = group_b if empty_is_a else group_a
            target = pivot_a if empty_is_a else pivot_b
            moved = min(donor, key=lambda e: (self.metric.one(target.obj, e.obj), e.obj_id or 0))
            donor.remove(moved)
            (group_a if empty_is_a else group_b).append(moved)
        return group_a, group_b

    def _cover(self, routing_obj, entries):
        """Smallest radius around ``routing_obj`` covering all ``entries``."""
        radius = 0.0
        for entry in entries:
            radius = max(radius, self.metric.one(routing_obj, entry.obj) + entry.r)
            self._n_metric_calls += 1
        return radius

    def _make_entry(self, node, routing=None):
        """Build an entry pointing at ``node`` with a tight covering radius."""
        if not node.entries:  # pragma: no cover - guarded by callers
            raise ValueError("cannot build an entry for an empty node")
        if routing is None:
            routing = node.entries[0].obj
        return _Entry(routing, child=node, r=self._cover(routing, node.entries))

    def _split(self, node):
        """Split the overflowing ``node`` and return its two replacement entries."""
        entries = node.entries
        pivot_a, pivot_b = self._pick_pivots(entries)
        group_a, group_b = self._partition(entries, pivot_a, pivot_b)
        node.entries = group_a
        sibling = _Node(leaf=node.leaf, entries=group_b)
        # Keep the old node around: only its routing object and radius change,
        # which is what the first promotion entry encodes.
        return [
            _Entry(pivot_a.obj, child=node, r=self._cover(pivot_a.obj, group_a)),
            _Entry(pivot_b.obj, child=sibling, r=self._cover(pivot_b.obj, group_b)),
        ]

    # ------------------------------------------------------------------
    # deletion
    # ------------------------------------------------------------------
    def remove(self, obj_id):
        obj_id = int(obj_id)
        if obj_id not in self._points:
            raise KeyError("unknown object id {!r}".format(obj_id))
        point = self._points[obj_id]
        removed = self._delete_rec(self._root, point, obj_id, self._root)
        if not removed:
            raise KeyError("object id {!r} was registered but missing in the tree".format(obj_id))
        self._unregister(obj_id)
        # Collapse an internal root that has a single child.
        while self._root is not None and not self._root.leaf and len(self._root.entries) == 1:
            self._root = self._root.entries[0].child
        if self._root is not None and not self._root.leaf and not self._root.entries:
            self._root = None

    def _delete_rec(self, node, point, obj_id, root):
        """Remove the object from ``node``'s subtree; return True on success."""
        if node.leaf:
            for index, entry in enumerate(node.entries):
                if entry.obj_id == obj_id:
                    node.entries.pop(index)
                    return True
            return False

        for index, entry in enumerate(list(node.entries)):
            current = node.entries[index]
            if not current.is_object and self.metric.one(current.obj, point) > current.r + 1e-9:
                continue  # the object cannot be inside this subtree
            child = current.child
            if not self._delete_rec(child, point, obj_id, root):
                continue
            if not child.entries:
                node.entries.pop(index)
            elif child is not root and len(child.entries) < self.min_entries:
                # Condense: drop the sparse node and reinsert its objects.
                node.entries.pop(index)
                orphans = self._collect_objects(child)
                for orphan in orphans:
                    self._reinsert(orphan)
            else:
                current.r = self._cover(current.obj, child.entries)
            return True
        return False

    def _reinsert(self, entry):
        """Reinsert a leaf entry whose id is still registered."""
        if self._root is None:
            self._root = _Node(leaf=True, entries=[entry])
            return
        promotion = self._insert_rec(self._root, entry)
        if promotion is not None:
            self._root = _Node(leaf=False, entries=promotion)

    @staticmethod
    def _collect_objects(node):
        """Collect every leaf entry below ``node`` (recursively)."""
        if node.leaf:
            return list(node.entries)
        out = []
        for entry in node.entries:
            out.extend(MTree._collect_objects(entry.child))
        return out

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    def query(self, point, k=1):
        point = check_point(point, self.dim)
        if self._root is None:
            raise ValueError("the index is empty")
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1, got {}".format(k))
        k = min(k, len(self))
        self._n_queries += 1
        best = []  # max-heap of (negated distance, negated id, id)
        counter = 0
        heap = [(0.0, counter, self._root)]
        while heap:
            bound, _, node = heapq.heappop(heap)
            if len(best) == k and bound > -best[0][0]:
                break
            for entry in node.entries:
                dist = self.metric.one(entry.obj, point)
                self._n_metric_calls += 1
                if entry.child is None:
                    item = (-dist, -entry.obj_id, entry.obj_id)
                    if len(best) < k:
                        heapq.heappush(best, item)
                    elif item > best[0]:
                        heapq.heapreplace(best, item)
                else:
                    lower = dist - entry.r
                    if lower < 0.0:
                        lower = 0.0
                    if len(best) == k and lower > -best[0][0]:
                        continue
                    counter += 1
                    heapq.heappush(heap, (lower, counter, entry.child))
        ranked = sorted((-neg_dist, obj_id) for neg_dist, _, obj_id in best)
        ids = np.fromiter((obj_id for _, obj_id in ranked), dtype=np.int64, count=len(ranked))
        dists = np.fromiter((dist for dist, _ in ranked), dtype=np.float64, count=len(ranked))
        return ids, dists

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def items(self):
        """Return ``[(id, point)]`` for every stored object."""
        return [(obj_id, self._points[obj_id]) for obj_id in self._points]

    def stats(self):
        """Dictionary with tree shape and query statistics (for debugging)."""
        height = 0
        entries = 0

        def walk(node, depth):
            nonlocal height, entries
            height = max(height, depth)
            entries += len(node.entries)
            if not node.leaf:
                for entry in node.entries:
                    walk(entry.child, depth + 1)

        if self._root is not None:
            walk(self._root, 1)
        return {
            "n_objects": len(self),
            "height": height,
            "n_entries": entries,
            "n_queries": self._n_queries,
            "n_metric_calls": self._n_metric_calls,
            "max_entries": self.max_entries,
        }

    def __repr__(self):  # pragma: no cover - cosmetic
        shape = self.stats()
        return "MTree(n_objects={n_objects}, height={height}, dim={}, metric={!r})".format(
            self.dim, self.metric.name, **shape
        )
