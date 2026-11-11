# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Factory for the kNN cores used by ODM.
#
# ****************************************************************************

"""Public entry point for the kNN cores.

The private ``dSalmon`` M-Tree used by the original code is replaced by
self-contained implementations selectable through ``core``:

===========  ===========================================================
``'m-tree'`` :class:`pyodm.trees.mtree.MTree` -- the default, dynamic,
             metric-space tree used in the paper.
``'kd-tree'`` :class:`pyodm.trees.kdtree.KDTreeIndex` -- scipy ``cKDTree``
             with lazy deletions; only Minkowski metrics.
``'brute'``  :class:`pyodm.trees.brute.BruteForceIndex` -- exact scan.
===========  ===========================================================

``make_index`` is the only function the ODM implementation needs.
"""

from __future__ import annotations

from .base import PDIST_METRIC, SUPPORTED_METRICS, BaseIndex, DistanceMetric
from .brute import BruteForceIndex
from .kdtree import KDTreeIndex
from .mtree import MTree

__all__ = [
    "BaseIndex",
    "DistanceMetric",
    "SUPPORTED_METRICS",
    "PDIST_METRIC",
    "MTree",
    "KDTreeIndex",
    "BruteForceIndex",
    "make_index",
    "SUPPORTED_CORES",
]

SUPPORTED_CORES = ("m-tree", "kd-tree", "brute")

_CORE_ALIASES = {
    "m-tree": "m-tree",
    "mtree": "m-tree",
    "m_tree": "m-tree",
    "mt": "m-tree",
    "kd-tree": "kd-tree",
    "kdtree": "kd-tree",
    "kd_tree": "kd-tree",
    "kd": "kd-tree",
    "brute": "brute",
    "brute-force": "brute",
    "bruteforce": "brute",
    "linear": "brute",
}


def make_index(core="m-tree", dim=None, metric="euclidean", **kwargs):
    """Instantiate one of the exact kNN cores.

    Parameters
    ----------
    core : str, optional (default='m-tree')
        ``'m-tree'``, ``'kd-tree'`` or ``'brute'`` (aliases such as
        ``'kdtree'`` are accepted).
    dim : int, optional
        Dimensionality of the objects; inferred on the first insert.
    metric : str or callable, optional (default='euclidean')
        Distance metric.  Callables are only supported by ``'m-tree'`` and
        ``'brute'``.
    **kwargs
        Extra options forwarded to the index constructor (for example
        ``max_entries`` for the M-Tree or ``leaf_size``/``rebuild_ratio`` for
        the KD-Tree).

    Returns
    -------
    index : :class:`pyodm.trees.base.BaseIndex`
    """
    if isinstance(core, BaseIndex):  # allow injecting a pre-built index
        return core
    if not isinstance(core, str):
        raise TypeError("core must be a string, got {!r}".format(type(core)))
    key = core.strip().lower().replace(" ", "")
    if key not in _CORE_ALIASES:
        raise ValueError(
            "unsupported core {!r}; choose from {}".format(core, ", ".join(SUPPORTED_CORES))
        )
    core = _CORE_ALIASES[key]

    if core == "m-tree":
        return MTree(dim=dim, metric=metric, **kwargs)
    if core == "kd-tree":
        if not DistanceMetric(metric).is_builtin:
            raise ValueError("the 'kd-tree' core requires a built-in metric")
        return KDTreeIndex(dim=dim, metric=metric, **kwargs)
    return BruteForceIndex(dim=dim, metric=metric)
