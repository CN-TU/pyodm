# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Name        : __init__.py
# Description : Observers-based Data Modeling algorithm (pyODM)
# Author      : Fares Meghdouri
#
# Notes       : The private dSalmon M-Tree originally used as the kNN core has
#               been replaced by self-contained, exact implementations shipped
#               with this package (see pyodm/trees).
#
# Change-log  : 28-10-2019: //FM// Uploaded v1 with a KD-tree core.
#               06-02-2020: //FM// Rewrote the core v2 (M-tree as described
#                           in the paper - KD-tree core will come back in the
#                           next version).
#               08-06-2020: //FM// New features: select the number of
#                           observers, rename parameters, add parameters.
#               11-11-2026: //FM// Bug fixes, self-contained kNN cores and
#                           observer trajectory tracking.
#
#******************************************************************************

"""pyODM -- Observers-based Data Modeling.

The public API is the :class:`pyodm.ODM` estimator::

    import pyodm
    model = pyodm.ODM(random_state=1).fit(X)
    model.observers           # the coreset
    model.outlierness(X_test) # outlierness scores
    model.predict(X_test)     # binary anomaly labels
"""

from .odm import ODM
from .trees import MTree, KDTreeIndex, BruteForceIndex, make_index

#: Version of the package (kept in sync with pyproject.toml / setup.py).
__version__ = "3.1.0"

__all__ = [
    "ODM",
    "MTree",
    "KDTreeIndex",
    "BruteForceIndex",
    "make_index",
    "__version__",
]
