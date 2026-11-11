"""Shared test fixtures for the pyODM test-suite."""

from __future__ import annotations

import numpy as np
import pytest


def make_blobs(n_per_cluster=200, centers=None, spread=0.6, seed=0):
    """Return a 2-D data set with well separated Gaussian clusters."""
    rng = np.random.RandomState(seed)
    if centers is None:
        centers = [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]]
    parts = [rng.normal(center, spread, size=(n_per_cluster, len(center))) for center in centers]
    return np.vstack(parts)


@pytest.fixture(scope="session")
def blobs():
    return make_blobs()
