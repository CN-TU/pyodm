"""Tests for the synthetic data generators and the animation helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)

import datasets  # noqa: E402  (path manipulation above)


@pytest.mark.parametrize("name", datasets.list_datasets())
def test_dataset_shapes_and_labels(name):
    X, y, meta = datasets.generate(name)
    assert X.ndim == 2 and X.shape[1] == 2
    assert X.shape[0] == y.shape[0]
    assert np.all(np.isfinite(X))
    assert meta["n_clusters"] >= 3  # the animations need "many" clusters
    # labelled blobs (if any) must have at least one point
    for label in np.unique(y):
        if label >= 0:
            assert np.count_nonzero(y == label) > 0


def test_dataset_registry_is_deterministic():
    for name in datasets.list_datasets():
        first = datasets.generate(name)[0]
        second = datasets.generate(name)[0]
        np.testing.assert_allclose(first, second)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        datasets.generate("does_not_exist")


def test_seed_override_changes_the_data():
    _, _, _ = datasets.generate("eight_blobs")
    a = datasets.generate("eight_blobs")[0]
    b = datasets.generate("eight_blobs", seed=99)[0]
    assert not np.allclose(a, b)


def test_trajectory_shape_matches_observers(blobs):
    """The animation helper must expose one trajectory per final observer."""
    import pyodm

    model = pyodm.ODM(R=1.0, random_state=0, track_history=True).fit(blobs)
    frames, positions, radii, alive = model.trajectories(n_frames=20)
    assert positions.shape == (20, model.observers.shape[0], blobs.shape[1])
    assert radii.shape == (20, model.observers.shape[0])
    assert alive.shape == (20, model.observers.shape[0])
    assert np.all(alive[-1])
