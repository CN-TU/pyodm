"""Tests for the ODM estimator: correctness, parameter handling, regressions."""

from __future__ import annotations

import numpy as np
import pytest

import pyodm
from pyodm._utils import check_random_state

CORES = ["m-tree", "kd-tree", "brute"]


# ---------------------------------------------------------------------------
# Regression tests for the bugs found in the original implementation.
# ---------------------------------------------------------------------------
def test_does_not_depend_on_private_dsalmon_package():
    """The old ``from dSalmon.trees import MTree`` import made pyodm unusable."""
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "dSalmon" or name.startswith("dSalmon."):
            pytest.skip("dSalmon installed in this environment")
    importlib.import_module("pyodm")
    assert not any(name.startswith("dSalmon") for name in sys.modules)


def test_fit_accepts_random_state_object(blobs):
    """``np.random.seed(random_state)`` raised TypeError for RandomState objects."""
    model = pyodm.ODM(random_state=np.random.RandomState(0)).fit(blobs)
    assert model.observers.shape[0] > 0
    generator = np.random.default_rng(1)
    model = pyodm.ODM(random_state=generator).fit(blobs)
    assert model.observers.shape[0] > 0


def test_get_params_works_before_and_after_fit(blobs):
    """``get_params`` referenced ``self.contamination``, which never existed."""
    model = pyodm.ODM()
    params = model.get_params()
    assert "factor" in params and "core" in params
    model.fit(blobs)
    assert model.get_params()["core"] == "m-tree"


def test_chunksize_larger_than_dataset(blobs):
    """``np.array_split(X, n/chunksize)`` crashed for float section counts."""
    model = pyodm.ODM(chunksize=len(blobs) * 10, random_state=0).fit(blobs)
    assert model.observers.shape[0] > 0
    scores = model.outlierness(blobs[:20], chunksize=1000)
    assert scores.shape == (20,)


def test_outlierness_on_small_input(blobs):
    """``m/chunksize`` was a float, so any input smaller than 1000 samples crashed."""
    model = pyodm.ODM(random_state=0).fit(blobs)
    scores = model.outlierness(blobs[:5])
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    labels = model.predict(blobs[:5], contamination=0.5)
    assert labels.dtype == bool


def test_m_larger_than_created_observers_keeps_arrays_consistent(blobs):
    """The old code appended random points to ``observers`` only, breaking shapes."""
    model = pyodm.ODM(m=len(blobs) * 2, random_state=0).fit(blobs[:200])
    n = model.observers.shape[0]
    assert model.population.shape == (n,)
    assert model.radius.shape == (n,)


def test_alpha_and_beta_are_used(blobs):
    """``alpha`` was documented but never applied; ``beta`` did not scale radii."""
    model = pyodm.ODM(alpha=0.5, random_state=0).fit(blobs)
    base = pyodm.ODM(alpha=1.0, random_state=0).fit(blobs)
    assert model.observers.shape != base.observers.shape or not np.allclose(
        model.observers, base.observers
    )

    small = pyodm.ODM(R=None, beta=0.5, random_state=0).fit(blobs)
    large = pyodm.ODM(R=None, beta=2.0, random_state=0).fit(blobs)
    assert small.R_ < large.R_
    assert small.observers.shape[0] >= large.observers.shape[0]


def test_label_shuffling_regression():
    """``self._shuffle(X, labels)`` returned the wrong object and raised ValueError."""
    model = pyodm.ODM(random_state=0)
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)
    shuffled_x, shuffled_y = model._shuffle(X, y)
    assert shuffled_x.shape == X.shape
    assert shuffled_y.shape == y.shape
    # rows must stay paired
    pairs = {(row[0], label) for row, label in zip(shuffled_x.tolist(), shuffled_y.tolist())}
    expected = {(row[0], label) for row, label in zip(X.tolist(), y.tolist())}
    assert pairs == expected


def test_refit_resets_state(blobs):
    model = pyodm.ODM(random_state=0)
    model.fit(blobs)
    first = model.observers.copy()
    model.fit(blobs[:100])
    assert model.observers.shape[0] != first.shape[0] or not np.allclose(model.observers, first)
    assert model.n_samples_ == 100


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        pyodm.ODM(core="dSalmon")
    with pytest.raises(ValueError):
        pyodm.ODM(mode="average")
    with pytest.raises(ValueError):
        pyodm.ODM(distance_estimation="banana")
    with pytest.raises(ValueError):
        pyodm.ODM(rho=0)
    with pytest.raises(ValueError):
        pyodm.ODM(core="m-tree").fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_unfitted_model_raises():
    model = pyodm.ODM()
    with pytest.raises(RuntimeError):
        model.outlierness([[0.0, 0.0]])
    with pytest.raises(RuntimeError):
        model.labels([[0.0, 0.0]])


# ---------------------------------------------------------------------------
# Algorithmic behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("core", CORES)
def test_all_cores_produce_the_same_model(core, blobs):
    reference = pyodm.ODM(core="m-tree", random_state=42, shuffle_data=False).fit(blobs)
    other = pyodm.ODM(core=core, random_state=42, shuffle_data=False).fit(blobs)
    np.testing.assert_allclose(other.observers, reference.observers)
    assert np.array_equal(other.population, reference.population)


def test_population_covers_all_samples(blobs):
    model = pyodm.ODM(random_state=0, shuffle_data=False).fit(blobs)
    assert int(model.population.sum()) == blobs.shape[0]


def test_model_keeps_the_structure_of_the_data(blobs):
    """The observers must stay near the four cluster centres."""
    model = pyodm.ODM(random_state=0, m=8).fit(blobs)
    assert model.observers.shape[0] == 8
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
    # every centre must be covered by at least one observer
    for center in centers:
        assert np.min(np.linalg.norm(model.observers - center, axis=1)) < 1.0


def test_single_sample_and_identical_points():
    model = pyodm.ODM(random_state=0).fit(np.zeros((1, 2)))
    assert model.observers.shape == (1, 2)
    model = pyodm.ODM(random_state=0).fit(np.zeros((50, 2)))
    assert model.observers.shape[0] == 1


def test_outlierness_chunking_does_not_change_scores(blobs):
    model = pyodm.ODM(random_state=0).fit(blobs)
    whole = model.outlierness(blobs)
    chunked = model.outlierness(blobs, chunksize=17)
    np.testing.assert_allclose(whole, chunked)


def test_outlier_scores_rank_far_points_higher(blobs):
    model = pyodm.ODM(random_state=0).fit(blobs)
    outliers = np.array([[50.0, 50.0], [51.0, 51.0], [-50.0, -50.0]])
    scores = model.outlierness(np.vstack([blobs[:10], outliers]))
    assert scores[-3:].min() > scores[:10].max()


def test_labels_reference_observer_rows(blobs):
    model = pyodm.ODM(random_state=0).fit(blobs)
    ids = model.labels(blobs[:5])
    assert ids.shape == (5,)
    assert ids.min() >= 0 and ids.max() < model.observers.shape[0]
    manual = np.argmin(np.linalg.norm(model.observers[None, :, :] - blobs[:5, None, :], axis=2), axis=1)
    assert np.array_equal(ids, manual)


def test_outlierness_modes(blobs):
    model = pyodm.ODM(random_state=0).fit(blobs)
    for mode in ("median", "mean", "max", "min", "sum"):
        scores = model.outlierness(blobs[:10], O=3, mode=mode)
        assert scores.shape == (10,)
    with pytest.raises(ValueError):
        model.outlierness(blobs[:10], mode="average")


def test_parallel_and_chunked_runs(blobs):
    single = pyodm.ODM(random_state=0, core="brute").fit(blobs)
    chunked = pyodm.ODM(random_state=0, core="brute", chunksize=25).fit(blobs)
    parallel = pyodm.ODM(
        random_state=0, core="brute", chunksize=25, parallelize=True, n_cores=2
    ).fit(blobs)
    assert chunked.observers.shape[0] >= single.observers.shape[0]
    # parallel per-chunk radius estimation changes R, but the output must be valid
    assert int(parallel.population.sum()) == blobs.shape[0]
    assert np.all(np.isfinite(parallel.observers))


def test_metric_parameter_changes_model(blobs):
    euclid = pyodm.ODM(metric="euclidean", core="brute", random_state=0).fit(blobs)
    manhattan = pyodm.ODM(metric="manhattan", core="brute", random_state=0).fit(blobs)
    assert euclid.observers.shape != manhattan.observers.shape or not np.allclose(
        euclid.observers, manhattan.observers
    )


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
def test_trajectories_replay_matches_final_model(blobs):
    model = pyodm.ODM(random_state=0, track_history=True).fit(blobs)
    frames, positions, radii, alive = model.trajectories(n_frames=40)
    assert frames.shape == (40,)
    last = alive[-1]
    assert last.sum() == model.observers.shape[0]
    np.testing.assert_allclose(positions[-1][last], model.observers)
    np.testing.assert_allclose(radii[-1][last], model.radius)


def test_trajectories_respect_selected_observers(blobs):
    """With ``m`` set, trajectories must contain only the kept observers."""
    model = pyodm.ODM(m=8, random_state=0, track_history=True).fit(blobs)
    frames, positions, radii, alive = model.trajectories(n_frames=30)
    assert positions.shape[1] == model.observers.shape[0] == 8
    last = alive[-1]
    assert last.all()
    np.testing.assert_allclose(positions[-1], model.observers)
    np.testing.assert_allclose(radii[-1], model.radius)


def test_trajectories_with_chunks(blobs):
    """Chunked fits must produce a consistent (global-id) event log."""
    model = pyodm.ODM(chunksize=40, random_state=0, track_history=True).fit(blobs)
    frames, positions, radii, alive = model.trajectories(n_frames=25)
    assert positions.shape[1] == model.observers.shape[0]
    np.testing.assert_allclose(positions[-1], model.observers)


def test_trajectories_require_tracking(blobs):
    model = pyodm.ODM(random_state=0).fit(blobs)
    assert model.history_ is None
    with pytest.raises(RuntimeError):
        model.trajectories()


def test_trajectory_observers_move_over_time():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 2))
    model = pyodm.ODM(random_state=0, R=1.0, track_history=True).fit(X)
    frames, positions, _, alive = model.trajectories(n_frames=10)
    movement = 0.0
    for obs_id in range(positions.shape[1]):
        valid = alive[:, obs_id]
        if valid.sum() < 2:
            continue
        track = positions[valid, obs_id]
        movement += np.abs(np.diff(track, axis=0)).sum()
    assert movement > 0.0


def test_trajectory_alive_mask_is_monotone(blobs):
    model = pyodm.ODM(random_state=0, track_history=True).fit(blobs)
    _, _, _, alive = model.trajectories(n_frames=25)
    # once an observer is born it never disappears
    assert np.all(alive[1:] >= alive[:-1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_check_random_state_helper():
    assert check_random_state(3) is not None
    with pytest.raises(ValueError):
        check_random_state("seed")
