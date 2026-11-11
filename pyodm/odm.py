# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Name        : odm.py
# Description : Observers-based Data Modeling algorithm
# Author      : Fares Meghdouri
#
# Notes       : The private dSalmon M-Tree has been replaced by the exact,
#               self-contained indexes in :mod:`pyodm.trees`.
#
# Change-log  : 28-10-2019: //FM// Uploaded v1 with a KD-tree core.
#               06-02-2020: //FM// Rewrote the core v2 (M-tree as described
#                           in the paper).
#               08-06-2020: //FM// New features: number of observers, renamed
#                           parameters, additional parameters.
#               11-11-2026: Bug fixes and self-contained kNN cores (see
#                           BUG_FIXES.md); observer-trajectory tracking.
#
#******************************************************************************

from __future__ import annotations

import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist

from ._utils import as_array2d, check_random_state, rng_choice, rng_permutation
from .trees import SUPPORTED_CORES, make_index

__all__ = ["ODM", "EVENT_MOVE", "EVENT_BIRTH", "EVENT_RADIUS"]

#: Event kinds recorded by the trajectory tracker.
EVENT_MOVE = 0
EVENT_BIRTH = 1
EVENT_RADIUS = 2

_VALID_MODES = ("median", "mean", "max", "min", "sum")
_VALID_DISTANCE_ESTIMATION = ("adtm", "aid")


def _cpu_count():
    """Number of usable CPUs (``multiprocessing.cpu_count`` can return None)."""
    count = multiprocessing.cpu_count()
    return int(count) if count else 1


def _process_batch(batch, default_radius, factor, alpha, metric, core, track, start_step):
    """Run ODM on a single batch.

    Returns
    -------
    observers : ndarray, shape (n_created, n_features)
    population : ndarray, shape (n_created,)
    radius : ndarray, shape (n_created,)
    history : dict or None
        Compact event log: for every processed sample that changed the model,
        one entry per affected observer.  ``None`` when ``track`` is False.
    """
    batch = np.asarray(batch, dtype=np.float64)
    n_samples, n_features = batch.shape
    if n_samples == 0:
        return (
            np.empty((0, n_features), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            None,
        )

    observers = np.empty((n_samples, n_features), dtype=np.float64)
    population = np.zeros(n_samples, dtype=np.int64)
    radius = np.full(n_samples, float(default_radius), dtype=np.float64)

    index = make_index(core=core, dim=n_features, metric=metric)

    # The first sample always creates the first observer.
    observers[0] = batch[0]
    population[0] = 1
    index.insert(observers[0], 0)
    n_observers = 1

    if track:
        step_log = [start_step]
        obs_log = [0]
        kind_log = [EVENT_BIRTH]
        position_log = [observers[0].copy()]
        radius_log = [float(radius[0])]
        population_log = [int(population[0])]
    else:
        step_log = obs_log = kind_log = position_log = radius_log = population_log = None

    for t in range(1, n_samples):
        data_point = batch[t]
        nearest_ids, nearest_dist = index.query(data_point, 1)
        j = int(nearest_ids[0])
        dist = float(nearest_dist[0])

        if dist <= radius[j]:
            # Assigned: move the observer (scaled incremental mean) and shrink
            # its radius proportionally to how close the point landed.
            shift = alpha * (data_point - observers[j]) / (population[j] + 1)
            observers[j] += shift
            population[j] += 1
            radius[j] = max(0.0, radius[j] - factor * dist)
            index.update(j, observers[j])
            if track:
                step_log.append(start_step + t)
                obs_log.append(j)
                kind_log.append(EVENT_MOVE)
                position_log.append(observers[j].copy())
                radius_log.append(float(radius[j]))
                population_log.append(int(population[j]))
        else:
            # Rejected: the nearest observer's radius grows, and a new
            # observer is born on the data point.
            radius[j] = max(0.0, radius[j] + factor * dist)
            if track:
                step_log.append(start_step + t)
                obs_log.append(j)
                kind_log.append(EVENT_RADIUS)
                position_log.append(observers[j].copy())
                radius_log.append(float(radius[j]))
                population_log.append(int(population[j]))

            observers[n_observers] = data_point
            population[n_observers] = 1
            index.insert(data_point, n_observers)
            if track:
                step_log.append(start_step + t)
                obs_log.append(n_observers)
                kind_log.append(EVENT_BIRTH)
                position_log.append(data_point.copy())
                radius_log.append(float(radius[n_observers]))
                population_log.append(int(population[n_observers]))
            n_observers += 1

    history = None
    if track:
        history = {
            "step": np.asarray(step_log, dtype=np.int64),
            "obs": np.asarray(obs_log, dtype=np.int64),
            "kind": np.asarray(kind_log, dtype=np.int8),
            "position": np.vstack(position_log),
            "radius": np.asarray(radius_log, dtype=np.float64),
            "population": np.asarray(population_log, dtype=np.int64),
            "n_observers": int(n_observers),
            "start_step": int(start_step),
            "n_samples": int(n_samples),
        }

    return (
        observers[:n_observers].copy(),
        population[:n_observers].copy(),
        radius[:n_observers].copy(),
        history,
    )


class ODM(object):
    """Observers-based Data Modeling (ODM).

    ODM builds a low-density model of a data set: a data point that falls
    inside the radius of an existing *observer* moves that observer towards
    it, otherwise a new observer is created on the point.  The final set of
    observers is a coreset that keeps the internal structure of the data.

    Parameters
    ----------
    factor : float, optional (default=0.1)
        Factor by which an observer's radius is shrunk (assigned point) or
        grown (rejected point).

    R : float, optional
        Initial radius of new observers.  If None, it is estimated from a
        random sample of the data and scaled with ``beta``.

    rho : float, optional (default=0.05)
        Fraction of the data used to estimate ``R`` when ``R`` is None.

    m : int, optional
        Number of observers to keep (the ``m`` most populated ones).  If None,
        every observer created during the scan is kept.

    alpha : float, optional (default=1)
        Scaling factor of the observer position shift.  ``alpha=1`` is the
        plain incremental mean.

    beta : float, optional (default=1)
        Scale factor applied to the estimated radius ``R``.

    core : str, optional (default='m-tree')
        kNN index used to find the closest observer: ``'m-tree'`` (dynamic
        metric tree), ``'kd-tree'`` (scipy cKDTree with amortised rebuilds) or
        ``'brute'`` (exact linear scan).

    distance_estimation : str, optional (default='adtm')
        How ``R`` is estimated: ``'adtm'`` (average distance to the mean) or
        ``'aid'`` (average inter-distance between sampled points).

    metric : str, optional (default='euclidean')
        Distance metric: ``'euclidean'``, ``'manhattan'`` or ``'chebyshev'``.

    parallelize : bool, optional (default=False)
        Estimate ``R`` per chunk and process the chunks in parallel.  Chunked
        runs create more observers, because every chunk starts its own model.

    chunksize : int, optional
        Number of samples processed per chunk.  None processes the whole data
        set in a single chunk.

    random_state : None, int, RandomState or Generator, optional
        Seed used for shuffling and radius estimation.

    shuffle_data : bool, optional (default=True)
        Shuffle the data before processing it.

    n_cores : int, optional (default=1)
        Number of workers used when ``parallelize=True``.

    O : int, optional (default=5)
        Number of nearest observers used to compute outlierness scores.

    mode : str, optional (default='median')
        Aggregation over the ``O`` nearest-observer distances: ``'median'``,
        ``'mean'``, ``'max'``, ``'min'`` or ``'sum'``.

    track_history : bool, optional (default=False)
        Record every observer position/radius change so trajectories can be
        replayed (see :meth:`trajectories`).  Costs ``O(n_samples)`` memory.

    verbose : int, optional (default=0)
        ``0`` prints nothing, ``1`` prints model information.

    Attributes
    ----------
    observers : ndarray, shape (n_observers, n_features)
        Observer locations (the coreset).
    population : ndarray, shape (n_observers,)
        Number of samples assigned to each observer.
    radius : ndarray, shape (n_observers,)
        Final radius of each observer.
    R_ : float
        Radius actually used for new observers.
    history_ : dict or None
        Event log used by :meth:`trajectories` (only when ``track_history``).
    data_ : ndarray or None
        The data in processing order (only stored when ``track_history``).
    n_features_ : int
        Dimensionality of the fitted data.
    n_samples_ : int
        Number of fitted samples.
    """

    def __init__(
        self,
        factor=0.1,
        R=None,
        rho=0.05,
        m=None,
        alpha=1.0,
        beta=1.0,
        core="m-tree",
        distance_estimation="adtm",
        metric="euclidean",
        parallelize=False,
        chunksize=None,
        random_state=None,
        shuffle_data=True,
        n_cores=1,
        O=5,
        mode="median",
        track_history=False,
        verbose=0,
    ):
        if not float(factor) >= 0:
            raise ValueError("factor must be >= 0, got {!r}".format(factor))
        if not 0 < float(rho) <= 1:
            raise ValueError("rho must be within (0, 1], got {!r}".format(rho))
        if R is not None and not float(R) > 0:
            raise ValueError("R must be positive or None, got {!r}".format(R))
        if m is not None and int(m) < 1:
            raise ValueError("m must be >= 1 or None, got {!r}".format(m))
        if not float(alpha) > 0:
            raise ValueError("alpha must be positive, got {!r}".format(alpha))
        if not float(beta) > 0:
            raise ValueError("beta must be positive, got {!r}".format(beta))
        if chunksize is not None and int(chunksize) < 1:
            raise ValueError("chunksize must be >= 1 or None, got {!r}".format(chunksize))
        if int(n_cores) < 1:
            raise ValueError("n_cores must be >= 1, got {!r}".format(n_cores))
        if int(O) < 1:
            raise ValueError("O must be >= 1, got {!r}".format(O))
        if mode not in _VALID_MODES:
            raise ValueError("mode must be one of {}, got {!r}".format(_VALID_MODES, mode))
        distance_estimation = str(distance_estimation).strip().lower()
        if distance_estimation not in _VALID_DISTANCE_ESTIMATION:
            raise ValueError(
                "distance_estimation must be one of {}, got {!r}".format(
                    _VALID_DISTANCE_ESTIMATION, distance_estimation
                )
            )
        core = str(core).strip().lower()
        if core not in SUPPORTED_CORES:
            raise ValueError("core must be one of {}, got {!r}".format(SUPPORTED_CORES, core))

        self.factor = float(factor)
        self.R = None if R is None else float(R)
        self.rho = float(rho)
        self.m = None if m is None else int(m)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.core = core
        self.distance_estimation = distance_estimation
        self.metric = metric
        self.parallelize = bool(parallelize)
        self.chunksize = None if chunksize is None else int(chunksize)
        self.random_state = random_state
        self.shuffle_data = bool(shuffle_data)
        self.n_cores = int(n_cores)
        self.O = int(O)
        self.mode = mode
        self.track_history = bool(track_history)
        self.verbose = int(verbose)

        # Fitted attributes are None until fit() is called.
        self.observers = None
        self.population = None
        self.radius = None
        self.R_ = None
        self.history_ = None
        self.data_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.order_ = None
        self.selected_ = None
        self._fitted = False

    # ------------------------------------------------------------------
    # parameters / helpers
    # ------------------------------------------------------------------
    @property
    def estim_R(self):
        """True when the initial radius is estimated from the data."""
        return self.R is None

    def _log(self, level, message, *args):
        if self.verbose >= level:
            print(message.format(*args))

    def get_params(self, deep=True):
        """Return the model parameters as a dictionary."""
        return {
            "factor": self.factor,
            "R": self.R,
            "rho": self.rho,
            "m": self.m,
            "alpha": self.alpha,
            "beta": self.beta,
            "core": self.core,
            "distance_estimation": self.distance_estimation,
            "metric": self.metric,
            "parallelize": self.parallelize,
            "chunksize": self.chunksize,
            "random_state": self.random_state,
            "shuffle_data": self.shuffle_data,
            "n_cores": self.n_cores,
            "O": self.O,
            "mode": self.mode,
            "track_history": self.track_history,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """Set parameters (sklearn-style) and return ``self``."""
        known = self.get_params()
        for key, value in params.items():
            if key not in known:
                raise ValueError("unknown parameter {!r}".format(key))
            setattr(self, key, value)
        return self

    def _rng(self):
        return check_random_state(self.random_state)

    def _shuffle(self, X, labels=None):
        """Shuffle ``X`` (and ``labels`` together when given).

        Unlike the previous implementation, ``labels`` is handled correctly:
        ``bool(numpy_array)`` raised ``ValueError`` and the conditional
        expression returned the wrong object.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        permutation = rng_permutation(self._rng(), X.shape[0])
        if labels is None:
            return X[permutation]
        labels = np.asarray(labels)
        if labels.shape[0] != X.shape[0]:
            raise ValueError("labels and X must have the same number of rows")
        return X[permutation], labels[permutation]

    def _estimate_R(self, X, method=None):
        """Estimate the initial radius from a random subset of ``X``."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_points = int(round(n_samples * self.rho))
        n_points = min(max(n_points, 3), n_samples)
        data = X[rng_choice(self._rng(), n_samples, size=n_points, replace=False)]
        method = method or self.distance_estimation

        if method == "aid":
            return float(np.mean(pdist(data)))
        centroid = data.mean(axis=0, keepdims=True)
        return float(np.mean(np.sqrt(((data - centroid) ** 2).sum(axis=1))))

    # Names kept for backwards compatibility.
    def _estimate_R_average_distance(self, X):
        return self._estimate_R(X, "adtm")

    def _estimate_R_inter_distances(self, X):
        return self._estimate_R(X, "aid")

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, X):
        """Build the ODM model (coreset) from ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : ODM
        """
        X = as_array2d(X)
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # A refit must not accumulate state from the previous run.
        self.history_ = None
        self.data_ = None
        self.order_ = None
        self.selected_ = None

        if self.shuffle_data:
            X = self._shuffle(X)

        if not self.estim_R:
            self.R_ = float(self.R)
        elif not self.parallelize:
            self.R_ = float(self.beta) * self._estimate_R(X)
        else:
            self.R_ = 0.0  # estimated per chunk below

        if self.n_cores > 1:
            available = _cpu_count()
            if self.n_cores > available:
                self._log(0, "WARNING: only {0} cores available", available)
                self.n_cores = available

        # ``np.array_split`` requires an integer number of sections; the
        # original code passed ``n_samples/chunksize`` (a float), which raised
        # ValueError whenever the chunk was larger than the data set.
        chunk_size = self.chunksize or n_samples
        n_chunks = max(1, int(np.ceil(n_samples / float(chunk_size))))
        batches = list(np.array_split(X, n_chunks))
        starts = np.cumsum([0] + [batch.shape[0] for batch in batches[:-1]])

        radii = []
        for batch, offset in zip(batches, starts):
            if self.estim_R and self.parallelize:
                radii.append(self.beta * self._estimate_R(batch))
            else:
                radii.append(self.R_)

        results = Parallel(n_jobs=self.n_cores, backend="loky")(
            delayed(_process_batch)(
                batch, radius, self.factor, self.alpha, self.metric, self.core,
                self.track_history, int(offset),
            )
            for batch, radius, offset in zip(batches, radii, starts)
        )

        observers, population, radius = [], [], []
        history_parts = []
        observer_offset = 0
        for batch, (batch_observers, batch_population, batch_radius, history) in zip(batches, results):
            if batch_observers.shape[0] == 0:  # pragma: no cover - impossible
                continue
            observers.append(batch_observers)
            population.append(batch_population)
            radius.append(batch_radius)
            if history is not None:
                history = dict(history)
                history["obs"] = history["obs"] + observer_offset
                history["n_observers"] = observer_offset + int(history["n_observers"])
                history_parts.append(history)
            observer_offset += batch_observers.shape[0]

        self.observers = np.vstack(observers)
        self.population = np.concatenate(population)
        self.radius = np.concatenate(radius)

        if self.track_history:
            self.history_ = {
                key: np.concatenate([part[key] for part in history_parts])
                for key in ("step", "obs", "kind", "position", "radius", "population")
            }
            self.history_["n_observers"] = int(observer_offset)
            self.history_["batch_starts"] = np.asarray(starts, dtype=np.int64)
            self.history_["batch_sizes"] = np.asarray([b.shape[0] for b in batches], dtype=np.int64)
            self.data_ = X

        self._fitted = True
        self._select_observers()

        self._log(
            1,
            "INFO: model built with {} observers (R = {:.4f})",
            self.observers.shape[0],
            self.R_,
        )
        return self

    def _select_observers(self):
        """Keep the ``m`` most populated observers (all of them when m is None)."""
        available = int(self.observers.shape[0])
        if self.m is None:
            self.order_ = np.arange(available)
            self.selected_ = None
            return

        if self.m > available:
            # The original implementation appended random data points to
            # ``observers`` without extending ``population``/``radius``, which
            # made any later indexing crash.  Keeping all observers is safe.
            self.order_ = np.arange(available)
            self.selected_ = None
            self._log(
                1,
                "WARNING: only {} observers were created, keeping all of them (m={})",
                available,
                self.m,
            )
            return

        selected = np.sort(np.argsort(self.population, kind="stable")[-self.m:])
        self.order_ = selected
        # ``np.take`` builds fresh arrays instead of (possibly non-contiguous)
        # views, so later code can rely on the fitted attributes being owned.
        self.observers = np.take(self.observers, selected, axis=0)
        self.population = np.take(self.population, selected)
        self.radius = np.take(self.radius, selected)
        # Trajectories refer to the full event log (ids are global), so the
        # mapping is stored separately rather than mutating the log.
        self.selected_ = selected

    # Backwards-compatible worker entry points (the old private API).
    def _doWork_mtrees(self, batch):  # pragma: no cover - compatibility shim
        return _process_batch(
            batch,
            self.R_ if self.R_ is not None else 0.0,
            self.factor,
            self.alpha,
            self.metric,
            self.core,
            False,
            0,
        )[:3]

    def _doWork_kdtrees(self, batch):  # pragma: no cover - compatibility shim
        return self._doWork_mtrees(batch)

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("this ODM instance is not fitted yet; call fit(X) first")

    def _observer_index(self):
        index = make_index(core=self.core, dim=int(self.n_features_), metric=self.metric)
        if hasattr(index, "bulk_load"):
            index.bulk_load(self.observers, np.arange(self.observers.shape[0], dtype=np.int64))
        else:
            for obs_id, point in enumerate(self.observers):
                index.insert(point, obs_id)
        return index

    def _query_distances(self, X, O):
        """Distances from every row of ``X`` to its ``O`` nearest observers."""
        X = as_array2d(X, name="X")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "X has {} features but the model was fitted with {}".format(
                    X.shape[1], self.n_features_
                )
            )
        if self.observers is None or self.observers.shape[0] == 0:  # pragma: no cover
            raise RuntimeError("the model has no observers")
        requested = self.observers.shape[0] if O is None else int(O)
        if requested < 1:
            raise ValueError("O must be >= 1, got {!r}".format(O))
        k = min(requested, self.observers.shape[0])

        index = self._observer_index()
        query_many = getattr(index, "query_many", None)
        if query_many is not None:
            return query_many(X, k=k)

        ids = np.empty((X.shape[0], k), dtype=np.int64)
        dists = np.empty((X.shape[0], k), dtype=np.float64)
        for row, point in enumerate(X):
            neighbour_ids, neighbour_dists = index.query(point, k)
            ids[row, : neighbour_ids.size] = neighbour_ids
            dists[row, : neighbour_dists.size] = neighbour_dists
        return ids, dists

    @staticmethod
    def _aggregate(dists, mode):
        if mode == "mean":
            return np.mean(dists, axis=1)
        if mode == "max":
            return np.max(dists, axis=1)
        if mode == "min":
            return np.min(dists, axis=1)
        if mode == "sum":
            return np.sum(dists, axis=1)
        return np.median(dists, axis=1)

    def outlierness(self, X, O=None, mode=None, chunksize=None):
        """Outlierness score of every row of ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        O : int, optional
            Number of nearest observers to use; defaults to the model's ``O``.
        mode : str, optional
            Aggregation over the ``O`` distances; defaults to the model's
            ``mode``.
        chunksize : int, optional
            Split the data into chunks of this size.  It affects memory and
            parallelism only, never the result.

        Returns
        -------
        scores : ndarray, shape (n_samples,)
        """
        self._check_fitted()
        X = as_array2d(X, name="X")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "X has {} features but the model was fitted with {}".format(
                    X.shape[1], self.n_features_
                )
            )
        O = self.O if O is None else int(O)
        mode = self.mode if mode is None else mode
        if mode not in _VALID_MODES:
            raise ValueError("mode must be one of {}, got {!r}".format(_VALID_MODES, mode))

        chunk_size = 1000 if chunksize is None else int(chunksize)
        if chunk_size < 1:
            raise ValueError("chunksize must be >= 1, got {!r}".format(chunksize))
        # Same float-sections bug as in fit(): use ceil(n / chunksize).
        n_chunks = max(1, int(np.ceil(X.shape[0] / float(chunk_size))))
        parts = Parallel(n_jobs=self.n_cores, backend="loky")(
            delayed(self._score_batch)(chunk, O, mode)
            for chunk in np.array_split(X, n_chunks)
            if chunk.shape[0] > 0
        )
        return np.concatenate(parts)

    def _score_batch(self, batch, O, mode):
        _, dists = self._query_distances(batch, O)
        return self._aggregate(dists, mode)

    def predict(self, X, contamination=0.1, O=None, mode=None, chunksize=None):
        """Binary outlier labels from a contamination threshold.

        Returns ``True`` for the ``contamination`` fraction of the points with
        the highest outlierness.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        contamination : float, optional (default=0.1)
            Expected ratio of outliers, ``0 < contamination <= 1``.
        O, mode, chunksize
            See :meth:`outlierness`.
        """
        if not 0 < float(contamination) <= 1:
            raise ValueError("contamination must be within (0, 1]")
        scores = self.outlierness(X, O=O, mode=mode, chunksize=chunksize)
        threshold = np.quantile(scores, 1 - float(contamination))
        return scores > threshold

    def decision_function(self, X, **kwargs):
        """Alias of :meth:`outlierness` (higher means more outlying)."""
        return self.outlierness(X, **kwargs)

    def labels(self, X):
        """Id of the closest observer for every row of ``X``.

        The returned ids index the rows of ``model.observers``.
        """
        self._check_fitted()
        ids, _ = self._query_distances(X, 1)
        return ids[:, 0]

    # ------------------------------------------------------------------
    # trajectories (for animations)
    # ------------------------------------------------------------------
    def trajectories(self, n_frames=None, max_frames=600):
        """Replay the tracking log into per-frame observer positions.

        Parameters
        ----------
        n_frames : int, optional
            Number of frames to sample from the log.  Defaults to
            ``min(n_samples + 1, max_frames)``.
        max_frames : int, optional (default=600)

        Returns
        -------
        frames : ndarray, shape (n_frames,)
            Sample index of every frame (state *after* processing them).
        positions : ndarray, shape (n_frames, n_observers, n_features)
            Observer positions; ``NaN`` where an observer does not exist yet.
        radii : ndarray, shape (n_frames, n_observers)
            Observer radius at every frame.
        alive : ndarray of bool, shape (n_frames, n_observers)
            Whether the observer exists at that frame.
        """
        self._check_fitted()
        if self.history_ is None:
            raise RuntimeError(
                "no trajectory was recorded; construct ODM(track_history=True) and fit again"
            )
        total_steps = int(self.n_samples_)
        if n_frames is None:
            n_frames = min(total_steps + 1, int(max_frames))
        frames = np.unique(
            np.linspace(0, total_steps, int(n_frames), endpoint=True, dtype=np.int64)
        )

        history = self.history_
        total_observers = int(history["n_observers"])
        dim = int(self.n_features_)
        positions = np.full((frames.size, total_observers, dim), np.nan, dtype=np.float64)
        radii = np.full((frames.size, total_observers), np.nan, dtype=np.float64)
        alive = np.zeros((frames.size, total_observers), dtype=bool)
        filled = np.zeros(total_observers, dtype=bool)

        order = np.argsort(history["step"], kind="stable")
        cursor = 0
        n_events = order.size
        pos_state = np.full((total_observers, dim), np.nan, dtype=np.float64)
        rad_state = np.full(total_observers, np.nan, dtype=np.float64)
        for frame_index, frame_step in enumerate(frames):
            while cursor < n_events and history["step"][order[cursor]] < frame_step:
                event = order[cursor]
                obs_id = int(history["obs"][event])
                pos_state[obs_id] = history["position"][event]
                rad_state[obs_id] = history["radius"][event]
                filled[obs_id] = True
                cursor += 1
            positions[frame_index] = pos_state
            radii[frame_index] = rad_state
            alive[frame_index] = filled
        # Restrict the output to the observers the model actually kept (``m``
        # may have dropped the least populated ones).
        if self.selected_ is not None:
            positions = positions[:, self.selected_, :]
            radii = radii[:, self.selected_]
            alive = alive[:, self.selected_]
        return frames, positions, radii, alive

    def __repr__(self):  # pragma: no cover - cosmetic
        state = "fitted" if self._fitted else "not fitted"
        return "ODM(core={!r}, factor={}, m={}, {} observers, {})".format(
            self.core,
            self.factor,
            self.m,
            None if self.observers is None else self.observers.shape[0],
            state,
        )
