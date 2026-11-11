# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Small validation / RNG helpers shared by pyodm modules.
#
# ****************************************************************************

"""Internal helpers (no public API guarantees).

The original implementation reseeded the global numpy RNG with
``np.random.seed(self.random_state)`` in several places.  That has two
problems: it makes pyodm affect unrelated code, and it raises ``TypeError``
when the user passes an actual ``RandomState`` (which the documentation
explicitly allows).  The helpers below implement proper seeding and support
both ``RandomState`` and the newer ``Generator`` API.
"""

from __future__ import annotations

import numbers

import numpy as np

__all__ = [
    "check_random_state",
    "rng_choice",
    "rng_permutation",
    "rng_integers",
    "as_array2d",
]


def check_random_state(seed):
    """Turn ``seed`` into a random number generator instance.

    Parameters
    ----------
    seed : None, int, ``numpy.random.RandomState`` or ``numpy.random.Generator``
        ``None`` uses numpy's global ``RandomState``; an integer creates a new
        independent ``RandomState``.

    Returns
    -------
    numpy.random.RandomState or numpy.random.Generator
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(int(seed))
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    raise ValueError(
        "{!r} cannot be used as a random seed; use None, an int, a RandomState or a Generator".format(
            seed
        )
    )


def rng_integers(rng, low, high, size=None):
    """``randint``/``integers`` across both numpy generator APIs."""
    if isinstance(rng, np.random.Generator):
        return rng.integers(low, high, size=size)
    return rng.randint(low, high, size=size)


def rng_choice(rng, a, size=None, replace=True):
    """``RandomState.choice`` / ``Generator.choice`` wrapper."""
    return rng.choice(a, size=size, replace=replace)


def rng_permutation(rng, n):
    """Permutation of ``range(n)`` for both generator APIs."""
    if isinstance(rng, np.random.Generator):
        return rng.permutation(n)
    return rng.permutation(n)


def as_array2d(X, name="X", dtype=np.float64):
    """Validate a data matrix and return a contiguous array.

    Raises a ``ValueError`` when ``X`` is empty, not 2-D, or contains
    non-finite values.  The original code accepted such inputs silently and
    failed much later with confusing errors.
    """
    try:
        array = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a numeric array: {}".format(name, exc))
    if array.ndim != 2:
        raise ValueError(
            "{} must be 2-D (n_samples, n_features), got shape {}".format(name, array.shape)
        )
    if array.shape[0] == 0:
        raise ValueError("{} is empty (0 samples)".format(name))
    if array.shape[1] == 0:
        raise ValueError("{} has 0 features".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinite values".format(name))
    return np.ascontiguousarray(array)
