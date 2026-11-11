"""Artificial 2-D data sets used for the observer-movement animations.

Each generator returns ``(X, y, meta)`` where

* ``X``    -- ``(n_samples, 2)`` float array of data points,
* ``y``    -- ``(n_samples,)`` int array with the ground-truth cluster label,
* ``meta`` -- dictionary with a human readable title/description and the
  parameters that produced the data set (stored next to the animation).

The data sets are deliberately varied: separated blobs, blobs with very
different densities, concentric rings (non-convex clusters), anisotropic
clusters that overlap, a chain/crescent shape, and a uniform background with
a few dense islands.  They all have *many* clusters so the movement of the
observers is easy to follow.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DATASETS", "generate", "list_datasets"]


def _rng(seed):
    return np.random.RandomState(seed)


# ---------------------------------------------------------------------------
# individual generators
# ---------------------------------------------------------------------------
def blobs(seed=0, n_per_cluster=250):
    """Eight well separated Gaussian blobs on a circle."""
    rng = _rng(seed)
    n_clusters = 8
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centres = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
    parts, labels = [], []
    for index, centre in enumerate(centres):
        spread = rng.uniform(0.5, 1.2)
        parts.append(rng.normal(centre, spread, size=(n_per_cluster, 2)))
        labels.append(np.full(n_per_cluster, index))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    meta = {
        "name": "eight_blobs",
        "title": "Eight Gaussian blobs",
        "description": "8 well separated blobs, each with its own spread ({:.1f} points per blob).".format(
            n_per_cluster
        ),
        "n_clusters": n_clusters,
    }
    return X, y, meta


def variable_density(seed=1):
    """Six blobs with sizes and spreads differing by an order of magnitude."""
    rng = _rng(seed)
    spec = [
        ((0.0, 0.0), 0.25, 220),
        ((4.0, 2.0), 1.60, 900),
        ((9.0, -1.0), 0.60, 400),
        ((2.0, 9.5), 1.10, 700),
        ((12.0, 8.0), 0.35, 260),
        ((7.0, 13.0), 1.90, 1100),
    ]
    parts, labels = [], []
    for index, (centre, spread, count) in enumerate(spec):
        parts.append(rng.normal(centre, spread, size=(count, 2)))
        labels.append(np.full(count, index))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y, {
        "name": "variable_density",
        "title": "Clusters with very different density",
        "description": "6 clusters whose sizes range from 220 to 1100 points and whose spread "
        "differs by a factor of ~8.",
        "n_clusters": len(spec),
    }


def concentric_rings(seed=2):
    """Four concentric rings plus a dense core (non-convex clusters)."""
    rng = _rng(seed)
    parts, labels = [], []
    n_points = [200, 300, 400, 500]
    for index, (radius, count) in enumerate(zip([1.5, 3.5, 5.5, 7.5], n_points)):
        theta = rng.uniform(0, 2 * np.pi, count)
        noise = rng.normal(0, 0.12, count)
        radius_jitter = radius + noise
        parts.append(
            np.column_stack([radius_jitter * np.cos(theta), radius_jitter * np.sin(theta)])
        )
        labels.append(np.full(count, index))
    theta = rng.uniform(0, 2 * np.pi, 300)
    core = np.column_stack([0.4 * rng.uniform(0, 1, 300) * np.cos(theta),
                            0.4 * rng.uniform(0, 1, 300) * np.sin(theta)])
    parts.append(core)
    labels.append(np.full(300, len(n_points)))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y, {
        "name": "concentric_rings",
        "title": "Concentric rings",
        "description": "4 concentric rings plus a dense core: non-convex clusters that Euclidean "
        "k-means-like methods handle badly.",
        "n_clusters": len(n_points) + 1,
    }


def anisotropic_overlapping(seed=3):
    """Five elongated, rotated clusters that partly overlap."""
    rng = _rng(seed)
    centres = [(0, 0), (7, 1), (3, 7), (10, 8), (6, 13)]
    parts, labels = [], []
    for index, centre in enumerate(centres):
        angle = np.deg2rad(rng.uniform(0, 180))
        cov = np.array([[2.2, 0.0], [0.0, 0.35]])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = rng.multivariate_normal((0, 0), cov, size=350)
        points = points @ rotation.T + np.asarray(centre)
        parts.append(points)
        labels.append(np.full(350, index))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y, {
        "name": "anisotropic_overlapping",
        "title": "Elongated overlapping clusters",
        "description": "5 anisotropic (elongated, rotated) clusters that partly overlap.",
        "n_clusters": len(centres),
    }


def crescents(seed=4):
    """Six half-moon clusters arranged in a grid plus a spiral arm count."""
    rng = _rng(seed)
    parts, labels = [], []
    for index, (cx, cy) in enumerate([(0, 0), (7, 0), (14, 0), (0, 8), (7, 8), (14, 8)]):
        # a half moon built from an arc with radial noise
        theta = rng.uniform(np.pi * 0.15, np.pi * 0.85, 320)
        r = 2.0 + rng.normal(0, 0.18, theta.size)
        arc = np.column_stack([r * np.cos(theta), r * np.sin(theta) - 1.0])
        blob = rng.normal((0.0, -1.8), 0.45, size=(180, 2))
        points = np.vstack([arc, blob]) + np.asarray((cx, cy))
        parts.append(points)
        labels.append(np.full(points.shape[0], index))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y, {
        "name": "crescents",
        "title": "Six crescent-shaped clusters",
        "description": "6 crescent (half-moon) clusters with a small satellite blob each.",
        "n_clusters": 6,
    }


def islands_in_noise(seed=5):
    """Five dense islands embedded in a large low-density background."""
    rng = _rng(seed)
    background = rng.uniform(-2, 18, size=(1200, 2))
    parts = [background]
    labels = [np.full(background.shape[0], -1)]
    centres = [(2, 2), (10, 3), (16, 9), (5, 13), (12, 15)]
    for index, centre in enumerate(centres):
        points = rng.normal(centre, 0.55, size=(280, 2))
        parts.append(points)
        labels.append(np.full(points.shape[0], index))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y, {
        "name": "islands_in_noise",
        "title": "Dense islands in uniform noise",
        "description": "5 dense islands plus 1200 uniformly distributed background points "
        "(label -1). The background is what ODM turns into a diffuse web of observers.",
        "n_clusters": len(centres),
    }


def spiral(seed=6):
    """Three interleaved spiral arms (a classic hard clustering benchmark)."""
    rng = _rng(seed)
    n_points = 650
    parts, labels = [], []
    for arm in range(3):
        t = np.linspace(0.25, 4 * np.pi, n_points)
        x = t * np.cos(t + arm * 2 * np.pi / 3) + rng.normal(0, 0.12, t.size)
        y = t * np.sin(t + arm * 2 * np.pi / 3) + rng.normal(0, 0.12, t.size)
        points = np.column_stack([x, y]) * 0.75
        parts.append(points)
        labels.append(np.full(n_points, arm))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y, {
        "name": "spiral_arms",
        "title": "Three interleaved spiral arms",
        "description": "3 interleaved Archimedean spirals: clusters that are highly non-convex.",
        "n_clusters": 3,
    }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
DATASETS = {
    "eight_blobs": blobs,
    "variable_density": variable_density,
    "concentric_rings": concentric_rings,
    "anisotropic_overlapping": anisotropic_overlapping,
    "crescents": crescents,
    "islands_in_noise": islands_in_noise,
    "spiral_arms": spiral,
}


def list_datasets():
    """Return the names of all available synthetic data sets."""
    return list(DATASETS)


def generate(name, seed=None):
    """Generate one of the data sets by name.

    Parameters
    ----------
    name : str
        One of :func:`list_datasets`.
    seed : int, optional
        Overrides the default seed of that data set.
    """
    if name not in DATASETS:
        raise KeyError("unknown data set {!r}; choose from {}".format(name, list(DATASETS)))
    generator = DATASETS[name]
    if seed is None:
        return generator()
    return generator(seed=seed)
