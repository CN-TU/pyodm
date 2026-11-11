# -*- coding: utf-8 -*-

# ****************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Render observer-movement animations for synthetic data sets.
#
# ****************************************************************************

"""Create one animation per synthetic data set showing how ODM observers move.

For every data set the script

1. generates the 2-D data (:mod:`examples.datasets`),
2. fits an ODM model with ``track_history=True``,
3. replays the recorded event log (:meth:`pyodm.ODM.trajectories`) into
   per-frame observer positions / radii,
4. draws one frame per sampled iteration and writes it with matplotlib
   (Pillow writer -> GIF, ffmpeg writer -> MP4 when available).

Outputs (default ``animations/``):

* ``<dataset>.gif``  -- the animation (README-friendly),
* ``<dataset>.mp4``  -- the same animation (only if ffmpeg is installed),
* ``<dataset>.png``  -- static overview: data, start markers, final observers
  and the (straightened) trajectories,
* ``<dataset>_data.npy`` / ``<dataset>_labels.npy`` -- the generated data,
* ``<dataset>.json`` -- parameters and run statistics.

Usage
-----
    python examples/animate_observers.py                 # every data set
    python examples/animate_observers.py eight_blobs     # a single one
    python examples/animate_observers.py --list          # names only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

# Make the package importable when the script is run from a checkout.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
from matplotlib import animation

import pyodm

try:  # the data generators live next to this script
    from datasets import generate, list_datasets
except ImportError:  # pragma: no cover - only when run from another directory
    sys.path.insert(0, HERE)
    from datasets import generate, list_datasets

#: Number of animation frames (sampled iterations).
DEFAULT_N_FRAMES = 120
#: Playback speed.
DEFAULT_FPS = 14
#: Initial radius as a fraction of the data extent (see ``pick_parameters``).
RADIUS_FRACTION = 0.13
#: Radius adaptation factor used for the animations.
RADIUS_FACTOR = 0.01
#: Colour-blind friendly palette; observers cycle through it.
OBSERVER_COLOURS = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080",
    "#f032e6", "#9a6324", "#800000", "#000075", "#2f9e44", "#c92a2a",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pretty(name):
    """Turn ``eight_blobs`` into ``Eight Blobs``."""
    return name.replace("_", " ").title()


def observer_colour(obs_id):
    return OBSERVER_COLOURS[int(obs_id) % len(OBSERVER_COLOURS)]


def pick_parameters(X):
    """Choose ODM parameters giving an interpretable, low-density model.

    The initial radius is a fixed fraction of the data extent.  This produces
    a handful of observers per cluster (a few dozen in total) instead of the
    hundreds implied by the nearest-neighbour based estimation, so the motion
    of each observer is visible in the animation.
    """
    extent = float(np.max(np.ptp(X, axis=0)))
    return {
        "R": round(RADIUS_FRACTION * extent, 3),
        "factor": RADIUS_FACTOR,
        "alpha": 1.0,
    }


def fit_model(X, params, seed=2020):
    """Fit ODM with history tracking on ``X``."""
    model = pyodm.ODM(
        R=params["R"],
        factor=params["factor"],
        alpha=params["alpha"],
        random_state=seed,
        shuffle_data=True,
        core="m-tree",
        track_history=True,
    )
    model.fit(X)
    return model


def trajectories_for(model, n_frames=DEFAULT_N_FRAMES):
    return model.trajectories(n_frames=n_frames)


def _trajectory_segments(positions, alive):
    """Yield ``(observer_id, start, end, n_frames_tracked)`` for alive observers."""
    for obs_id in range(positions.shape[1]):
        rows = np.where(alive[:, obs_id])[0]
        if rows.size < 2:
            continue
        yield obs_id, positions[rows[0], obs_id], positions[rows[-1], obs_id], rows.size


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def build_figure(X, model, frames, positions, radii, alive, meta):
    """Return ``(fig, update, init)`` for a matplotlib FuncAnimation."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.0, 7.2), dpi=110)
    fig.patch.set_facecolor("white")

    data_xy = model.data_[:, :2] if model.data_ is not None else X[:, :2]
    seen = ax.scatter(data_xy[:, 0], data_xy[:, 1], s=9, c="#9aa3ad", alpha=0.28,
                      linewidths=0, zorder=1)
    unseen = ax.scatter(data_xy[:, 0], data_xy[:, 1], s=9, c="#2b2f36", alpha=0.55,
                        linewidths=0, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=0.0)  # keeps the axes limits stable

    n_observers = int(positions.shape[1])
    colour_list = [observer_colour(i) for i in range(n_observers)]

    # start markers (hollow) and moving markers (X), drawn per frame
    start_scatter = ax.scatter(
        np.full(n_observers, np.nan), np.full(n_observers, np.nan), s=54,
        facecolors="none", edgecolors=colour_list, linewidths=1.3, zorder=3,
    )
    obs_scatter = ax.scatter(
        np.full(n_observers, np.nan), np.full(n_observers, np.nan), s=110,
        c=colour_list, marker="X", edgecolors="white", linewidths=1.1, zorder=6,
    )
    trails = [
        ax.plot([], [], color=colour_list[i], linewidth=2.0, alpha=0.95, zorder=5,
                solid_capstyle="round")[0]
        for i in range(n_observers)
    ]
    circles = []
    for obs_id in range(n_observers):
        circle = plt.Circle((0.0, 0.0), 0.0, facecolor=colour_list[obs_id], alpha=0.10,
                            linewidth=0.0, zorder=2, visible=False)
        ax.add_patch(circle)
        circles.append(circle)

    ax.set_title("{} - observer movement".format(pretty(meta["name"]), ), fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("#f6f7f9")
    ax.grid(color="white", linewidth=1.1)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="#2b2f36",
                       markeredgecolor="#2b2f36", label="data point"),
            plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="none",
                       markeredgecolor="#7a7f87", label="observer at first iteration"),
            plt.Line2D([], [], marker="X", linestyle="", markersize=9, markerfacecolor="#e6194b",
                       markeredgecolor="white", label="observer now"),
        ],
        loc="upper right", fontsize=9, framealpha=0.92,
    )
    info = ax.annotate(
        "", xy=(0.012, 0.985), xycoords="axes fraction", va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d0d4da", alpha=0.95),
    )
    total_steps = int(frames[-1]) if frames.size else 0
    pos_xy = positions[:, :, :2]

    def _start_markers(index):
        offsets = np.full((n_observers, 2), np.nan)
        for obs_id in range(n_observers):
            rows = np.where(alive[: index + 1, obs_id])[0]
            if rows.size:
                offsets[obs_id] = pos_xy[rows[0], obs_id]
        start_scatter.set_offsets(offsets)

    def _current_markers(index):
        offsets = pos_xy[index].copy()
        offsets[~alive[index]] = np.nan
        obs_scatter.set_offsets(offsets)

    def init():
        _start_markers(0)
        _current_markers(0)
        for ops in trails:
            ops.set_data([], [])
        for circle in circles:
            circle.set_visible(False)
        unseen.set_alpha(0.55)
        info.set_text("iteration 0/{}\nobservers: {}".format(
            total_steps, int(alive[0].sum())))
        return []

    def update(frame_index):
        index = int(frame_index)
        step = int(frames[index])
        # data points scanned so far become "seen"
        seen.set_offsets(data_xy[:step] if step else np.empty((0, 2)))
        unseen.set_offsets(data_xy[step:] if step else data_xy)

        for obs_id, ops in enumerate(trails):
            rows = np.where(alive[: index + 1, obs_id])[0]
            if rows.size < 2:
                ops.set_data([], [])
            else:
                ops.set_data(pos_xy[rows, obs_id, 0], pos_xy[rows, obs_id, 1])

        _start_markers(index)
        _current_markers(index)

        for obs_id, circle in enumerate(circles):
            if alive[index, obs_id]:
                circle.set_center(pos_xy[index, obs_id])
                circle.set_radius(float(radii[index, obs_id]))
                circle.set_visible(True)
            else:
                circle.set_visible(False)

        info.set_text(
            "iteration {}/{}\nobservers: {}   radius: {:.2f}".format(
                step, total_steps, int(alive[index].sum()), float(radii[index][alive[index]].mean())
                if alive[index].any() else 0.0
            )
        )
        return []

    return fig, update, init


def render_overview(X, model, frames, positions, radii, alive, meta):
    """Static overview: data, start markers, final observers and trajectories."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.0, 7.2), dpi=110)
    ax.scatter(X[:, 0], X[:, 1], s=9, c="#b7bdc7", linewidths=0, zorder=1)
    for obs_id, start, end, _ in _trajectory_segments(positions, alive):
        colour = observer_colour(obs_id)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=colour, alpha=0.85, linewidth=1.7,
                zorder=3)
        ax.scatter([start[0]], [start[1]], s=54, marker="o", facecolors="none",
                   edgecolors=colour, linewidths=1.3, zorder=4)
        ax.scatter([end[0]], [end[1]], s=95, marker="X", c=colour, edgecolors="white",
                   linewidths=1.0, zorder=5)
    ax.set_title("{} - observer trajectories".format(pretty(meta["name"])), fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("#f6f7f9")
    ax.grid(color="white", linewidth=1.1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_dataset(name, outdir, n_frames=DEFAULT_N_FRAMES, fps=DEFAULT_FPS, seed=2020,
                   make_mp4=True, verbose=True):
    """Build data, fit the model and write every artefact for one data set."""
    X, y, meta = generate(name)
    params = pick_parameters(X)
    model = fit_model(X, params, seed=seed)
    frames, positions, radii, alive = trajectories_for(model, n_frames=n_frames)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)

    # Render the animation once to a short MP4 (lossless enough for re-encoding)
    # and derive the GIF from it; fall back to the Pillow writer when ffmpeg is
    # not installed.
    gif_path = base + ".gif"
    mp4_path = None
    if make_mp4 and shutil.which("ffmpeg"):
        animation_object = _animate(X, model, frames, positions, radii, alive, meta, fps)
        mp4_path = base + ".mp4"
        try:
            animation_object.save(
                mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=100
            )
        except Exception as exc:  # pragma: no cover - ffmpeg availability varies
            print("  (mp4 skipped: {})".format(exc))
            mp4_path = None
        plt.close("all")
        if mp4_path and os.path.exists(mp4_path):
            optimised = _optimise_gif(mp4_path, gif_path, fps)
            if optimised:
                gif_path = optimised
            else:  # pragma: no cover - palette filter unavailable
                _gif_from_pillow(X, model, frames, positions, radii, alive, meta, fps, gif_path)
        elif not os.path.exists(gif_path):  # pragma: no cover - defensive
            _gif_from_pillow(X, model, frames, positions, radii, alive, meta, fps, gif_path)
    else:
        _gif_from_pillow(X, model, frames, positions, radii, alive, meta, fps, gif_path)

    overview = render_overview(X, model, frames, positions, radii, alive, meta)
    png_path = base + ".png"
    overview.savefig(png_path, bbox_inches="tight")
    plt.close(overview)

    # Keep the generated data so runs are reproducible / inspectable.
    np.save(base + "_data.npy", X)
    np.save(base + "_labels.npy", y)

    stats = {
        "dataset": name,
        "title": meta["title"],
        "description": meta["description"],
        "n_clusters": meta["n_clusters"],
        "n_samples": int(X.shape[0]),
        "n_observers": int(model.observers.shape[0]),
        "parameters": {
            "R": float(model.R_),
            "factor": float(model.factor),
            "alpha": float(model.alpha),
            "core": model.core,
            "metric": model.metric,
            "random_state": seed,
            "shuffle_data": bool(model.shuffle_data),
        },
        "n_frames": int(positions.shape[0]),
        "frames": [int(value) for value in frames],
        "total_movement": round(_total_movement(positions, alive), 3),
        "median_displacement": round(_median_displacement(positions, alive), 3),
        "files": {
            "gif": os.path.basename(gif_path),
            "mp4": None if not mp4_path else os.path.basename(mp4_path),
            "overview": os.path.basename(png_path),
            "data": os.path.basename(base + "_data.npy"),
            "labels": os.path.basename(base + "_labels.npy"),
        },
    }
    with open(base + ".json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    if verbose:
        print(
            "  {:<26} {:<4} observers  {:>3} frames  median displacement {:5.2f}".format(
                name,
                stats["n_observers"],
                stats["n_frames"],
                stats["median_displacement"],
            )
        )
    return stats


def _optimise_gif(mp4_path, gif_path, fps):
    """Re-encode ``gif_path`` from ``mp4_path`` with a compact palette."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:  # pragma: no cover - guarded by the caller
        return None
    temporary = gif_path + ".tmp.gif"
    lavfi = (
        "fps={fps},scale=800:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=none"
    ).format(fps=int(fps))
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", mp4_path,
               "-vf", lavfi, temporary]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except Exception:  # pragma: no cover - subprocess availability
        return None
    if completed.returncode != 0 or not os.path.exists(temporary):
        return None
    os.replace(temporary, gif_path)
    return gif_path


def _gif_from_pillow(X, model, frames, positions, radii, alive, meta, fps, gif_path):
    """Write the GIF with matplotlib's Pillow writer (no ffmpeg required)."""
    animation_object = _animate(X, model, frames, positions, radii, alive, meta, fps)
    animation_object.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close("all")


def _animate(X, model, frames, positions, radii, alive, meta, fps):
    fig, update, init = build_figure(X, model, frames, positions, radii, alive, meta)
    animation_object = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=positions.shape[0],
        interval=max(1, int(round(1000.0 / max(fps, 1)))),
        blit=False,
        repeat=False,
    )
    return animation_object


def _total_movement(positions, alive):
    total = 0.0
    for obs_id in range(positions.shape[1]):
        rows = np.where(alive[:, obs_id])[0]
        if rows.size < 2:
            continue
        total += float(np.linalg.norm(np.diff(positions[rows, obs_id], axis=0), axis=1).sum())
    return total


def _median_displacement(positions, alive):
    displacements = [
        float(np.linalg.norm(end - start))
        for _, start, end, _ in _trajectory_segments(positions, alive)
    ]
    return float(np.median(displacements)) if displacements else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="data sets to animate (default: all)")
    parser.add_argument("--list", action="store_true", help="print the data set names and exit")
    parser.add_argument("--outdir", default=os.path.join(ROOT, "animations"),
                        help="output directory (default: <repo>/animations)")
    parser.add_argument("--frames", type=int, default=DEFAULT_N_FRAMES,
                        help="number of animation frames (default: %(default)s)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="frames per second (default: %(default)s)")
    parser.add_argument("--no-mp4", action="store_true", help="only write GIFs")
    args = parser.parse_args(argv)

    names = list_datasets()
    if args.list:
        print("\n".join(names))
        return 0

    selected = args.datasets or names
    unknown = [name for name in selected if name not in names]
    if unknown:
        parser.error("unknown data set(s): {} (choose from {})".format(unknown, names))

    print("Rendering {} data set(s) into {}".format(len(selected), args.outdir))
    for name in selected:
        render_dataset(name, args.outdir, n_frames=args.frames, fps=args.fps,
                       make_mp4=not args.no_mp4)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
