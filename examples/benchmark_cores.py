"""Small benchmark comparing the exact kNN cores used by ODM.

Run with::

    python examples/benchmark_cores.py [--n 20000] [--dims 8]

The script is intentionally dependency-free (no matplotlib, no sklearn): it
only times the three cores of the package on random data, then checks that
they build the same model and prints the index statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyodm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20000, help="number of samples")
    parser.add_argument("--dims", type=int, default=8, help="number of dimensions")
    parser.add_argument("--clusters", type=int, default=12, help="number of clusters")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    args = parser.parse_args(argv)

    rng = np.random.RandomState(args.seed)
    centres = rng.uniform(-50, 50, size=(args.clusters, args.dims))
    X = np.vstack(
        [rng.normal(centres[row], 1.5, size=(args.n // args.clusters, args.dims))
         for row in range(args.clusters)]
    )

    header = "{:<10} {:>10} {:>12} {:>14} {:>12}".format(
        "core", "fit (s)", "observers", "outlierness (s)", "AUC-ish"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for core in ("m-tree", "kd-tree", "brute"):
        model = pyodm.ODM(core=core, random_state=args.seed, R=2.5)
        started = time.perf_counter()
        model.fit(X)
        fit_time = time.perf_counter() - started

        started = time.perf_counter()
        scores = model.outlierness(X[:5000])
        score_time = time.perf_counter() - started
        results[core] = (model.observers, model.population, scores)
        print("{:<10} {:>10.2f} {:>12d} {:>14.2f} {:>12.3f}".format(
            core, fit_time, model.observers.shape[0], score_time,
            float(scores.mean()),
        ))

    reference = results["m-tree"]
    for core in ("kd-tree", "brute"):
        same = (
            reference[0].shape == results[core][0].shape
            and np.allclose(reference[0], results[core][0])
            and np.array_equal(reference[1], results[core][1])
        )
        print("core '{}' builds the same model as 'm-tree': {}".format(core, same))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
