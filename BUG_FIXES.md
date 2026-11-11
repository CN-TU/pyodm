# pyODM — bug analysis and fixes

This document records the defects found in the original `pyodm/__init__.py`
(the version that depended on the private `dSalmon` package) and what was done
about each of them.  The public behaviour of the algorithm is unchanged except
where the old code was plainly wrong; every change is covered by a test in
`tests/`.

## Blocking defects (the package could not run at all)

| # | Problem | Where | Fix |
|---|---------|-------|-----|
| 1 | Hard dependency on the private `dSalmon` package (`from dSalmon.trees import MTree`). The repository is not public, so the package was unusable outside TU Wien. | module import | Self-contained exact indexes in `pyodm/trees/` (`m-tree`, `kd-tree`, `brute`). |
| 2 | `np.array_split(X, self.n/self.chunksize)` passes a **float** as the number of sections. NumPy requires an integer and raises `ValueError: number sections must be larger than 0` whenever the chunk is larger than the data (i.e. the default configuration). | `fit`, `outlierness` | `n_chunks = ceil(n_samples / chunk_size)`. |
| 3 | Same float-section bug in the outlierness path; additionally `m / chunksize` was computed with the *variable* `m` (number of samples) shadowing the *parameter* `m` (number of observers). | `outlierness`, `_get_dist_mtree` | Ceil-based chunk count, no shadowing. |
| 4 | `np.random.seed(self.random_state)` in `_shuffle`, `_estimate_R_*` raises `TypeError` when `random_state` is a `RandomState` object — which the docstring explicitly allows. It also reseeds NumPy's **global** RNG. | `_shuffle`, radius estimation | `_utils.check_random_state` + local generator; supports `int`, `None`, `RandomState`, `Generator`. |

## Crashes and wrong results after fitting

| # | Problem | Where | Fix |
|---|---------|-------|-----|
| 5 | `get_params()` returns `self.contamination`, an attribute that is never set → `AttributeError` on every call (the README documents `model.get_params()`). | `get_params` | Removed from the dict; `metric`/`track_history` added. |
| 6 | `m > created_observers` path appends random data points to `self.observers` **without** extending `population`/`radius`, so the three arrays have different lengths and any later indexing/slicing crashes. It also uses the global `random` module, ignoring `random_state`. | `fit` | When `m` exceeds the number of created observers, all observers are kept (consistent arrays) and a warning is logged. |
| 7 | In `_doWork_mtrees`, ids are recycled through the M-Tree between inserts: `ids[_id] = i` uses the *new* tree id, but when an observer moves the code writes `ids[_id] = ids[ind]` while `sub_radius`/`sub_observers` are indexed by the *observer* index. With the old data structure this worked by accident; the logic was fragile and broke whenever the tree renumbered entries. | `_doWork_mtrees` | Observer bookkeeping is decoupled from the index: observers are indexed by a plain integer `0..n-1`, the index maps them 1:1. |
| 8 | `if dist <= sub_radius[ids[ind]]` compares the query distance with the radius of the observer, but the radius is updated with `max(0, ...)`, clamping at zero; the original code never re-grows a radius that reached 0, so an observer whose radius collapsed can never accept a point again. | `_doWork_mtrees` | Radii are clamped at 0 on the downward update only (as in the paper); rejected points grow the nearest radius again. Documented and covered by tests. |
| 9 | `alpha` is documented ("scaling factor of the observer's location shift") but **never used**. | `_doWork_mtrees` | Applied as `observer += alpha * (x - observer) / (population + 1)`. `alpha=1` reproduces the previous update exactly. |
| 10 | `beta` is documented ("correction parameter of the estimated default radius") but is only applied in the sequential path; in the `parallelize=True` path it is applied per chunk — the two code paths disagreed. | `fit` | Both paths apply `R = beta * estimate`. |
| 11 | `_shuffle` ends with `return X[p], labels[p] if labels else X[p]`: `bool(ndarray)` raises `ValueError` for any non-empty label array, and when it did not raise, the returned tuple/np array mixed the arguments. | `_shuffle` | Correct `(X, labels)` pair or `X` when no labels are given. |
| 12 | `predict`/`outlierness` silently mutate model state: they assign `self.O`/`self.mode` from the call arguments, so a later call without arguments uses a stale value, and the attribute is named `node` in the constructor (`self.node = mode`) but referenced as `self.mode` — an `AttributeError` waiting to happen. | `predict`, `outlierness`, `__init__` | Parameters are resolved per call (`O = self.O if O is None else ...`), the attribute is `self.mode` everywhere. |
| 13 | `if not self.O: self.O = O` treats `O=0` as "unset" (0 is a valid (degenerate) request that then crashes elsewhere). | `predict`, `outlierness` | Explicit `None` sentinel plus range validation. |
| 14 | `fit` never resets fitted state, so calling `fit` twice on the same object stacks arrays (the `np.vstack` accumulator is re-initialised, but stale history/attributes survive) and `self.chunksize` is *mutated* to `self.n` on the first call, silently changing the model parameters. | `fit`, `__init__` | `fit` is idempotent; the effective chunk size is a local variable. |
| 15 | `_estimate_R_inter_distances` / `_estimate_R_average_distance` call `np.random.choice(n, points)` with `points = int(n * rho)`, which is **0** for `n < 1/rho` (20 samples by default) and raises `ValueError: Cannot take a larger sample than population` (or produces an empty pdist average → `nan` radius). | radius estimation | Sample size clamped to `[3, n]`; validated inputs. |
| 16 | No input validation anywhere: 1-D arrays, `NaN`s, mismatched feature counts, empty arrays and non-finite values all produce obscure downstream failures. | `fit`, `outlierness`, `predict`, `labels` | `_utils.as_array2d` validates shape/dtype/finiteness; `labels` checks the feature count against the fitted model. |

## The k-NN core

| # | Problem | Fix |
|---|---------|-----|
| 17 | `_get_dist_mtree` builds a **new** M-Tree for every chunk (and the tree is rebuilt per chunk), so the observer set is re-inserted over and over. | A single index is built per scoring call (`ODM._observer_index`), and bulk loading is used when the index supports it. |
| 18 | `tree.knn_query(point, k)` with `k > number_of_observers` returns fewer entries; `out[index] = np.mean(dist)` then silently used a shorter vector (fine) but `[ind, dist, counter] = ...` in the fit loop unpacked a fixed-size tuple and crashed when the tree returned a scalar for a single object. | Indexes clamp `k` and always return arrays; the fit loop uses `index.query` with a normalised `(ids, dists)` contract. |

## Design of the new cores

`pyodm/trees/` implements three interchangeable, **exact** indexes with the
same interface (`insert`, `remove`, `update`, `query`, `query_many`):

* **`MTree`** (`pyodm/trees/mtree.py`) — a real dynamic metric tree as in the
  paper: routing objects stored *by value*, tight covering radii recomputed
  after every structural change, best-first kNN with branch-and-bound pruning,
  node splits using the `PickPivots2` heuristic, deletion with node
  condensation. Works with any metric, including a user-supplied callable.
* **`KDTreeIndex`** (`pyodm/trees/kdtree.py`) — `scipy.spatial.cKDTree` with
  lazy deletions and amortised rebuilds (`rebuild_ratio`), exact because
  changed objects are scanned separately until the next rebuild. Fastest for
  Euclidean/manhattan/chebyshev data.
* **`BruteForceIndex`** (`pyodm/trees/brute.py`) — linear scan, used as the
  ground truth in the test suite and as a fallback for custom metrics.

All three return results sorted by `(distance, object_id)`, so the three cores
produce **bit-identical models** for the same data and seed (verified by
`tests/test_odm.py::test_all_cores_produce_the_same_model`).

The M-Tree stores **copies** of the inserted points. This matters: the original
fit loop updated its `sub_observers` array in place and re-inserted the row,
and an index that kept a reference would silently change its coordinates
without any structural update, corrupting the covering radii (a bug found
while porting, see `test_insert_move_order_does_not_change_observer_state`).

## Additions

* `track_history=True` records every observer event (birth / move / radius
  update) during `fit`.
* `ODM.trajectories(n_frames=...)` replays that log into
  `(frames, positions, radii, alive)` arrays, ready for animation.
* `examples/datasets.py` — seven synthetic 2-D data sets with many clusters.
* `examples/animate_observers.py` — renders one animation per data set.
* `pyproject.toml` + `requirements.txt` — installable without any private
  dependency; `pip install -e .` inside a private venv is enough.

## Deliberate behaviour notes

* `factor` still shrinks the radius on an accepted point and grows it on a
  rejected one, exactly as in the paper. With the default `factor=0.1` and a
  nearest-neighbour based `R`, the model converges to very few observers; the
  animation example uses a larger `R` (a fraction of the data extent) and a
  small `factor` so that observers keep moving and stay visible.
* `population` counts the points assigned to an observer (including the point
  that created it), so `population.sum() == n_samples` always holds.
