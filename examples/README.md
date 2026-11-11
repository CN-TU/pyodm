# Examples

## Observer-movement animations

`animate_observers.py` generates synthetic 2-D data sets and renders, for each
one, an animation showing how the ODM observers move while the data are
processed.

```bash
# from the repository root, with the private venv activated
python examples/animate_observers.py --list          # available data sets
python examples/animate_observers.py                 # render every data set
python examples/animate_observers.py eight_blobs     # render a single one
python examples/animate_observers.py --frames 150 --fps 15 --no-mp4
```

Outputs are written to `animations/` (override with `--outdir`):

| file | content |
|------|---------|
| `<name>.gif` | the animation (optimised palette, ~1 MB) |
| `<name>.mp4` | the same animation, if `ffmpeg` is on the `PATH` |
| `<name>.png` | static overview: data, start markers, final observers and trajectories |
| `<name>_data.npy`, `<name>_labels.npy` | the generated data and ground-truth labels |
| `<name>.json` | parameters, number of observers, per-frame sample indices |

### Data sets (`datasets.py`)

| name | clusters | what it stresses |
|------|----------|------------------|
| `eight_blobs` | 8 | well separated Gaussian clusters |
| `variable_density` | 6 | cluster sizes/spreads differing by an order of magnitude |
| `concentric_rings` | 5 | non-convex, interleaved clusters |
| `anisotropic_overlapping` | 5 | elongated, rotated, partly overlapping clusters |
| `crescents` | 6 | crescent-shaped clusters with satellite blobs |
| `islands_in_noise` | 5 | dense islands inside a large uniform background |
| `spiral_arms` | 3 | interleaved spiral arms |

Every generator returns `(X, y, meta)`; `meta` holds the title, description and
the number of ground-truth clusters (written to the JSON next to the
animation).

### Frame content

* grey dots -- data points, drawn progressively as they are scanned,
* hollow circles -- where each observer was born (end of the first iteration),
* coloured **X** markers -- the live observer positions of the current frame,
* coloured trails -- the path each observer travelled so far,
* translucent discs -- the current radius of each observer,
* top-left text -- current iteration, number of observers and mean radius.

### Parameters

The initial radius is a fraction of the data extent (`RADIUS_FRACTION = 0.13`)
and `factor = 0.01`. The radius estimation of the library (nearest-neighbour
based) yields hundreds of observers on these data sets, which makes an
individual observer's motion invisible; using a larger `R` produces a handful
of observers per cluster (13-62 in total), which is what the animation needs.
`ODM.trajectories()` is a pure replay of the logged events, so the final frame
matches `model.observers` exactly.
