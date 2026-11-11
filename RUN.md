# Running pyODM (quick start)

Everything below assumes you are in the repository root.

## 1. Private virtual environment

The venv lives in `.venv/` (ignored by git):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -r requirements.txt` brings in numpy, scipy, joblib (runtime),
matplotlib (animations) and pytest (tests).  No private package is needed any
more — the `dSalmon` M-Tree dependency is gone.

## 2. Smoke test

```bash
python -c "import pyodm; print(pyodm.__version__)"
python -m pytest tests -q
```

## 3. Build a model

```python
import numpy as np, pyodm

X = np.load('my_dataset.npy')
model = pyodm.ODM(core='m-tree', random_state=1).fit(X)
print(model.observers, model.population, model.radius)
print(model.outlierness(X[:100]))
print(model.predict(X[:100]))
```

## 4. Animations

```bash
python examples/animate_observers.py --list     # data set names
python examples/animate_observers.py            # all data sets -> animations/
python examples/animate_observers.py eight_blobs --frames 120 --fps 14
```

Each run writes `<name>.gif`, `<name>.mp4` (if ffmpeg is installed),
`<name>.png` (static overview), the generated data and a JSON with the
parameters.

MPLCONFIGDIR note: matplotlib writes a font cache; if `~/.matplotlib` is not
writable, export `MPLCONFIGDIR` to a writable directory first, e.g.

```bash
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
```

## 5. Benchmark of the kNN cores

```bash
python examples/benchmark_cores.py --n 6000 --dims 5
```
