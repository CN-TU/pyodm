# pyODM
Contact: Fares Meghdouri - fares.meghdouri@tuwien.ac.at

Paper: "Modeling Data with Observers"
```
@article{meghdouri2022modeling,
  title={Modeling data with observers},
  author={Meghdouri, Fares and Iglesias V{\'a}zquez, F{\'e}lix and Zseby, Tanja},
  journal={Intelligent Data Analysis},
  volume={26},
  number={3},
  pages={785--803},
  year={2022},
  publisher={IOS Press}
}

```

## Installation

pyodm can be installed using pip by running
```pip install git+https://github.com/CN-TU/pyodm```

Note that in order for ODM to work with an M-Tree core, the implementation (package) in [M-Trees](https://github.com/CN-TU/dSalmon) needs to be installed. The repository is private and will be available soon.

## Usage
Please note that many parameters can be adjusted in order to build a representative model refer to the paper for more information.

### Create a model
```python
import pyodm

# create a new model with default parameters
model = pyodm.ODM(random_state=1)
```

### Construct a coreset
```python
import numpy
#import pandas

# read the data
X = np.load('my_dataset.npy')

#or
#X = pandas.read_csv('my_dataset.csv').values

# model the data
model.fit(X)

# access the array of observers
print(model.observers)

# access the array of radius
print(model.radius)

# access the array of populations
print(model.population)

```

### Outlierness scores
In  order to get the outlierness score of a set of points (based on an ODM model), run the foolowing after fitting a model
```python
# read the data
X_test = np.load('my_test_dataset.npy')

# get the outlierness scores
outlierness_scores = model.outlierness(X_test)
```

### Anomaly detection
One can convert the outlierness score into a binary label (outlier/inlier) using the following
```python
# read the data
X_test = np.load('my_test_dataset.npy')

# convert the outlierness scores into binary labels using a contamination threshold
predictions = model.predict(X_test) 
```

### Points labeling
to get the label of the closest observer to a set of points use
```python
# read the data
X_test = np.load('my_test_dataset.npy')

# return the predicted label of each test point (refering to `model.observers`)
predicted_labels = model.labels(X_test)
```

### Get parameters
```python
# return a dictionnary of the current parameters
model.get_params()
```

This will return a dictionnary of parameters used to build the model.\

## Visual Examples
Example1: Three datasets in which datapoints are represented in gray and the ODM model in red each with a different configuration.
![Three datasets in which data-points are represented in gray and the ODM model in red each with a different configuration.](/experiements/arti.png)

Example2: Convergence path of an observer.
![Convergence path of an observer.](/experiements/track0.png)

Example3: Two clusters datasets with two observers.
![Two clusters datasets with two observers.](/experiements/track1.png)

Example4: Five clusters datasets with six observers.
![Five clusters datasets with six observers.](/experiements/track2.png)


