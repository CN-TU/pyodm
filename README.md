# pyOBDM
Observers Based Data Modeling

## Installation
(currently the repository is private).

pyobdm can be installed using pip by running
```pip install git+https://github.com/CN-TU/pyobdm```

## Usage

### Create yout model
```python
import pyobdm
import numpy
import pandas

X_train = np.load('my_dataset_training.npy')
X_test = np.load('my_dataset_testing.npy')

#or
#X_train = pandas.read_csv('my_dataset_training.csv').values
#X_test = pandas.read_csv('my_dataset_testing.csv').values

model = pyobdm.OBDM(shuffle=True)
```

### Binary classification
```python
model.fit(X_train)
predictions = model.predict(X_test)
```
The same data can be used for trainign and testing. In this case, the algorithm works as a modeling algorithm and returns a set of observers wich are a compressed version of the original data (can be used as a plug-in for SDO for instance, parse `model.observers` after fitting).


### Outlierness scores
```python
model.fit(X_train)
outlierness_scores = model.outlierness(X_test, mode='mean')
```

### Points labeling

to get the label of the closest observer to a set of points use:
```python
model.fit(X_train)
outlierness_scores = model.labels(X_test)
```

### Get parameters
```python
model.fit(X_train)
model.get_params()
```

This will return a dictionnary of parameters used to build the model.

## Contact
fares.meghdouri@tuwien.ac.at