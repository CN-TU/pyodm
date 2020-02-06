# pyODM
Observers-based Data Modeling from the paper "Modeling Data with Observers"

## Installation
(currently the repository is private).

pyodm can be installed using pip by running
```pip install git+https://github.com/CN-TU/pyodm```

Note that in order for ODM to work with an M-Tree core, the implementation (package) in [M-Trees](https://github.com/CN-TU/mtree) needs to be installed.

## Usage
Please note that many parameters can be adjusted in order to build a model that perfectly models the data. To know more about ODM parameters, read the full paper or refere to the documentation provided with the code.

### Create your model (coreset)
```python
import pyodm
import numpy
#import pandas

X_train = np.load('my_dataset_training.npy')
X_test = np.load('my_dataset_testing.npy')

#or
#X_train = pandas.read_csv('my_dataset_training.csv').values
#X_test = pandas.read_csv('my_dataset_testing.csv').values

model = pyodm.ODM(random_state=1)
```

### Binary classification
```python
model.fit(X_train) # fit the data and generate a coreset
predictions = model.predict(X_test) # predict the labels of the test data absed on the outlierness score
```
The same data can be used for trainign and testing. In this case, the algorithm works as a modeling algorithm and returns a set of observers wich are a compressed version (coreset) of the original data (can be used as a plug-in for SDO for instance, parse `model.observers` after fitting).


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

This will return a dictionnary of parameters used to build the model.\

## Coresets Examples
Example1: Three datasets in which data-points are represented in gray and the ODM model in red each with different configuration.
![Three datasets in which data-points are represented in gray and the ODM model in red each with different configuration.](/experiements/arti.png)

Example2: Convergence path of an observer.
![Convergence path of an observer.](/experiements/track0.png)

Example3: Two clusters datasets with two observers.
![Two clusters datasets with two observers.](/experiements/track1.png)

Example4: Five clusters datasets with six observers.
![Five clusters datasets with six observers.](/experiements/track2.png)

## Contact
fares.meghdouri@tuwien.ac.at
