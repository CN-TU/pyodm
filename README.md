# pyDMC
Data Modeling and compression algorithm

## Installation
(currently the repository is private).

pysdo can be installed using pip by running
```pip install git+https://github.com/CN-TU/pydmc```

## Usage

### Create yout model
```python
import pydmc
import pandas

X_train = pandas.read_csv('my_dataset_training.csv')
X_test = pandas.read_csv('my_dataset_testing.csv')

model = pydmc.DMC()
```

### Supervised classification
```python
centroides = model.fit(X=X_train)
predictions = model.predict(X=X_test)
```
The same data can be used as trainign and testing data.In this case, the algorithm works as a modeling algorithm and returns a set of centroides with are a compressed version of the original data (can be used for SDO for instance).


### Unsupervised classification
```python
centroides = model.fit(X=X_train)
outlierness_scores = model.outlierness(X=X_test, mode='mean')
```
