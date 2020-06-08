# Experiments
Here we collect th set of scripts used for the paper's experiments.

## Tree
* `anomaly_detection.py` : executes the anomaly detetcion testbed.
* `clustering.py` : executes the clustering testbed.
* `supervised_classification.py` : executes the supervised classification testbed.

## Summarized results
The following files contain results summarized from the `results` directory by running the following (example) script
```python
with open('supervised_classification_table.txt', 'a') as the_file:
    for r, rate in enumerate([0.5, 1, 5, 10]):
    	# remove CNN for other experiments
        for i, algorithm in enumerate(['ODM', 'RS', 'SDO', 'KMC', 'WGM', 'GNG', 'CNN']):
            data = np.load('../results/supervised/{}_RES.npy'.format(algorithm))
            if algorithm == 'ODM':
                line = "{} & \multirow{}\% & ".format(algorithm, rate)
            else:
                line = "{} & & ".format(algorithm)
            for score in range(7):
                line += "${:.2f} \pm {:.2f}$ & ".format(np.mean(data[:,score,r]), np.std(data[:,score,r]))
            the_file.write(line[:-2] + "\\\\ \n")
        the_file.write('\midrule\n')
```
use the following for the baseline variant
```python
for r, rate in enumerate(['']):
    for i, algorithm in enumerate(['BASELINE']):
        data = np.load('../results/supervised/{}_RES.npy'.format(algorithm))
        line = "Baseline & ----- & "
        for score in range(7):
            line += "${:.2f} \pm {:.2f}$ & ".format(np.mean(data[:,score,r]), np.std(data[:,score,r]))
        print(line[:-2] + "\\\\")
```

* `anomaly_detection_table.txt`
* `clustering_table.txt`
* `supervised_classification_table.txt`

## Other algorithms
All other algorithms and additional functions can be found in `utils.py`


## Contact
fares.meghdouri@tuwien.ac.at

datasets will be uploaded soon.