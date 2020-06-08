# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Name        : __init__.py
# Description : Observers-based Data Modeling algorithm
# Author      : Fares Meghdouri
#
# Notes       : Currently only the M-Tree core is implemented
#
# Change-log  : 28-10-2019: //FM// Uploaded v1 with a KD-tree core.
#               06-02-2020: //FM// Rewrote the core v2 (now we use M-tree 
#                           as described in the paper -KD-tree core 
#                           will come back in the next version).
#               08-06-2020: //FM// add new features: select the number 
#                           of observers, rename parameters, add parameters.
#
#
#
#******************************************************************************

from joblib import Parallel, delayed
#from MTree import MTree
from streamod import MTree
import multiprocessing
import numpy as np
import cProfile
from scipy.spatial.distance import pdist, cdist
import random
from scipy.spatial import cKDTree

# TODO: delete redundant assignment of O and mode

class ODM(object):
    """Observers Based Data Modeling"""
    
    def __init__(self, factor=0.1, R=None, rho=0.05, m=None, alpha=1, beta=1, core='m-tree', 
        distance_estimation='adtm', parallelize=False, chunksize=None, random_state=None, 
        shuffle_data=True, n_cores=1, O=5, mode='median', verbose=0):
        """
        Parameters
        ----------
        factor: float, required (default=0.1)
            A factor by wich the radius of a cluster is adjusted.

        R: float, optional
            Default new observer's radius.
            If None, the default radius is estimated based 
            on the mean distance of a set 
            'Y' of randomely chosen points and 
            scaled with the 'beta' parameter.
        
        rho: float, required if 'R' s not given (default=0.025 ~ 2.5%)
            Fraction of data points to use for estimating
            the initial radius of a new observer.
        
        m: int, optional (consider all observers if not given)
            The number of observers to keep. 

        alpha: float, optional (default=1)
            A scaling factor of the observer's location shift. 

        beta: float, optional (default=1)
            A correction parameter of the estimated default radius. 

        core: str, required ['kd-tree', 'm-tree' (default)]
            The algorithm used for getting shortest distances.

        distance_estimation: str, required ['adtm' (default), 'aid']
            The algorithm used for estimating R.
            adtm: Average Distance To the Mean
            aid: Average Inter-Distances

        parallelize: boolean, required (default=True)
            Estimate the default radius as per chunk, this may
            yell better performance and an less runtime.
            Note that this might results in more observers.
        
        chunksize: int, optional (default=None)
            Process data in chunks of size 'chunksize'.
            If None, the whole data is taken at once.  Note that 
            this might results in more observers.
            
        random_state: RandomState, int or None, optional (default=None)
            If np.RandomState, use random_state directly. If int, use
            random_state as seed value. If None, use default random state.
        
        shuffle_data: boolean, optional (default=True)
            Shuffle the data before starting to avoid bias.
        
        n_cores: int, (default=1) 
            Number of cores to use for the parallelization. 
            Ignored if parallelize is set to False.
            '1' means no parallelization.

        O: int, optional (default=1)
            Number of closest observers to consider for outlierness estimating.
            
        mode: str, [median (default), 'mean', 'max', 'min', 'sum']
            The way the outlierness distance is calculated based 
            on the 'O' nearest observers.

        verbose: int, optional (default=0)
            Level of stdout information to print, 0 prints only warnings/errors
            and 1 prints also the state.
        """
        
        self.factor              = factor
        if R:
            self.estim_R         = False
            self.R               = R
        else:
            self.estim_R         = True
            self.R               = 0.
        self.rho                 = rho
        self.m                   = m
        self.alpha               = alpha
        self.beta                = beta
        self.core                = core
        self.distance_estimation = self._estimate_R_average_distance if distance_estimation=='adtm' else self._estimate_R_inter_distances
        self.parallelize         = bool(parallelize)
        self.chunksize           = int(chunksize) if chunksize else None
        self.random_state        = random_state
        self.shuffle_data        = bool(shuffle_data)
        self.n_cores             = int(n_cores)
        self.O                   = int(O)
        self.node                = mode
        self.verbose             = int(verbose)

        if self.verbose == 1:
            print('INFO: model created! check model.get_params() to get the current parameters')

        return

    def fit(self,X):
        """ Create an ODM coreset from the data """

        [self.n, self.N] = X.shape

        self.observers   = np.empty((1, self.N))
        self.population  = np.empty(1)
        self.radius      = np.empty(1)

        if self.shuffle_data:
            X = self._shuffle(X)

        if self.estim_R and not self.parallelize:
            self.R = self.beta * self.distance_estimation(X)
            if self.verbose == 1:
                print('INFO: R estimated to be {}'.format(self.R))

        if not self.chunksize:
            self.chunksize = self.n
        
        if self.n_cores !=1:
            n = multiprocessing.cpu_count()
            if self.n_cores > n:
                print('WARNING: only {} cores available'.format(n))
                self.n_cores = n

        for subtask in Parallel(n_jobs=self.n_cores)(delayed(self._doWork_mtrees)(batch) for batch in np.array_split(X, self.n/self.chunksize)):
            self.observers  = np.vstack((self.observers, subtask[0]))
            self.population = np.vstack((self.population, subtask[1]))
            self.radius     = np.vstack((self.radius, subtask[2]))

        self.observers      = np.delete(self.observers, 0, axis=0)
        self.population     = np.delete(self.population, 0, axis=0)
        self.radius         = np.delete(self.radius, 0, axis=0)

        if self.m:
            print('INFO: extracting {} observers'.format(self.m))
            if self.m > self.observers.shape[0]:
                append_random   = random.sample(range(0, self.n), self.m-self.observers.shape[0])
                self.observers  = np.vstack((self.observers, X[append_random]))
            else:
                self.order      = np.squeeze(self.population).argsort()[-self.m:]
                self.observers  = self.observers[self.order]
                self.population = self.population[self.order]
                self.radius     = self.radius[self.order]

        print('INFO: Done.')
        return

    def _doWork_mtrees(self, batch):
        """ ODM Algorithm - mTrees"""

        # FIXME: rename variables
        [mm, nn] = batch.shape

        if self.estim_R and self.parallelize:
            R = self.distance_estimation(batch) * self.beta
        else:
            R = self.R
            
        sub_population     = np.zeros([mm, 1])
        ids                = np.empty([mm+1], dtype=int)
        sub_observers      = np.empty([mm, nn])
        
        sub_observers[0]   = batch[0]
        sub_population[0] += 1
        sub_radius         = np.full([mm, 1], R)

        i = 0

        tree     = MTree.MTree()
        [_id]    = tree.insert(sub_observers[0])

        ids[_id] = 0

        for data_point in batch[1:]:

            [[ind], [dist], [counter]] = tree.knn_query(data_point, k=1)

            if dist <= sub_radius[ids[ind]]:
                
                distances = np.subtract(data_point, sub_observers[ids[ind]])
                
                tree.remove(np.array([ind]))
                sub_observers[ids[ind]]   = sub_observers[ids[ind]] + distances/(sub_population[ids[ind]] + 1)
                [_id]                     = tree.insert(sub_observers[ids[ind]])
                
                sub_population[ids[ind]] +=1
                sub_radius[ids[ind]]      = max(0, sub_radius[ids[ind]] - self.factor * dist)

                ids[_id]                  = ids[ind]
                
            else:
                
                sub_radius[ids[ind]]      = max(0, sub_radius[ids[ind]] + self.factor * dist)
                i +=1
                
                sub_observers[i]   = data_point
                [_id]              = tree.insert(sub_observers[i])

                ids[_id]           = i

                sub_population[i] += 1
                
        return sub_observers[:i+1], sub_population[:i+1], sub_radius[:i+1]

    def _doWork_kdtrees(self, batch):
        """ ODM Algorithm - kdTrees"""

        pass
        
    def predict(self, X, contamination=0.1, O=5, mode='median', chunksize=1000):
        """
        Only perform outlier detection based on a trained model.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        contamination: float, optional (default=0.1)
            Ratio of outliers in the dataset.

        O: int, optional (default=5)
            Number of closest observers to use for estimating 
            the outlierness.
            
        mode: str, [median (default), 'mean', 'max', 'min', 'sum']
            The way the outlierness distance is calculated based 
            on the "O" nearest observers distances.
        
        chunksize: int, optional
            split data into smaller chunks for multiprocessing. This might create overlaping sub-coresets.
        
        Returns
        ---------------
        y: ndarray, shape (n_samples)
            Binary predicted labels for input data.
        """
        
        if not self.O:
            self.O = O

        if not self.mode:
            self.mode = mode
        
        outlierness = self.outlierness(X, O, mode, self.chunksize if self.chunksize else chunksize)
        threshold = np.quantile(outlierness, 1-contamination)
        return outlierness > threshold

    def outlierness(self, X, O=5, mode='median', chunksize=1000):
        """
        Compute the outlierness scores based on a trained model.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
        
        O: int, optional (default=1)
            Number of closest observers to use for estimating 
            the outlierness.
            
        mode: str, [median (default), 'mean', 'max', 'min', 'sum']
            The way the outlierness distance is calculated based 
            on the "O" nearest observers.

        chunksize: int, optional
            split data into smaller chunks for multiprocessing. This might create overlaping sub-coresets.
            
        Returns
        ---------------
        y: ndarray, shape (n_samples)
            Outlierness scores for input data.
        """
        
        if not self.O:
            self.O    = O

        if not self.mode:
            self.mode = mode
        
        [m, n]    = X.shape
        outli     = []
        
        for i in Parallel(n_jobs=self.n_cores)(delayed(self._get_dist_mtree)(sub_batch) for sub_batch in np.array_split(X, m/self.chunksize if self.chunksize else chunksize)):
            outli.append(i)
        
        return np.concatenate(outli, 0)
    
    def _get_dist_mtree(self, batch):
        """ Get a distance metric of a point to its O neighbours """

        [m, n] = batch.shape
        out    = np.empty(m)
        
        tree   = MTree.MTree()
        tree.insert(self.observers)
        
        for index, data_point in enumerate(batch):
            [ind, dist, counter] = tree.knn_query(data_point, self.O)

            if self.node   == 'mean':
                out[index] = np.mean(dist)
            elif self.node == 'max':
                out[index] = np.max(dist)
            elif self.node == 'min':
                out[index] = np.min(dist)
            elif self.node == 'sum':
                out[index] = np.sum(dist)
            else:
                out[index] = np.median(dist)

        return out
        
    def _estimate_R_inter_distances(self, X):
        """
        Estimate the average disatnce between points which 
        will be used as initial new cluster radius.
        The distance is estimated by taking a random 
        (random_state) subset of points (n_points) and average 
        the distance between points.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, N_features)
            The input data.
            
        Returns
        ---------------
        R: float
            The estimated default radius.
        """
        
        [n, N] = X.shape
        points = int(n*self.rho)

        np.random.seed(self.random_state)
        data = X[np.random.choice(n, points, replace=False), :]

        # Old way
        #tot  = 0.
        #for i in range(points-1):
        #    tot += (np.abs((np.abs(data[i+1:]-data[i])**2).sum(1))**.5).sum()
        #out = tot/((points-1)*(points)/2.)

        return np.mean(pdist(data))

    def _estimate_R_average_distance(self, X):
        """
        Estimate the average disatnce between points which 
        will be used as initial new cluster radius.
        The distance is estimated by taking a random 
        (random_state) subset of points (n_points) and average 
        the distance to the centroid.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, N_features)
            The input data.
            
        Returns
        ---------------
        R: float
            The estimnated default radius.
        """

        [n, N] = X.shape
        points = int(n*self.rho)

        np.random.seed(self.random_state)
        data = X[np.random.choice(n, points, replace=False), :]
        return np.mean(cdist(data, np.mean(data, axis=0).reshape(1,-1)))


    def labels(self, X):
        """
        Return the label of each data point based on the closest observer
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        Returns
        ---------------
        labels: ndarray, shape (n_samples)
            Outlier scores for input data.
        """

        tree      = cKDTree(self.observers)   
        dist, ind = tree.query(X, k=1)
        return ind

    def _shuffle(self, X, labels=None):
        """
        Unison shuffling of data and labels (keep the order).
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, N_features)
            The input data.

        labels: 1darray, shape (n_samples)
            The input labels.
        
            
        Returns
        ---------------
        X: ndarray, shape (n_samples, N_features)
            Shuffled input data.

        labels: 1darray, shape (n_samples)
           Shuffeled input labels.
        """

        np.random.seed(self.random_state)
        p = np.random.permutation(X.shape[0])
        return X[p], labels[p] if labels else X[p]
        
    def get_params(self, deep=True):
        """
        Return the model's parameters
        
        Parameters
        ---------------
        deep : bool, optional (default=True)
            Return sub-models parameters.
            
        Returns
        ---------------
        params: dict, shape (n_parameters,)
            A dictionnary mapping of the model's parameters.
        """
        
        return {"R":self.R,
                "O":self.O,
                "m":self.m,
                "factor":self.factor,
                "core":self.core,
                "rho":self.rho,
                "parallelize":self.parallelize,
                "alpha":self.alpha,
                "beta":self.beta,
                "random_state":str(self.random_state),
                "shuffle_data":self.shuffle_data,
                "chunksize":self.chunksize,
                "contamination":self.contamination,
                "n_cores":self.n_cores,
                "distance_estimation":self.distance_estimation,}