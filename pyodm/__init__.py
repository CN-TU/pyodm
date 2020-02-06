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
# Change-log  : 28-10-2019: //FM// Uploaded v1 with a KD-tree core
#               06-02-2020: //FM// Rewrote the core v2 (no we use M-tree 
#                           as described in the paper -KD-tree core 
#                           will come back in the next version)
#
#
#
#******************************************************************************

from joblib import Parallel, delayed
#from scipy.spatial import cKDTree
from MTree import MTree
import multiprocessing
import numpy as np
import cProfile




class ODM(object):
    """Observers Based Data Modeling"""
    
    def __init__(self, R=None, factor=0.1, core='m-tree', n_points_frac=0.025, per_chunk_estimate=True, radius_est_factor=0.5, random_state=None, shuffle_data=True, chunksize=None, n_cores=1, O=5, mode='median'):
        """
        Parameters
        ----------
        R: float, optional
            Default new observer's radius.
            If None, the default radius is estimated based 
            on the mean distance of a set 
            'n_points_frac' of randomely chosen points and 
            scaled with the 'radius_est_factor' parameter.
            
        factor: float, optional (default=0.1)
            A factor by wich the radius of a cluster is adjusted.

        core: str, ['kd-tree', 'm-tree' (default)]
            The algorithm used for getting shortest distances.
            
        n_points_frac: float, optional (default=0.05 ~ 5%)
            Fraction of data points to use for estimating
            the initial radius of a new observer.
        
        per_chunk_estimate: boolean, optional (default=True)
            Estimate the default radius as per chunk, this may
            yell better performance and an additionnal runtime.
        
        radius_est_factor: float, optional (default=0.5)
            A scaling factor of the estimated default radius. 
            By default the distance is devided by 2.
            
        random_state: RandomState, int or None, optional (default=None)
            If np.RandomState, use random_state directly. If int, use
            random_state as seed value. If None, use default random state.
        
        shuffle_data: boolean, optional
            Shuffle the data before starting. This is advisable
            and set to True by default.
        
        chunksize: int, optional (default=None)
            Process data in parallel in chunks of size chunksize.
            If None, the whole data is taken at once.  Note that 
            this might results
            in more observers.
            
        n_cores: int, (default=-1) 
            Number of cores to use for the parallelization. 
            '1' means no parallelization.

        O: int, optional (default=1)
            Number of closest observers to use for estimating 
            the outlierness.
            
        mode: str, [median (default), 'mean', 'max', 'min', 'sum']
            The way the outlierness distance is calculated based 
            on the "O" nearest observers.
        """
        
        if R:
            self.estim_R        = False
            self.R              = R
        else:
            self.estim_R        = True
            self.R              = 0.
            
        self.factor             = factor
        self.core               = core
        self.n_points_frac      = n_points_frac
        self.per_chunk_estimate = per_chunk_estimate
        self.radius_est_factor  = radius_est_factor
        self.random_state       = random_state
        self.shuffle_data       = shuffle_data
        self.chunksize          = chunksize
        self.n_cores            = n_cores
        self.O                  = O
        self.mode               + mode
        
        return

    def fit(self,X):
        """ Create an ODM coreset from the data """

        [self.m, self.n] = X.shape

        self.observers   = np.empty((1, self.n))
        self.population  = np.empty(1)
        self.radius      = np.empty(1)

        if self.shuffle_data:
            X = self.shuffle(X)
        
        if self.estim_R and not self.per_chunk_estimate:
            self.R = self.estimate_R(X) * self.radius_est_factor

        if not self.chunksize:
            self.chunksize = self.m
            
        # TODO: check all cases
        
        if self.n_cores !=1:
            n = multiprocessing.cpu_count()
            if self.n_cores > n:
                print('WARNING: only {} cores available'.format(n))
                self.n_cores = n
                
        #results = Parallel(n_jobs=self.n_cores)(delayed(self.doWork)(batch) for batch in np.array_split(X, self.m/self.chunksize))
        
        for i in Parallel(n_jobs=self.n_cores)(delayed(self.doWork)(batch) for batch in np.array_split(X, self.m/self.chunksize)):
            self.observers  = np.vstack((self.observers, i[0]))
            self.population = np.vstack((self.population, i[1]))
            self.radius     = np.vstack((self.radius, i[2]))

        self.observers      = np.delete(self.observers, 0, axis=0)
        self.population     = np.delete(self.population, 0, axis=0)
        self.radius         = np.delete(self.radius, 0, axis=0)
        
        return

    def doWork(self, batch):
        """ ODM Algorithm """

        [mm, nn] = batch.shape

        if self.estim_R and self.per_chunk_estimate:
            R = self.estimate_R(batch) * self.radius_est_factor
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
        
        self.O = O
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
        
        self.O    = O
        self.mode = mode
        
        [m, n]    = X.shape
        outli     = []
        
        for i in Parallel(n_jobs=self.n_cores)(delayed(self.get_dist_mtree)(sub_batch) for sub_batch in np.array_split(X, m/self.chunksize if self.chunksize else chunksize)):
            outli.append(i)
        
        return np.concatenate(outli, 0)
    
    def get_dist_mtree(self, batch):
        """ Get a distance metric of a point to its O neighbours """

        [m, n] = batch.shape
        out    = np.empty(m)
        
        tree   = MTree.MTree()
        tree.insert(self.observers)
        
        for index, data_point in enumerate(batch):
            [ind, dist, counter] = tree.knn_query(data_point, self.O)

            if self.mode   == 'mean':
                out[index] = np.mean(dist)
            elif self.mode == 'max':
                out[index] = np.max(dist)
            elif self.mode == 'min':
                out[index] = np.min(dist)
            elif self.mode == 'sum':
                out[index] = np.sum(dist)
            else:
                out[index] = np.median(dist)

        return out
        
    def estimate_R(self, X):
        """
        Estimate the average disatnce between points which 
        will be used as initial new cluster radius.
        The distance is estimated by taking a random 
        (random_state) subset of points (n_points) and average 
        the distance btween them.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        Returns
        ---------------
        R: float
            The estimnated default radius.
        """
        
        [m, n] = X.shape
        points = int(m*self.n_points_frac)

        np.random.seed(self.random_state)
        data = X[np.random.choice(m, points, replace=False), :]
        tot  = 0.
        for i in range(points-1):
            tot += (np.abs((np.abs(data[i+1:]-data[i])**2).sum(1))**.5).sum()
            
        return tot/((points-1)*(points)/2.)

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

    def shuffle(self, X):
        """
        Unison shuffling of data and labels (keep the order).
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
        
            
        Returns
        ---------------
        X: ndarray, shape (n_samples, n_features)
            Shuffled input data.
        
        """

        np.random.seed(self.random_state)
        p = np.random.permutation(X.shape[0])
        return X[p]
        
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
                "factor":self.factor,
                "core":self.core,
                "n_points_frac":self.n_points_frac,
                "per_chunk_estimate":self.per_chunk_estimate,
                "radius_est_factor":self.radius_est_factor,
                "random_state":str(self.random_state),
                "shuffle_data":self.shuffle_data,
                "chunksize":self.chunksize,
                "contamination":self.contamination,
                "n_cores":self.n_cores,}