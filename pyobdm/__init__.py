# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2019, Institute of Telecommunications, TU Wien
#
# Name        : __init__.py
# Description : Observers Based Data Modeling algorithm
# Author      : Fares Meghdouri
#
# Notes : 
#
#
#******************************************************************************

import numpy as np
from scipy.spatial import cKDTree

class OBDM(object):
    """Observers Based Data Modeling"""
    
    def __init__(self, R=None, O=5, factor=0.1, contamination=0.1, n_points=100, random_state=None, chunksize=None, shuffle=False):
        """
        Parameters
        ----------
        R: float, optional
            Default new cluster radius.
            If None, the default radius is estimated based on the mean distance of a set of randomely chosen points
            
        O: int, optional (default=1)
            Number of closest observers to use for estimating the outlierness.
            
        factor: float, optional (default=0.1)
            A factor by wich the radius of a cluster is adjusted.
        
        contamination: float, optional (default=0.1)
            Ratio of outliers in data set. Only used with predict()
            
        n_points: int, optional (default=100)
            Number of data points to use for estimating the initial radius of a new cluster.
            
        shuffle: bool, optional (default=False)
            Shuffle the input data, this is suited for static data.
            
        random_state: RandomState, int or None, optional (default=None)
            If np.RandomState, use random_state directly. If int, use
            random_state as seed value. If None, use default random
            state.
            
        chunksize: int, optional (default=None)
            Process data in chunks of size chunksize.
            If None, the whole data is taken at once. 
        """
        
        if R:
            self.estim_R = False
            self.R = R
        else:
            self.estim_R = True
            self.R = 'dynamic'
            
        self.O = O
        self.factor = factor
        self.random_state = random_state
        self.n_points = n_points
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.contamination = contamination
        
        return

    def fit(self, X):
        """
        Train a new model based on X.

        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
        """

        np.random.seed(self.random_state)
        if self.shuffle:
            np.random.shuffle(X)
            
        [self.m, self.n] = X.shape
        
        self.observers = np.empty((1, self.n))
        self.population = np.empty(1)
        self.radius = np.empty(1)
        
        if self.estim_R:
            alpha = 0.5 #to be adjusted
            self.R = self.estimate_R(X)*alpha
        
        if not self.chunksize:
            self.chunksize = self.m

        for batch_n, batch in enumerate(np.array_split(X, self.m/self.chunksize)):
            
            [self.mm, self.nn] = batch.shape
            shift = self.mm*batch_n
            
            sub_population = np.zeros([self.mm, 1])
            sub_observers = np.empty([self.mm, self.nn])
            sub_observers[0] = batch[0]
            sub_population[0] += 1
            sub_radius = np.full([self.mm, 1], self.R)
            
            i = 0
            
            for index, data_point in enumerate(batch[1:]):

                self.tree = cKDTree(sub_observers[:i+1])
                
                dist, ind = self.tree.query(data_point.reshape(1, -1), k=1)

                if dist <= sub_radius[ind]:
                    distances = np.subtract(data_point, sub_observers[ind])
                    sub_observers[ind] = sub_observers[ind] + distances/(sub_population[ind] + 1)
                    sub_population[ind] +=1
                    sub_radius[ind] = max(0, sub_radius[ind] - self.factor * dist)
                else:
                    sub_radius[ind] = max(0, sub_radius[ind] + self.factor * dist)
                    i +=1
                    sub_observers[i] = data_point
                    sub_population[i] += 1
                    sub_radius[i] = self.R
    
            self.observers = np.vstack((self.observers, sub_observers[:i+1]))
            self.population = np.vstack((self.population, sub_population[:i+1]))
            self.radius = np.vstack((self.radius, sub_radius[:i+1]))
            
        self.observers = np.delete(self.observers, 0, axis=0)
        self.population = np.delete(self.population, 0, axis=0)
        self.radius = np.delete(self.radius, 0, axis=0)
            
        return

    def predict(self, X, mode='median'):
        """
        Only perform outlier detection based on a trained model.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
        
        mode: str, [median (default), 'mean', 'max', 'min', 'sum']
            The way the outlierness distance is calculated based on the "O" nearest observers.
            
        Returns
        ---------------
        y: ndarray, shape (n_samples)
            Binary predicted labels for input data.
        """
        
        np.random.seed(self.random_state)
        if self.shuffle:
            np.random.shuffle(X)
            
        outlierness = self.outlierness(X, mode=mode)
        threshold = np.quantile(outlierness, 1-self.contamination)
        return outlierness > threshold

    def outlierness(self, X, mode='median'):
        """
        Compute the outlierness scores based on a trained model.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
        
        mode: str, [median (default), 'mean', 'max', 'min', 'sum']
            The way the outlierness distance is calculated based on the "O" nearest observers.
            
        Returns
        ---------------
        y: ndarray, shape (n_samples)
            Outlier scores for input data.
        """
        
        np.random.seed(self.random_state)
        if self.shuffle:
            np.random.shuffle(X)
            
        out = np.empty(self.m)
        
        for index, data_point in enumerate(X):
            dist, ind = self.tree.query(data_point.reshape(1, -1), k=self.O)
            
            if mode == 'mean':
                out[index] = np.mean(dist)
            elif mode == 'max':
                out[index] = np.max(dist)
            elif mode == 'min':
                out[index] = np.min(dist)
            elif mode == 'sum':
                out[index] = np.sum(dist)
            else:
                out[index] = np.median(dist)

        return out

    def estimate_R(self, X):
        """
        Estimate the average disatnce between points which will be used as initial new cluster radius.
        The distance is estimated by taking a random (random_state) subset of points (n_points) and average the distance btween them.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        Returns
        ---------------
        R: float
            The estimnated default radius.
        """
        
        data = X[np.random.choice(self.m, self.n_points, replace=False), :]
        tot = 0.
        for i in range(self.n_points-1):
            tot += ((((data[i+1:]-data[i])**2).sum(1))**.5).sum()
            
        return tot/((self.n_points-1)*(self.n_points)/2.)

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
            
        dist, ind = self.tree.query(X, k=1)
        return ind
        
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
                "random_state":str(self.random_state),
                "n_points":self.n_points,
                "chunksize":self.chunksize,
                "progress_bar":self.progress_bar,
                "contamination":self.contamination,
                "shuffle":self.shuffle}