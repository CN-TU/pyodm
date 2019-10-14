# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2019, Institute of Telecommunications, TU Wien
#
# Name        : __init__.py
# Description : Data Modeling and Compression algorithm
# Author      : Fares Meghdouri
#
# Notes : 
#
#
#******************************************************************************

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

#%matplotlib inline
import matplotlib.pyplot as plt

class DMC(object):
    """Data Modeling and Compression"""
    # TODO: implement different distances, currently DMC supports only the eucledian distance.
    
    def __init__(self, threshold=None, O=5, factor=0.1, contamination=0.1, n_points=100, random_state=None, chunksize=None, progress_bar=True):
        """
        Parameters
        ----------
        threshold: float, optional
            Default new cluster radius.
            If None, the default radius is estimated based on the mean distance of a set of randomely chosen points
            
        O: int, optional (default=1)
            Number of closest centroides to use for estimating the outlierness.
            
        factor: float, optional (default=0.1)
            A factor by wich the radius of a cluster is adjusted.
        
        contamination: float, optional (default=0.1)
            Ratio of outliers in data set. Only used with predict()
            
        n_points: int, optional (default=100)
            Number of data points to use for estimating the initial radius of a new cluster.
            
        progress_bar: bool, optional (default=True)
            Use a tqdm progress bar to observe the progress when dealing with large data.
            
        random_state: RandomState, int or None, optional (default=None)
            If np.RandomState, use random_state directly. If int, use
            random_state as seed value. If None, use default random
            state.
            
        chunksize: int, optional (default=None)
            Process data in chunks of size chunksize.
            If None, the whole data is taken at once. 
        """
        
        if threshold:
            self.estim_threshold = False
            self.threshold = threshold
        else:
            self.estim_threshold = True
            self.threshold = 'dynamic'
            
        self.O = O
        self.factor = factor
        self.random_state = random_state
        self.n_points = n_points
        self.chunksize = chunksize
        self.progress_bar = not progress_bar
        self.contamination = contamination
        
        return print('Initiating a DMC. Threshold = {}, 0 = {} and factor = {}'.format(self.threshold, self.O, self.factor))

    def fit(self, X):
        """
        Train a new model based on X.

        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
        """
            
        [self.m, self.n] = X.shape
        
        self.centroides = np.empty((1, self.n))
        self.counts = np.empty(1)
        self.radius = np.empty(1)
        
        if self.estim_threshold:
            # either take 100% or 50%
            self.threshold = self.estimate_threshold(X)/2
            print('Estimated threshold = {}'.format(self.threshold))
        
        if not self.chunksize:
            self.chunksize = self.m

        for batch_n, batch in enumerate(tqdm(np.array_split(X, self.m/self.chunksize), disable=self.progress_bar)):
            
            [self.mm, self.nn] = batch.shape
            shift = self.mm*batch_n
            
            sub_counts = np.zeros([self.mm, 1])
            sub_centroides = np.empty([self.mm, self.nn])
            sub_centroides[0] = batch[0]
            sub_counts[0] += 1
            sub_radius = np.full([self.mm, 1], self.threshold)
            
            i = 0
            
            for index, data_point in enumerate(batch[1:]):

                found = False
                tree = cKDTree(sub_centroides[:i+1])
                
                dist, ind = tree.query(data_point.reshape(1, -1), k=1)

                if dist <= sub_radius[ind]:
                    # TODO: check not only the nearest one...
                    distances = np.subtract(data_point, sub_centroides[ind])
                    sub_centroides[ind] = sub_centroides[ind] + distances/(sub_counts[ind] + 1)
                    sub_counts[ind] +=1
                    sub_radius[ind] = sub_radius[ind] - self.factor * dist#np.linalg.norm(distances)
                    found = True
                else:
                    sub_radius[ind] = sub_radius[ind] + self.factor * dist#np.linalg.norm(distances)
        
                if not found:
                    i +=1
                    sub_centroides[i] = data_point
                    sub_counts[i] += 1
                    sub_radius[i] = self.threshold
                
                #sub_centroides = np.delete(sub_centroides, list(set(to_delete)), 0)
                #plt.scatter(sub_centroides[:i,0], sub_centroides[:i,1])
                #display.clear_output(wait=True)
                #display.display(pl.gcf())
                #time.sleep(0.01)
    
            self.centroides = np.vstack((self.centroides, sub_centroides[:i+1]))
            self.counts = np.vstack((self.counts, sub_counts[:i+1]))
            self.radius = np.vstack((self.radius, sub_radius[:i+1]))
            
        self.centroides = np.delete(self.centroides, 0, axis=0)
        self.counts = np.delete(self.counts, 0, axis=0)
        self.radius = np.delete(self.radius, 0, axis=0)
            
        return self.centroides

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
            
        out = np.empty(self.m)
        tree = cKDTree(self.centroides)
        
        for index, data_point in enumerate(X):
            dist, ind = tree.query(data_point.reshape(1, -1), k=self.O)
            
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

    def estimate_threshold(self, X):
        """
        Estimate the average disatnce between points which will be used as initial new cluster radius.
        The distance is estimated by taking a random (random_state) subset of points (n_points) and average the distance btween them.
        
        Parameters
        ---------------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        Returns
        ---------------
        threshold: float
            The estimnated default radius.
        """
        
        np.random.seed(self.random_state)
        data = X[np.random.choice(self.m, self.n_points, replace=False), :]
        tot = 0.
        for i in range(self.n_points-1):
            tot += ((((data[i+1:]-data[i])**2).sum(1))**.5).sum()
            
        return tot/((self.n_points-1)*(self.n_points)/2.)
    
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
        
        return {"threshold":self.threshold,
                "O":self.O,
                "factor":self.factor,
                "random_state":str(self.random_state),
                "n_points":self.n_points,
                "chunksize":self.chunksize,
                "progress_bar":self.progress_bar,
                "contamination":self.contamination}
    