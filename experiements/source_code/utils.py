# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : A collection of algorithms and tools used by other scripts
# Author      : Fares Meghdouri
#
#******************************************************************************

import numpy as np
import collections
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms
import abc
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_array, check_random_state
from neupy import algorithms
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import scipy.spatial.distance as distance
import multiprocessing as mp
import ctypes
from joblib import Parallel, delayed
from pyod.models.knn import KNN

########################################################################

def get_out_score(X, observers):

	print('>> internal check : r = ', observers.shape[0]/X.shape[0]*100)
	clf = KNN()
	clf.fit(observers)
	return clf.decision_function(X)

def get_indices(y_true, scores):
	m = y_true.size
	num_outliers = np.sum(y_true)
	res = {}
	
	perm = np.argsort(scores)[::-1]
	scores_s = scores[perm]
	y_true_s = y_true[perm]

	# P@n
	try:
		res['Patn'] = np.sum(y_true_s[:num_outliers]) / num_outliers
	except:
		res['Patn'] = 0
	try:
		res['adj_Patn'] = (res['Patn'] - num_outliers/m) / (1 - num_outliers/m)
	except:
		res['adj_Patn'] = 0

	y_true_cs = np.cumsum(y_true_s[:])

	# average precision
	try:
		res['ap'] = np.sum( y_true_cs[:num_outliers] / np.arange(1, num_outliers + 1) ) / num_outliers
	except:
		res['ap'] = 0
	try:
		res['adj_ap'] = (res['ap'] - num_outliers/m) / (1 - num_outliers/m)
	except:
		res['adj_ap'] = 0

	# Max. F1 score
	try:
		res['maxf1'] = 2 * np.max(y_true_cs[:m] / np.arange(1 + num_outliers, m + 1 + num_outliers))
	except:
		res['maxf1'] = 0
	try:
		res['adj_maxf1'] = (res['maxf1'] - num_outliers/m) / (1 - num_outliers/m)
	except:
		res['adj_maxf1'] = 0

	 # ROC-AUC
	try:
		res['auc'] = roc_auc_score(y_true, scores)
	except:
		res['auc'] = 0

	return res

def get_indices_(y_true, y_pred):
	res = {}
	try:
		res['accuracy'] = accuracy_score(y_true, y_pred)
	except:
		res['accuracy'] = 0

	try:
		res['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division = 0)
	except:
		res['micro_precision'] = 0

	try:
		res['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division = 0)
	except:
		res['macro_precision'] = 0

	try:
		res['micro_recall'] = recall_score(y_true, y_pred, average='micro')
	except:
		res['micro_recall'] = 0

	try:
		res['macro_recall'] = recall_score(y_true, y_pred, average='macro')
	except:
		res['macro_recall'] = 0	

	return res

def create_gng(max_nodes, n_inputs, step=0.2, n_start_nodes=2, max_edge_age=50):
	return algorithms.GrowingNeuralGas(
		n_inputs=n_inputs,
		n_start_nodes=n_start_nodes,

		shuffle_data=True,
		verbose=False,

		step=step,
		neighbour_step=0.005,

		max_edge_age=max_edge_age,
		max_nodes=max_nodes,

		n_iter_before_neuron_added=100,
		after_split_error_decay_rate=0.5,
		error_decay_rate=0.995,
		min_distance_for_update=0.01,
	)

class Coreset(object):
	"""
	Abstract class for coresets.
	Parameters
	----------
	X : ndarray, shape (n_points, n_dims)
		The data set to generate coreset from.
	w : ndarray, shape (n_points), optional
		The weights of the data points. This allows generating coresets from a
		weighted data set, for example generating coreset of a coreset. If None,
		the data is treated as unweighted and w will be replaced by all ones array.
	random_state : int, RandomState instance or None, optional (default=None)
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, X, w=None, random_state=None):
		X = check_array(X, accept_sparse="csr", order='C',
						dtype=[np.float64, np.float32])
		self.X = X
		self.w = w if w is not None else np.ones(X.shape[0])
		self.n_samples = X.shape[0]
		self.random_state = check_random_state(random_state)
		self.calc_sampling_distribution()

	@abc.abstractmethod
	def calc_sampling_distribution(self):
		"""
		Calculates the coreset importance sampling distribution.
		"""
		pass

	def generate_coreset(self, size):
		"""
		Generates a coreset of the data set.
		Parameters
		----------
		size : int
			The size of the coreset to generate.
		"""
		ind = np.random.choice(self.n_samples, size=size, p=self.p)
		return self.X[ind], 1. / (size * self.p[ind])

def _launchParallel(func, n_jobs):
	if n_jobs is None:
		n_jobs = 1
	elif n_jobs < 0:
		n_jobs = mp.cpu_count()
	
	if n_jobs == 1:
		func()
	else:
		processes = []
		for i in range(n_jobs):
			processes.append( mp.Process(target=func))
			processes[-1].start()
		for i in range(n_jobs):
			processes[i].join()

class SDO:
	"""Outlier detection based on Sparse Data Observers"""
	
	def __init__(self, k=None, q=None, qv=0.3, x=6, hbs=False, return_scores = False, contamination=0.1, metric='euclidean', random_state=None, chunksize=1000, n_jobs=None):
		"""
		Parameters
		----------
		k: int, optional
			Number of observers. If None, estimate the number of
			observers using Principal Component Analysis (PCA).
			
		q: int, optional
			Threshold for observed points for model cleaning.
			If None, use qv instead.
			
		qv: float (default=0.3)
			Ratio of removed observers due to model cleaning.
			Only used if q is None.
			
		x: int, optional (default=6)
			Number of nearest observers to consider for outlier scoring
			and model cleaning.
			
		hbs: bool (default=False)
			Whether to use histogram-based sampling.
			
		return_scores: bool (default=False)
			Return outlier scores instead of binary labels.
			
		contamination: float (default=0.1)
			Ratio of outliers in data set. Only used if
			return_scores is False.
			
		metric: string or DistanceMetric (default='euclidean')
			Distance metric to use. Can be a string or distance metric
			as understood by sklearn.neighbors.DistanceMetric.
			
		random_state: RandomState, int or None (default=None)
			If np.RandomState, use random_state directly. If int, use
			random_state as seed value. If None, use default random
			state.
			
		chunksize: int (default=1000)
			Process data in chunks of size chunksize. chunksize has no
			influence on the algorithm's outcome.
			
		n_jobs: int (default=-1)
			Spawn n_jobs threads to process the data. Pass -1 to use as
			many threads as there are CPU cores. n_jobs has no influence
			on the algorithm's outcome.
		"""
		
		if isinstance(k, np.ndarray):
			# ~ self.observers = k.copy()
			self.model = KDTree(k, leaf_size=100, metric=metric)
			self.k = k.shape[0]
		else:
			self.k = k
		self.use_pca = k is None
		self.q = q
		self.qv = qv
		self.x = x
		self.hbs = hbs
		self.return_scores = return_scores
		self.contamination = contamination
		self.metric = metric
		self.random_state = random_state
		self.chunksize = chunksize
		self.n_jobs = n_jobs
		
	def fit(self, X):
		"""
		Train a new model based on X.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
		"""
		
		random = check_random_state(self.random_state)
		X = check_array(X, accept_sparse=['csc'])
		[m, n] = X.shape
		
		if self.use_pca:
			# choose number of observers as described in paper
			pca = PCA()
			pca.fit(X)
			var = max(1,pca.explained_variance_[0])
			sqerr = 0.01 * pca.explained_variance_[0]
			Z = 1.96
			self.k = int((m * Z**2 * var) // ((m-1) * sqerr + Z**2 * var))
			
		if self.hbs:
			Y = X.copy()
			binning_param = 20
			for i in range(n):
				dimMin = min(Y[:,i])
				dimMax = max(Y[:,i])
				if dimMax > dimMin:
					binWidth = (dimMax - dimMin) / round(math.log10(self.k) * binning_param)
					Y[:,i] = (np.floor( (Y[:,i] - dimMin) / binWidth) + .5) * binWidth + dimMin
			Y = np.unique(Y, axis=0)
		else:
			Y = X

		# sample observers randomly from dataset
		observers = Y[random.choice(Y.shape[0], self.k),:]
			
		# copy for efficient cache usage
		#observers = observers.copy()
		model = KDTree(observers, metric=self.metric)
			
		globalI = mp.Value('i', 0)
		
		P = np.frombuffer( mp.Array(ctypes.c_double, self.k).get_obj() )
		P[:] = 0

		def TrainWorker():
			thisP = np.zeros(self.k)
			while True:
				with globalI.get_lock():
					P[:] += thisP[:]
					i = globalI.value
					globalI.value += self.chunksize
				if i >= m: return
				#closest = model.query(X[i:(i+self.chunksize)], return_distance=False, k=self.x).flatten()
				dist = distance.cdist(X[i:(i+self.chunksize)], observers, self.metric)
				dist_sorted = np.argsort(dist, axis=1)
				closest = dist_sorted[:,0:self.x].flatten()

				thisP = np.sum (closest[:,None] == np.arange(self.k), axis=0)
				
		_launchParallel(TrainWorker, self.n_jobs)
	
		q = np.quantile(P, self.qv) if self.q is None else self.q
		#self.observers = observers[P>=q].copy()
		self.model = KDTree(observers[P>=q], metric=self.metric)
		
	def predict(self, X):
		"""
		Only perform outlier detection based on a trained model.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
					
		Returns
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for passed input data if return_scores==True,
			otherwise binary outlier labels.
		"""
		
		X = check_array(X, accept_sparse=['csc'])
		[m, n] = X.shape
		
		scores = np.frombuffer( mp.Array(ctypes.c_double, m).get_obj() )
		globalI = mp.Value('i', 0)
		
		def AppWorker():
			while True:
				with globalI.get_lock():
					i = globalI.value
					globalI.value += self.chunksize
				if i >= m: return
				dist_sorted,_ = self.model.query(X[i:(i+self.chunksize)], return_distance=True, k=self.x)
				scores[i:(i+self.chunksize)] = np.median(dist_sorted, axis=1)
				#dist = distance.cdist(X[i:(i+self.chunksize)], self.observers, self.metric)
				#dist_sorted = np.sort(dist, axis=1)
				#scores[i:(i+self.chunksize)] = np.median(dist_sorted[:,0:self.x], axis=1)
				
		_launchParallel(AppWorker, self.n_jobs)

		if self.return_scores:
			return scores
			
		threshold = np.quantile(scores, 1-self.contamination)
		
		return scores > threshold
		
	def fit_predict(self, X):
		"""
		Train a new model based on X and find outliers in X.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
					
		Returns
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for passed input data if return_scores==True,
			otherwise binary outlier labels.
		"""
		self.fit(X)
		return self.predict(X)

	def get_params(self, deep=True):
		"""
		Return the model's parameters.
		Parameters
		---------------
		deep : bool, optional (default=True)
			Return sub-models parameters.
		Returns
		---------------
		params: dict, shape (n_parameters,)
			A dictionnary mapping of the model's parameters.
		"""
		return {"k":None if self.use_pca else self.k,
			"q":self.q,
			"qv":self.qv,
			"x":self.x,
			"hbs":self.hbs,
			"return_scores":self.return_scores,
			"contamination":self.contamination,
			"metric":self.metric,
			"random_state":str(self.random_state),
			"chunksize":self.chunksize,
			"n_jobs":self.n_jobs}

	@property
	def observers(self):
		"""
		Gets or sets the trained model represented as a matrix
		of active observers.
		"""
		assert self.model is not None, 'Model has not been trained yet'
		return np.asarray(self.model.data)
		
	@observers.setter
	def observers(self, new_observers):
		self.model = KDTree(new_observers, leaf_size=100, metric=self.metric)
		self.k = new_observers.shape[0]

class WeightedBayesianGaussianMixture(BayesianGaussianMixture):
	"""
	Extends sklearn.mixture.BayesianGaussianMixture to support weighted data set.
	Its methods and attributes are identical to the parent's, except for
	the fit() method.
	Parameters
	----------
		See sklearn.mixture.BayesianGaussianMixture
	"""

	def __init__(self, n_components=1, covariance_type='full', tol=1e-5,
				 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
				 weight_concentration_prior_type='dirichlet_process',
				 weight_concentration_prior=None,
				 mean_precision_prior=None, mean_prior=None,
				 degrees_of_freedom_prior=None, covariance_prior=None,
				 random_state=None, warm_start=False, verbose=0,
				 verbose_interval=10):
		super(WeightedBayesianGaussianMixture, self).__init__(
			n_components=n_components, covariance_type=covariance_type, tol=tol,
			reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
			weight_concentration_prior_type=weight_concentration_prior_type,
			weight_concentration_prior=weight_concentration_prior,
			mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
			degrees_of_freedom_prior=degrees_of_freedom_prior, covariance_prior=covariance_prior,
			random_state=random_state, warm_start=warm_start, verbose=verbose,
			verbose_interval=verbose_interval)
		self.weights = None

	def _compute_lower_bound(self, log_resp, log_prob_norm):
		return super(WeightedBayesianGaussianMixture, self)._compute_lower_bound(log_resp, log_prob_norm) \
			   + np.sum(np.exp(log_resp) * (1 - np.exp(self.log_weight_mat)) * log_resp)

	def _initialize(self, X, resp):
		self.weight_mat = self.weights.repeat(self.n_components).reshape(X.shape[0], self.n_components)
		self.log_weight_mat = np.log(self.weight_mat)
		resp_w = resp * self.weight_mat
		super(WeightedBayesianGaussianMixture, self)._initialize(X, resp_w)

	def fit(self, X, weights=None, y=None):
		if weights is None:
			weights = np.ones(X.shape[0])
		if X.shape[0] != weights.shape[0]:
			raise ValueError("The number of weights must match the number of data points.")
		self.weights = weights
		super(WeightedBayesianGaussianMixture, self).fit(X, y)

	def _e_step(self, X):
		log_prob_norm, log_responsibility = super(WeightedBayesianGaussianMixture, self)._e_step(X)
		return log_prob_norm, log_responsibility + self.log_weight_mat

class KMeansCoreset(Coreset):
	"""
	Class for generating k-Means coreset based on the sensitivity framework
	with importance sampling [1].
	Parameters
	----------
		X : ndarray, shape (n_points, n_dims)
			The data set to generate coreset from.
		w : ndarray, shape (n_points), optional
			The weights of the data points. This allows generating coresets from a
			weighted data set, for example generating coreset of a coreset. If None,
			the data is treated as unweighted and w will be replaced by all ones array.
		n_clusters : int
			Number of clusters used for the initialization step.
		init : for avaiable types, please refer to sklearn.cluster.k_means_._init_centroids
			Method for initialization
		random_state : int, RandomState instance or None, optional (default=None)
	References
	----------
		[1] Bachem, O., Lucic, M., & Krause, A. (2017). Practical coreset constructions
		for machine learning. arXiv preprint arXiv:1703.06476.
	"""

	def __init__(self, X, w=None, n_clusters=10, init="k-means++", random_state=None):
		self.n_clusters = n_clusters
		self.init = init
		super(KMeansCoreset, self).__init__(X, w, random_state)

	def calc_sampling_distribution(self):
		x_squared_norms = row_norms(self.X, squared=True)
		centers = _init_centroids(self.X, self.n_clusters, self.init, random_state=self.random_state,
								  x_squared_norms=x_squared_norms)
		sens = sensitivity.kmeans_sensitivity(self.X, self.w, centers, max(np.log(self.n_clusters), 1))
		self.p = sens / np.sum(sens)

class KMeansLightweightCoreset(Coreset):
	"""
	   Class for generating k-Means coreset based on the importance sampling scheme of [1]
	   Parameters
	   ----------
		   X : ndarray, shape (n_points, n_dims)
			   The data set to generate coreset from.
		   w : ndarray, shape (n_points), optional
			   The weights of the data points. This allows generating coresets from a
			   weighted data set, for example generating coreset of a coreset. If None,
			   the data is treated as unweighted and w will be replaced by all ones array.
		   random_state : int, RandomState instance or None, optional (default=None)
	   References
	   ----------
		   [1] Bachem, O., Lucic, M., & Krause, A. (2017). Scalable and distributed
		   clustering via lightweight coresets. arXiv preprint arXiv:1702.08248.
	   """

	def __init__(self, X, w=None, random_state=None):
		super(KMeansLightweightCoreset, self).__init__(X, w, random_state)

	def calc_sampling_distribution(self):
		weighted_data = self.X * self.w[:, np.newaxis]
		data_mean = np.sum(weighted_data, axis=0) / np.sum(self.w)
		self.p = np.sum((weighted_data - data_mean[np.newaxis, :]) ** 2, axis=1) * self.w
		if np.sum(self.p) > 0:
			self.p = self.p / np.sum(self.p) * 0.5 + 0.5 / np.sum(self.w)
		else:
			self.p = np.ones(self.n_samples) / np.sum(self.w)

		# normalize in order to avoid numerical errors
		self.p /= np.sum(self.p)

class KMeansUniformCoreset(Coreset):
	"""
	   Class for generating uniform subsamples of the data.
	   Parameters
	   ----------
		   X : ndarray, shape (n_points, n_dims)
			   The data set to generate coreset from.
		   w : ndarray, shape (n_points), optional
			   The weights of the data points. This allows generating coresets from a
			   weighted data set, for example generating coreset of a coreset. If None,
			   the data is treated as unweighted and w will be replaced by all ones array.
		   random_state : int, RandomState instance or None, optional (default=None)
	   References
	   ----------
		   [1] Bachem, O., Lucic, M., & Krause, A. (2017). Scalable and distributed
		   clustering via lightweight coresets. arXiv preprint arXiv:1702.08248.
	   """

	def __init__(self, X, w=None, random_state=None):
		super(KMeansUniformCoreset, self).__init__(X, w, random_state)

	def calc_sampling_distribution(self):
		self.p = self.w
		self.p /= np.sum(self.p)