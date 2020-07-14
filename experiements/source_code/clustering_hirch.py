from __future__ import print_function, absolute_import, division
import pandas as pd
import numpy as np
from evolutionary_search import maximize
import collections
from sklearn.metrics import f1_score
import json
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial import cKDTree
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms
import abc
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_array, check_random_state
import neupy
from neupy import algorithms
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.utils import check_array, check_random_state
from datetime import datetime
import math
import numpy as np
import scipy.spatial.distance as distance
import multiprocessing as mp
import multiprocessing
import ctypes
from joblib import Parallel, delayed
from pyodm import ODM
from streamod import MTree
from pyod.models.knn import KNN
from utils import *
from sklearn.utils import shuffle
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid

def fill_table(output, scores, rate, time_ext, time_kmeans, time_extender):
	for key, value in scores_mapping.items():
		if key == 'rand':
			output[i, value, rate] = scores
		else:
			if key == 'ext_time':
				output[i, value, rate] = time_ext
			if key == 'kmeans':
				output[i, value, rate] = time_kmeans
			if key == 'label_extender':
				output[i, value, rate] = time_extender
	#print('>>>> ', output[:,:,0])
	print('>>>> everything is fine')
	return output

########################################################################
SEED = 2020

datalist = range(15)
r_list = [0.5, 1, 5, 10]
scores_mapping = {'rand':0,
				  'ext_time':1,
				  'kmeans':2,
				  'label_extender':3}
scores_cols = len(scores_mapping.keys())

BASELINE_FLAG = True
ODM_FLAG = True
RS_FLAG = True
KMC_FLAG = True
WGM_FLAG = True
GNG_FLAG = True
SDO_FLAG = True
CNN_FLAG = True

BASELINE_RES = np.zeros((len(datalist), scores_cols, 1))
ODM_RES = np.zeros((len(datalist), scores_cols, len(r_list)))
RS_RES = np.zeros((len(datalist), scores_cols, len(r_list)))
KMC_RES = np.zeros((len(datalist), scores_cols, len(r_list)))
WGM_RES = np.zeros((len(datalist), scores_cols, len(r_list)))
GNG_RES = np.zeros((len(datalist), scores_cols, len(r_list)))
SDO_RES = np.zeros((len(datalist), scores_cols, len(r_list)))
CNN_RES = np.zeros((len(datalist), scores_cols, len(r_list)))

########################################################################


for i, dataset in enumerate(datalist):

	print('>> ##################################################### <<')
	print('>> Dataset: {}'.format(dataset))

	X = np.load('../data/MDCG/Dataset_{}.npy'.format(dataset))

	y = np.load('../data/MDCG/Labels_{}.npy'.format(dataset))

	# shuffle the same way, it's better
	X, y = shuffle(X, y, random_state=SEED)

	n_centroides = np.unique(y).shape[0]

	# full data ####################################
	if BASELINE_FLAG:
		variant = 'BASELINE'
		print('>> {}, variant: {}'.format(dataset, variant))
		try:
			startTime = datetime.now()
			clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
			y_predict = clustering_model.fit_predict(X)
			clf = NearestCentroid()
			clf.fit(X, y_predict)
			_centers = clf.centroids_
			interm = (datetime.now() - startTime).total_seconds()
			tmp_model = KNeighborsClassifier(n_neighbors=1)
			tmp_model.fit(X, y)
			coreset_labels = tmp_model.predict(_centers)
			startTime = datetime.now()
			label_extender = KNeighborsClassifier(n_neighbors=1)
			pred = label_extender.fit(_centers, coreset_labels).predict(X)
			lextender = (datetime.now() - startTime).total_seconds()
			inds = adjusted_rand_score(y, pred)		
			BASELINE_RES = fill_table(BASELINE_RES, inds, 0, 0, interm,  lextender)

		except Exception as e:
			print('an issue with {}, rate: 100, variant: {}'.format(dataset, variant))
			print(e)

	for j, rate in enumerate(r_list):

		print('>> Rate: {}'.format(rate))
		dataset_size = X.shape[0]
		coreset_size = int(dataset_size*rate/100)
		# ODM ####################################
		if ODM_FLAG:
			variant = 'ODM'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				model = ODM(m=coreset_size, random_state=SEED, shuffle_data=False, n_cores=1)
				model.fit(X)
				ext = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
				y_predict = clustering_model.fit_predict(model.observers)
				clf = NearestCentroid()
				clf.fit(model.observers, y_predict)
				_centers = clf.centroids_
				interm = (datetime.now() - startTime).total_seconds()
				coreset_labels = tmp_model.predict(_centers)
				startTime = datetime.now()
				label_extender = KNeighborsClassifier(n_neighbors=1)
				l_model = label_extender.fit(_centers, coreset_labels)
				pred = l_model.predict(X)
				lextender = (datetime.now() - startTime).total_seconds()
				inds = adjusted_rand_score(y, pred)		
				ODM_RES = fill_table(ODM_RES, inds, j, ext, interm, lextender)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# random sampling/ KMeansUniformCoreset ##
		if RS_FLAG:
			variant = 'RS'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				model = KMeansUniformCoreset(X)
				C_u, w_u = model.generate_coreset(coreset_size)
				ext = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
				y_predict = clustering_model.fit_predict(C_u)
				clf = NearestCentroid()
				clf.fit(C_u, y_predict)
				_centers = clf.centroids_
				interm = (datetime.now() - startTime).total_seconds()
				coreset_labels = tmp_model.predict(_centers)
				startTime = datetime.now()
				label_extender = KNeighborsClassifier(n_neighbors=1)
				l_model = label_extender.fit(_centers, coreset_labels)
				pred = l_model.predict(X)
				lextender = (datetime.now() - startTime).total_seconds()
				inds = adjusted_rand_score(y, pred)		
				RS_RES = fill_table(RS_RES, inds, j, ext, interm, lextender)
			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# SDO + drop idle #########################
		if SDO_FLAG:
			variant = 'SDO'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				model = SDO(k = coreset_size, random_state=SEED)
				model.fit(X)
				ext = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
				y_predict = clustering_model.fit_predict(model.observers)
				clf = NearestCentroid()
				clf.fit(model.observers, y_predict)
				_centers = clf.centroids_
				interm = (datetime.now() - startTime).total_seconds()
				coreset_labels = tmp_model.predict(_centers)
				startTime = datetime.now()
				label_extender = KNeighborsClassifier(n_neighbors=1)
				l_model = label_extender.fit(_centers, coreset_labels)
				pred = l_model.predict(X)
				lextender = (datetime.now() - startTime).total_seconds()
				inds = adjusted_rand_score(y, pred)		
				SDO_RES = fill_table(SDO_RES, inds, j, ext, interm, lextender)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# KMeansCoreset ############################
		if KMC_FLAG:
			variant = 'KMC'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				model = KMeans(n_clusters=coreset_size, random_state=SEED, n_jobs=2)
				model.fit(X)
				ext = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
				y_predict = clustering_model.fit_predict(model.cluster_centers_)
				clf = NearestCentroid()
				clf.fit(model.cluster_centers_, y_predict)
				_centers = clf.centroids_
				interm = (datetime.now() - startTime).total_seconds()
				coreset_labels = tmp_model.predict(_centers)
				startTime = datetime.now()
				label_extender = KNeighborsClassifier(n_neighbors=1)
				l_model = label_extender.fit(_centers, coreset_labels)
				pred = l_model.predict(X)
				lextender = (datetime.now() - startTime).total_seconds()
				inds = adjusted_rand_score(y, pred)		
				KMC_RES = fill_table(KMC_RES, inds, j, ext, interm, lextender)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# WeightedBayesianGaussianMixture ##########
		if WGM_FLAG:
			variant = 'WBGM'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				model = WeightedBayesianGaussianMixture(n_components=coreset_size)
				model.fit(X)
				ext = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
				y_predict = clustering_model.fit_predict(model.means_)
				clf = NearestCentroid()
				clf.fit(model.means_, y_predict)
				_centers = clf.centroids_
				interm = (datetime.now() - startTime).total_seconds()
				coreset_labels = tmp_model.predict(_centers)
				startTime = datetime.now()
				label_extender = KNeighborsClassifier(n_neighbors=1)
				l_model = label_extender.fit(_centers, coreset_labels)
				pred = l_model.predict(X)
				lextender = (datetime.now() - startTime).total_seconds()
				inds = adjusted_rand_score(y, pred)		
				WGM_RES = fill_table(WGM_RES, inds, j, ext, interm, lextender)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# GNG ######################################
		if GNG_FLAG:
			variant = 'GNG'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				gng = create_gng(max_nodes=coreset_size, n_inputs=X.shape[1])

				for epoch in range(40):
					gng.train(X, epochs=1)
				ext = (datetime.now() - startTime).total_seconds()
				gngob = np.array([x.weight[0] for x in gng.graph.nodes])
				startTime = datetime.now()
				clustering_model = AgglomerativeClustering(n_clusters=n_centroides)
				y_predict = clustering_model.fit_predict(gngob)
				clf = NearestCentroid()
				clf.fit(gngob, y_predict)
				_centers = clf.centroids_
				interm = (datetime.now() - startTime).total_seconds()
				coreset_labels = tmp_model.predict(_centers)
				startTime = datetime.now()
				label_extender = KNeighborsClassifier(n_neighbors=1)
				l_model = label_extender.fit(_centers, coreset_labels)
				pred = l_model.predict(X)
				lextender = (datetime.now() - startTime).total_seconds()
				inds = adjusted_rand_score(y, pred)		
				GNG_RES = fill_table(GNG_RES, inds, j, ext, interm, lextender)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		np.save('../results/clustering_hirch/BASELINE_RES', BASELINE_RES)
		np.save('../results/clustering_hirch/ODM_RES', ODM_RES)
		np.save('../results/clustering_hirch/RS_RES', RS_RES)
		np.save('../results/clustering_hirch/KMC_RES', KMC_RES)
		np.save('../results/clustering_hirch/WGM_RES', WGM_RES)
		np.save('../results/clustering_hirch/GNG_RES', GNG_RES)
		np.save('../results/clustering_hirch/SDO_RES', SDO_RES)
		np.save('../results/clustering_hirch/CNN_RES', CNN_RES)