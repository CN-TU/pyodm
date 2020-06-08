# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Classification experiments
# Author      : Fares Meghdouri
#
#******************************************************************************

import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
from pyodm import ODM
from utils import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
import random

########################################################################

SEED = 2020

datalist = ['put_your_data_here']
r_list = [0.5, 1, 5, 10]
scores_mapping = {'accuracy':0,
				  'micro_precision':1,
				  'macro_precision':2,
				  'micro_recall':3,
				  'macro_recall':4,
				  'ext_time':5,
				  'knn_time':6}
scores_cols = len(scores_mapping.keys())

# select what you want to extract/compare
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

def fill_table(output, scores, rate, time_interm, time_final):
	for key, value in scores_mapping.items():
		if key not in ['ext_time', 'knn_time']:
			output[i, value, rate] = scores[key]
		else:
			if key == 'ext_time':
				output[i, value, rate] = time_interm
			if key == 'knn_time':
				output[i, value, rate] = time_final
	#print('>>>> ', output[:,:,0])
	print('>>>> everything is fine')
	return output

def fix_rate(X, y, observers, total_labels, coreset_size):
	if observers.shape[0] > coreset_size:
		chosen_random = random.sample(range(0, observers.shape[0]), coreset_size)
		observers  = observers[chosen_random]
		total_labels  = total_labels[chosen_random]
		return observers, total_labels
	elif observers.shape[0] < coreset_size:
		append_random   = random.sample(range(0, X.shape[0]), coreset_size-observers.shape[0])
		observers  = np.vstack((observers, X[append_random]))
		total_labels  = np.concatenate([total_labels, y[append_random]])
		return observers, total_labels
	else:
		return observers, total_labels

########################################################################

for i, dataset in enumerate(datalist):

	print('>> ##################################################### <<')
	print('>> Dataset: {}'.format(dataset))

	X = np.load('../data/MCC/Dataset_{}.npy'.format(dataset), allow_pickle=True)
	y = np.load('../data/MCC/Labels_{}.npy'.format(dataset), allow_pickle=True)

	# shuffle the same way, it's better
	X, y = shuffle(X, y, random_state=SEED)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

	# full data ####################################
	if BASELINE_FLAG:
		variant = 'BASELINE'
		print('>> {}, variant: {}'.format(dataset, variant))
		try:
			startTime = datetime.now()
			neigh = KNeighborsClassifier(n_neighbors=5)
			neigh.fit(X_train, y_train)
			y_pred = neigh.predict(X_test)
			final = (datetime.now() - startTime).total_seconds()
			inds = get_indices_(y_test, y_pred)
			BASELINE_RES = fill_table(BASELINE_RES, inds, 0, 0, final)

		except Exception as e:
			print('an issue with {}, rate: 100, variant: {}'.format(dataset, variant))
			print(e)

	for j, rate in enumerate(r_list):

		print('>> Rate: {}'.format(rate))
		
		#print(coreset_size)
		# ODM ####################################
		if ODM_FLAG:
			variant = 'ODM'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				total_data = []
				total_labels = []
				startTime = datetime.now()
				for distinct_value in np.unique(y_train):
					position = np.where(y_train == distinct_value)[0]
					sub_dataset_size = position.shape[0]
					sub_coreset_size = max(int(sub_dataset_size*rate/100),1)
					model = ODM(m=sub_coreset_size, random_state=SEED, shuffle_data=False, n_cores=1)
					model.fit(X_train[position])
					total_data.append(model.observers)
					total_labels.extend([distinct_value for i in range(model.observers.shape[0])])
				interm = (datetime.now() - startTime).total_seconds()
				observers = np.vstack(total_data)
				startTime = datetime.now()
				neigh = KNeighborsClassifier(n_neighbors=5)
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				ODM_RES = fill_table(ODM_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# random sampling/ KMeansUniformCoreset ##
		if RS_FLAG:
			variant = 'RS'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				total_data = []
				total_labels = []
				startTime = datetime.now()
				for distinct_value in np.unique(y_train):
					position = np.where(y_train == distinct_value)[0]
					sub_dataset_size = position.shape[0]
					sub_coreset_size = max(int(sub_dataset_size*rate/100),1)
					model = KMeansUniformCoreset(X_train[position])
					C_u, w_u = model.generate_coreset(sub_coreset_size)
					total_data.append(C_u)
					total_labels.extend([distinct_value for i in range(C_u.shape[0])])
				interm = (datetime.now() - startTime).total_seconds()
				observers = np.vstack(total_data)
				startTime = datetime.now()
				neigh = KNeighborsClassifier(n_neighbors=5)
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				RS_RES = fill_table(RS_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# SDO + drop idle #########################
		if SDO_FLAG:
			variant = 'SDO'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				total_data = []
				total_labels = []
				startTime = datetime.now()
				for distinct_value in np.unique(y_train):
					position = np.where(y_train == distinct_value)[0]
					sub_dataset_size = position.shape[0]
					sub_coreset_size = max(int(sub_dataset_size*rate/100),1)
					model = SDO(k = sub_coreset_size, random_state=SEED)
					model.fit(X_train[position])
					total_data.append(model.observers)
					total_labels.extend([distinct_value for i in range(model.observers.shape[0])])
				interm = (datetime.now() - startTime).total_seconds()
				observers = np.vstack(total_data)
				startTime = datetime.now()
				neigh = KNeighborsClassifier(n_neighbors=5)
				observers[np.isnan(observers)] = 0
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				SDO_RES = fill_table(SDO_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# KMeansCoreset ############################
		if KMC_FLAG:
			variant = 'KMC'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				total_data = []
				total_labels = []
				startTime = datetime.now()
				for distinct_value in np.unique(y_train):
					position = np.where(y_train == distinct_value)[0]
					sub_dataset_size = position.shape[0]
					sub_coreset_size = max(int(sub_dataset_size*rate/100),1)
					model = KMeans(n_clusters=sub_coreset_size, random_state=SEED, n_jobs=2)
					model.fit(X_train[position])
					total_data.append(model.cluster_centers_)
					total_labels.extend([distinct_value for i in range(model.cluster_centers_.shape[0])])
				interm = (datetime.now() - startTime).total_seconds()
				observers = np.vstack(total_data)
				startTime = datetime.now()
				neigh = KNeighborsClassifier(n_neighbors=5)
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				KMC_RES = fill_table(KMC_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# WeightedBayesianGaussianMixture ##########
		if WGM_FLAG:
			variant = 'WBGM'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				total_data = []
				total_labels = []
				startTime = datetime.now()
				for distinct_value in np.unique(y_train):
					position = np.where(y_train == distinct_value)[0]
					sub_dataset_size = position.shape[0]
					sub_coreset_size = max(int(sub_dataset_size*rate/100),1)
					model = WeightedBayesianGaussianMixture(n_components=sub_coreset_size)
					model.fit(X_train[position])
					total_data.append(model.means_)
					total_labels.extend([distinct_value for i in range(model.means_.shape[0])])
				interm = (datetime.now() - startTime).total_seconds()
				observers = np.vstack(total_data)
				startTime = datetime.now()
				neigh = KNeighborsClassifier(n_neighbors=5)
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				WGM_RES = fill_table(WGM_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# GNG ######################################
		if GNG_FLAG:
			variant = 'GNG'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				total_data = []
				total_labels = []
				startTime = datetime.now()
				for distinct_value in np.unique(y_train):
					position = np.where(y_train == distinct_value)[0]
					sub_dataset_size = position.shape[0]
					sub_coreset_size = max(int(sub_dataset_size*rate/100),1)
					gng = create_gng(max_nodes=sub_coreset_size, n_inputs=X.shape[1])

					for epoch in range(40):
						gng.train(X_train[position], epochs=1)
					obsrvs = np.array([x.weight[0] for x in gng.graph.nodes])
					total_data.append(obsrvs)
					total_labels.extend([distinct_value for i in range(obsrvs.shape[0])])
				interm = (datetime.now() - startTime).total_seconds()
				observers = np.vstack(total_data)
				startTime = datetime.now()
				neigh = KNeighborsClassifier(n_neighbors=5)
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				GNG_RES = fill_table(GNG_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		if CNN_FLAG:
			variant = 'CNN'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				dataset_size = X_train.shape[0]
				coreset_size = max(int(dataset_size*rate/100),1)
				startTime = datetime.now()
				cnn = CondensedNearestNeighbour(random_state=SEED)
				observers, total_labels = cnn.fit_sample(X_train, y_train)
				observers, total_labels = fix_rate(X_train, y_train, observers, total_labels, coreset_size)
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				try:
					neigh = KNeighborsClassifier(n_neighbors=5)
				except:
					neigh = KNeighborsClassifier(n_neighbors=1)
				neigh.fit(observers, total_labels)
				y_pred = neigh.predict(X_test)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices_(y_test, y_pred)
				CNN_RES = fill_table(CNN_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# save results
		np.save('../results/supervised/BASELINE_RES', BASELINE_RES)
		np.save('../results/supervised/ODM_RES', ODM_RES)
		np.save('../results/supervised/RS_RES', RS_RES)
		np.save('../results/supervised/KMC_RES', KMC_RES)
		np.save('../results/supervised/WGM_RES', WGM_RES)
		np.save('../results/supervised/GNG_RES', GNG_RES)
		np.save('../results/supervised/SDO_RES', SDO_RES)
		np.save('../results/supervised/CNN_RES', CNN_RES)