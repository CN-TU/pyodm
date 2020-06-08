# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Description : Anomaly Detection experiments
# Author      : Fares Meghdouri
#
#******************************************************************************

import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
from pyodm import ODM
# ALL CORESETS ALGOITHMS ARE INCLUDED IN utils.py
from utils import *
from sklearn.utils import shuffle

########################################################################

SEED = 2020

datalist = ['put_your_data_here']
r_list = [0.5, 1, 5, 10]
scores_mapping = {'maxf1':0,
				  'adj_maxf1':1,
				  'Patn':2,
				  'adj_Patn':3,
				  'ap':4,
				  'adj_ap':5,
				  'auc':6,
				  'ext_time':7,
				  'knn_time':8}
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

########################################################################

for i, dataset in enumerate(datalist):

	print('>> ##################################################### <<')
	print('>> Dataset: {}'.format(dataset))

	X_train = np.load('../data/OTDT/{}_X_train.npy'.format(dataset))
	X_test = np.load('../data/OTDT/{}_X_test.npy'.format(dataset))

	X = np.concatenate((X_train, X_test))

	y_train = np.load('../data/OTDT/{}_y_train.npy'.format(dataset))
	y_test = np.load('../data/OTDT/{}_y_test.npy'.format(dataset))

	y = np.concatenate((y_train, y_test))

	# shuffle the same way, it's better
	X, y = shuffle(X, y, random_state=SEED)

	# full data ####################################
	if BASELINE_FLAG:
		variant = 'BASELINE'
		print('>> {}, variant: {}'.format(dataset, variant))
		try:
			startTime = datetime.now()
			pred = get_out_score(X, X)
			final = (datetime.now() - startTime).total_seconds()
			inds = get_indices(y, pred)
			BASELINE_RES = fill_table(BASELINE_RES, inds, 0, 0, final)

		except Exception as e:
			print('an issue with {}, rate: 100, variant: {}'.format(dataset, variant))
			print(e)

	for j, rate in enumerate(r_list):

		print('>> Rate: {}'.format(rate))
		dataset_size = X.shape[0]
		coreset_size = int(dataset_size*rate/100)
		print(coreset_size)
		# ODM ####################################
		if ODM_FLAG:
			variant = 'ODM'
			print('>> {}, rate: {}, variant: {}'.format(dataset, rate, variant))
			try:
				startTime = datetime.now()
				model = ODM(m=coreset_size, random_state=SEED, shuffle_data=False, n_cores=1)
				model.fit(X)
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				pred = get_out_score(X, model.observers)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices(y, pred)
				ODM_RES = fill_table(ODM_RES, inds, j, interm, final)

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
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				pred = get_out_score(X, C_u)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices(y, pred)
				RS_RES = fill_table(RS_RES, inds, j, interm, final)

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
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				pred = get_out_score(X, model.observers)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices(y, pred)
				SDO_RES = fill_table(SDO_RES, inds, j, interm, final)

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
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				pred = get_out_score(X, model.cluster_centers_)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices(y, pred)
				KMC_RES = fill_table(KMC_RES, inds, j, interm, final)

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
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				pred = get_out_score(X, model.means_)
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices(y, pred)
				WGM_RES = fill_table(WGM_RES, inds, j, interm, final)

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
				interm = (datetime.now() - startTime).total_seconds()
				startTime = datetime.now()
				pred = get_out_score(X, np.array([x.weight[0] for x in gng.graph.nodes]))
				final = (datetime.now() - startTime).total_seconds()
				inds = get_indices(y, pred)
				GNG_RES = fill_table(GNG_RES, inds, j, interm, final)

			except Exception as e:
				print('an issue with {}, rate: {}, variant: {}'.format(dataset, rate, variant))
				print(e)

		# save results
		np.save('../results/anomaly_detection/BASELINE_RES', BASELINE_RES)
		np.save('../results/anomaly_detection/ODM_RES', ODM_RES)
		np.save('../results/anomaly_detection/RS_RES', RS_RES)
		np.save('../results/anomaly_detection/KMC_RES', KMC_RES)
		np.save('../results/anomaly_detection/WGM_RES', WGM_RES)
		np.save('../results/anomaly_detection/GNG_RES', GNG_RES)
		np.save('../results/anomaly_detection/SDO_RES', SDO_RES)
		np.save('../results/anomaly_detection/CNN_RES', CNN_RES)