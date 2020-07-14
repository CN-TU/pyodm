# plotting code for ODM paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def anomaly_detection_scores(outfile='anomaly_detection.pdf'):
	algos = ['RS', 'SDO', 'KMC', 'WGM', 'GNG', 'ODM']
	plt.figure(figsize=(16,8))
	sns.set()
	sns.set_palette('bright')
	sns.set(style="whitegrid")
	for i, algorithm in enumerate(algos):
	    data = np.load('../results/anomaly_detection/{}_RES.npy'.format(algorithm))
	    a = np.mean(data, axis=(0))
	    if algorithm == 'ODM':
	        sns.scatterplot(a[8,:], a[5,:], s=[500, 1000, 1500, 2500], label=algorithm, alpha=0.7, linewidth=2, linestyle='--', color = 'white', edgecolor='red')
	    else:
	        sns.scatterplot(a[8,:], a[5,:], s=[500, 1000, 1500, 2500], label=algorithm, alpha=0.7,)
	data = np.load('../results/anomaly_detection/{}_RES.npy'.format('BASELINE'))
	a = np.mean(data, axis=(0))
	sns.scatterplot(a[8,:], a[5,:], s=4500, color="black", edgecolor='black', linewidth=2, linestyle='--', label='BASELINE', alpha=0.8)
	#plt.xscale('log')
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.xlabel('kNN Time (s)', size=30)
	plt.ylabel('Adj-AP', size=30)
	lgnd = plt.legend(loc="upper right", ncol=4, prop={'size': 15})
	for i in range(len(algos)):
	    lgnd.legendHandles[i]._sizes = [250]
	lgnd.legendHandles[len(algos)]._sizes = [600]

	plt.savefig(outfile, bbox_inches='tight')

def clustering_scores(outfile="clustering.pdf"):
	algos = ['RS', 'SDO', 'KMC', 'WGM', 'GNG', 'ODM']
	plt.figure(figsize=(16,8))
	sns.set()
	sns.set_palette('bright')
	sns.set(style="whitegrid")
	for i, algorithm in enumerate(algos):
	    data = np.load('../results/clustering/{}_RES.npy'.format(algorithm))
	    a = np.mean(data, axis=(0))
	    if algorithm == 'ODM':
	        sns.scatterplot(a[2,:], a[0,:], s=[500, 1000, 1500, 2500], label=algorithm, alpha=0.7, linewidth=2, linestyle='--', color = 'white', edgecolor='red')
	    else:
	        sns.scatterplot(a[2,:], a[0,:], s=[500, 1000, 1500, 2500], label=algorithm, alpha=0.7)
	data = np.load('../results/clustering/{}_RES.npy'.format('BASELINE'))
	a = np.mean(data, axis=(0))
	sns.scatterplot(a[2,:], a[0,:], s=4500, color="black", label='BASELINE', alpha=0.8)

	#plt.xscale('log')
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.xlabel('KMeans Time (s)', size=30)
	plt.ylabel('Rand Index', size=30)
	lgnd = plt.legend(loc="upper left", ncol=4, prop={'size': 15})
	for i in range(len(algos)):
	    lgnd.legendHandles[i]._sizes = [250]
	lgnd.legendHandles[len(algos)]._sizes = [600]

	plt.savefig(outfile, bbox_inches='tight')

def supervised_classification_scores(outfile='supervised_classification.pdf'):
	algos = [ 'RS', 'SDO', 'KMC', 'WGM', 'GNG', 'CNN', 'ODM']
	metrics = ['Accuracy', 'Ma. Prec.', 'Ma. Rec.']
	scores = {1:0, 2:2, 3:4} # from np table
	fig = plt.figure(figsize=(16,16))
	fig.subplots_adjust(hspace=0.4, wspace=0.4)
	for i in range(1, 4):
	    plt.subplot(3, 1, i)
	    sns.set()
	    sns.set_palette('bright')
	    sns.set(style="whitegrid")
	    for p, algorithm in enumerate(algos):
	        data = np.load('../results/supervised/{}_RES.npy'.format(algorithm))
	        a = np.mean(data, axis=(0))
	        if algorithm == 'ODM':
	            sns.scatterplot(a[6,:], a[scores[i],:], s=[500, 1000, 1500, 2500], label=algorithm, alpha=0.7, linewidth=2, linestyle='--', color = 'white', edgecolor='red')
	        else:
	            sns.scatterplot(a[6,:], a[scores[i],:], s=[500, 1000, 1500, 2500], label=algorithm, alpha=0.7)
	    data = np.load('../results/supervised/{}_RES.npy'.format('BASELINE'))
	    a = np.mean(data, axis=(0))
	    sns.scatterplot(a[6,:], a[scores[i],:], s=4500, color="black", label='BASELINE', alpha=0.8)
	    #plt.xscale('log')
	    plt.xticks(size=20)
	    plt.yticks(size=20)
	    plt.xlabel('kNN Time (s)', size=30)
	    plt.ylabel(metrics[i-1], size=30)
	    if i ==2:
	        lgnd = plt.legend(loc="lower right", ncol=4, prop={'size': 14})
	    else:
	        lgnd = plt.legend(loc="lower right", ncol=4, prop={'size': 14})
	    for k in range(len(algos)):
	        lgnd.legendHandles[k]._sizes = [250]
	    lgnd.legendHandles[len(algos)]._sizes = [400]

	plt.savefig(outfile, bbox_inches='tight')

def anomaly_detection_times(outfile="anomaly_detection_ext_nonlog.pdf", _log=True):
	c = ['1','2','3','4','5','6','7', 'ext_time', '9', 'algo', 'r']
	final1 = pd.DataFrame(columns=c)
	algos = ['RS', 'SDO', 'ODM', 'KMC', 'WGM', 'GNG']
	for i, algorithm in enumerate(algos):
	    data = np.load('../results/anomaly_detection/{}_RES.npy'.format(algorithm))
	    a = pd.DataFrame(np.mean(data, axis=(0)).T, columns=c[:-2])
	    a['algo'] = algorithm
	    a['r'] = [0.5, 1, 5, 10]
	    final1 = final1.append(a)

	sns.set()
	sns.set_palette('bright')
	sns.set(style="whitegrid")
	g = sns.catplot(x="r", y="ext_time", hue="algo", palette='Paired', data=final1, kind="bar", height=6, aspect=2, alpha=0.8)
	g._legend.set_title('Algorithm')
	plt.setp(g._legend.get_title(), fontsize=15)
	plt.setp(g._legend.get_texts(), fontsize=15)
	#lgnd = plt.legend(prop={'size': 15})
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.xlabel('r (%)', size=30)
	plt.ylabel('A. Detection Ext. Time (s)', size=30)
	if _log:
		plt.yscale('log')

	plt.savefig(outfile, bbox_inches='tight')

def clustering_times(outfile="clustering_ext_nonlog.pdf", _log=True):
	c = ['rand', 'ext_time', 'kmeans', 'l_extender', 'algo', 'r']
	final2 = pd.DataFrame(columns=c)
	algos = ['RS', 'SDO', 'ODM', 'KMC', 'WGM', 'GNG']
	for i, algorithm in enumerate(algos):
	    data = np.load('../results/clustering/{}_RES.npy'.format(algorithm))
	    a = pd.DataFrame(np.mean(data, axis=(0)).T, columns=c[:-2])
	    a['algo'] = algorithm
	    a['r'] = [0.5, 1, 5, 10]
	    final2 = final2.append(a)

	sns.set()
	sns.set_palette('bright')
	sns.set(style="whitegrid")
	g = sns.catplot(x="r", y="ext_time", hue="algo", palette='Paired', data=final2, kind="bar", height=6, aspect=2, alpha=0.8)
	g._legend.set_title('Algorithm')
	plt.setp(g._legend.get_title(), fontsize=15)
	plt.setp(g._legend.get_texts(), fontsize=15)
	#lgnd = plt.legend(prop={'size': 15})
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.xlabel('r (%)', size=30)
	plt.ylabel('Clustering Ext. Time (s)', size=30)
	if _log:
		plt.yscale('log')

	plt.savefig(outfile, bbox_inches='tight')

def supervised_classification_times(outname="supervised_ext_nonlog.pdf", _log=True):
	c = ['1','2','3','4','5', 'ext_time', '7', 'algo', 'r']
	final3 = pd.DataFrame(columns=c)
	algos = ['RS', 'SDO', 'ODM', 'KMC', 'WGM', 'CNN', 'GNG']
	for i, algorithm in enumerate(algos):
	    data = np.load('../results/supervised/{}_RES.npy'.format(algorithm))
	    a = pd.DataFrame(np.mean(data, axis=(0)).T, columns=c[:-2])
	    a['algo'] = algorithm
	    a['r'] = [0.5, 1, 5, 10]
	    final3 = final3.append(a)

	sns.set()
	sns.set_palette('bright')
	sns.set(style="whitegrid")
	g = sns.catplot(x="r", y="ext_time", hue="algo", palette='Paired', data=final3, kind="bar", height=6, aspect=2, alpha=0.8)
	g._legend.set_title('Algorithm')
	plt.setp(g._legend.get_title(), fontsize=15)
	plt.setp(g._legend.get_texts(), fontsize=15)
	#lgnd = plt.legend(prop={'size': 15})
	plt.xticks(size=20)
	plt.yticks(size=20)
	plt.xlabel('r (%)', size=30)
	plt.ylabel('Classification Ext. Time (s)', size=30)
	if _log:
		plt.yscale('log')

	plt.savefig(outfile, bbox_inches='tight')


if __name__=="__main__":
	print('Hello, choose your function...')
	# supervised_classification_times()