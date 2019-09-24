import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def readfile(filename):
	data = pd.read_csv(filename, header=None, sep ="\t")
	data = np.array(data.values)
	data_parsed = data[:,:-1]
	labels = data[:,-1]
	return data_parsed, labels

def run_tsne(data):
	# data = data[:,:16]
	data_reduced = TSNE(n_components=2).fit_transform(data)
	return data_reduced

def main(filename):
	data, labels = readfile(filename)
	labels = pd.DataFrame(labels)
	tse_data = run_tsne(data)
	print(tse_data)
	plot_data = np.append(tse_data, labels, 1)
	plot_data = pd.DataFrame(plot_data, columns=["F1", "F2", "Diseases"])
	plot_data[["F1","F2"]] = plot_data[["F1","F2"]].apply(pd.to_numeric)
	print(plot_data.describe())
	sns.set(palette="muted",style='white')
	plot = sns.scatterplot(data=plot_data, x = "F1", y = "F2", hue = "Diseases")
	plt.title("Scatter plot for t-sne reduction, File: "+filename)
	plt.show()
main("pca_c.txt")