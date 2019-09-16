import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def readfile(filename):
	data = pd.read_csv(filename, header=None, sep ="\t")
	# data = np.genfromtxt(filename, delimiter="\t",usecols=tuple(range(4)))
	data = np.array(data.values)
	data_parsed = data[:,:-1]
	labels = data[:,-1]
	return data_parsed, labels

def run_svd(data):
	svd = TruncatedSVD(n_components=2)
	svd.fit(data)
	svd_data = svd.transform(data)
	return svd_data

def main():
	data, labels = readfile("pca_a.txt")
	labels = pd.DataFrame(labels)
	# data = data - np.mean(data, axis = 0)
	svd_data = run_svd(data)
	plot_data = np.append(svd_data, labels, 1)
	plot_data = pd.DataFrame(plot_data, columns=["F1", "F2", "Labels"])
	plot_data[["F1","F2"]] = plot_data[["F1","F2"]].apply(pd.to_numeric)
	# plot_data = plot_data.round(3)
	print(plot_data.describe())
	# print(plot_data)

	sns.set(palette="muted",style='white')
	plot = sns.scatterplot(data=plot_data, x = "F1", y = "F2", hue = "Labels")
	# ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
	# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
	# plot.set(xticks = np.arange(min(plot_data["F1"])-1, max(plot_data["F1"])+1, 10))
	# plot.add_legend(title="Diseases")
	plt.title("Scatter plot for SVD reduction : pca_a.txt")
	# plt.xticks(np.arange(min(plot_data["F1"].astype("int"))-1, max(plot_data["F1"].astype("int"))+1, 10))
	plt.show()
main()