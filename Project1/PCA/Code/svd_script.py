import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def readfile(filename):
    data = pd.read_csv(filename, header=None, sep="\t")
    data = np.array(data.values)
    data_parsed = data[:, :-1]
    labels = data[:, -1]
    return data_parsed, labels


def run_svd(data):
    svd = TruncatedSVD(n_components=2)
    svd.fit(data)
    svd_data = svd.transform(data)
    return svd_data


def main():
    file = "pca_c.txt"
    data, labels = readfile(file)
    labels = pd.DataFrame(labels)
    svd_data = run_svd(data)
    plot_data = np.append(svd_data, labels, 1)
    plot_data = pd.DataFrame(plot_data, columns=["F1", "F2", "Labels"])
    plot_data[["F1", "F2"]] = plot_data[["F1", "F2"]].apply(pd.to_numeric)
    print(plot_data.describe())

    sns.set(palette="muted", style='white')
    plot = sns.scatterplot(data=plot_data, x="F1", y="F2", hue="Labels")
    plt.title("Scatter plot for SVD reduction : pca_a.txt")
    plt.show()


main()
