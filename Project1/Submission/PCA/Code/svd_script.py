import numpy as np
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
    u, s, v = np.linalg.svd(np.matrix(data, dtype='float'), full_matrices=True)
    svd_data = u[:, :2]
    # print(s)
    return svd_data


def main(filename):
    data, labels = readfile(filename)
    labels = pd.DataFrame(labels)
    data = data - np.mean(data, axis=0)
    svd_data = run_svd(data)
    plot_data = np.append(svd_data, labels, 1)
    plot_data = pd.DataFrame(plot_data, columns=["F1", "F2", "Diseases"])
    plot_data[["F1", "F2"]] = plot_data[["F1", "F2"]].apply(pd.to_numeric)
    sns.set(palette="muted", style='white')
    plot = sns.scatterplot(data=plot_data, x="F1", y="F2", hue="Diseases")
    plt.title("Scatter plot for SVD reduction : " + filename)
    plt.show()


main("pca_a.txt")
