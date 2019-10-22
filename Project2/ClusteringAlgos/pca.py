import numpy as np

# Calculating PCA
def pca(data):
    # Get mean along each dimension
    mean = np.mean(data, axis=0)

    # Normalize data
    normalizedData = data - mean

    # find co-variance matrix
    cov = np.cov(normalizedData.T)

    # Get eigen-value and eigen-vector of the co-variance matrix
    eig_val, eig_vec = np.linalg.eig(cov)

    # Pick top-two eigen-value and corresponding eigen-vector
    top_ind = eig_val.argsort()[-2:][::-1]
    top_eig_vec = eig_vec[:, top_ind]

    p = np.zeros([data.shape[0], top_eig_vec.shape[1]])

    p = np.dot(top_eig_vec.T, normalizedData.T).T
    return p