import numpy as np
import pandas as pd


def readfile(filename):
    d = pd.read_csv(filename, header=None, sep=" ", dtype=np.float32)
    d = np.array(d.values).astype(np.float32)
    return d[:, :32]


def printmm(out):
    for i in out[0]:
        print(str(i))


def predict(key, data):
    first_hidden_layer = np.reshape(data[0, :], (-1, 1))
    second_hidden_layer = data[1:33, :]
    output_layer = np.reshape(data[33:, :], (-1, 1))


    
    out_1 = np.dot(key, first_hidden_layer).T
    printmm(out_1)
    out_1 = np.maximum(out_1, 0)
    print(out_1)
    out_2 = np.dot(out_1, second_hidden_layer)
    out_2 = np.maximum(out_2, 0).T
    res = np.dot(out_2, output_layer)

    return res


weights = '/Users/deepak/Downloads/WebLogs/model_params_layer_11.txt'
# filename = '/Users/deepak/Downloads/WebLogs/model_params_layer_22.txt'
key = np.float64(1427228465523.5 - 1425168000107.1)
key = np.array(key)
print(str(key))
data = readfile(weights)
model_index = predict(key, data)
print(model_index)
