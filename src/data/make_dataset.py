import numpy as np
import pandas as pd

def make_dataset(data_src, n_past: int, n_future: int, train_split=(70, 15, 15)) -> np.ndarray:
    train_portion = data_src[:int(len(data_src) * train_split[0] / 100)]
    val_portion = data_src[int(len(data_src) * train_split[0] / 100):int(len(data_src) * (train_split[0] + train_split[1]) / 100)]
    test_portion = data_src[int(len(data_src) * (train_split[0] + train_split[1]) / 100):]

    train_X = []
    train_Y = []
    for i in range(len(train_portion) - n_past - n_future + 1):
        train_X.append(train_portion[i:i + n_past])
        train_Y.append(train_portion[i + n_past:i + n_past + n_future])
    # for i in range(n_past, len(train_portion) - n_future + 1):
    #     train_X.append(train_portion[i - n_past: i, 0: train_portion.shape[1]])
    #     train_Y.append(train_portion[i + n_future - 1: i + n_future, :])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    val_X = []
    val_Y = []
    for i in range(len(val_portion) - n_past - n_future + 1):
        val_X.append(val_portion[i:i + n_past])
        val_Y.append(val_portion[i + n_past:i + n_past + n_future])
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)

    test_X = []
    for i in range(len(test_portion) - n_past - n_future + 1):
        test_X.append(test_portion[i:i + n_past])
    test_X = np.array(test_X)

    return train_X, train_Y, val_X, val_Y, test_X
