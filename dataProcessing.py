import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing as preprocessing

def normalize(x):
    x = x.astype("float32")
    x = x / 255.0
    return x

def preprocessData(x_train, y_train, x_test, y_test):
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

def createBatches(x, y, batch_size=128):
    assert len(x) == len(y)
    x_batches = []
    y_batches = []
    n_batches = round(len(x) / batch_size)

    for i in range(n_batches):
        index = i * batch_size

        if index + batch_size >= len(x):
            x_batch = x[index:]
            y_batch = y[index:]
        else:
            x_batch = x[index:index+batch_size]
            y_batch = y[index:index+batch_size]

        x_batches.append(np.array(x_batch))
        y_batches.append(np.array(y_batch))

    return np.array(x_batches), np.array(y_batches)