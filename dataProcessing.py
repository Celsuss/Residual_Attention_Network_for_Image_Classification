import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing as preprocessing

def normalize(x):
    # x = x.astype(np.float32)
    x = x.astype("float32")
    x = x / 255.0
    return x

def preprocessData(x_train, y_train, x_test, y_test, batch_size=32):
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    generator = preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    train_batches = generator.flow(x_train, y_train, batch_size=batch_size)
    # test_batches = generator.flow(x_test, y_test)

    return train_batches, x_test, y_test