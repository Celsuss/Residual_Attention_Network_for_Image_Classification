import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import dataProcessing
import pickle
import math
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def getData(file):
    file_content = unpickle(file)
    features = file_content['data'].reshape(file_content['data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    labels = file_content['labels']
    return features, labels

def getTrainData(files):
    features = []
    labels = []
    for file in files:
        file_features, file_labels = getData(file)
        features.append(file_features)
        labels.append(file_labels)

    return features, labels

def getTestData(file):
    return getData(file)

def getCifar10Dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_train, y_train, x_test, y_test

def getMNISTDataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    shape = x_train.shape
    x_train = x_train.reshape(x_train.shape[0], shape[1], shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], shape[1], shape[2], 1)

    return x_train, y_train, x_test, y_test

def drawImage(image, label=None):
    fig, ax = plt.subplots()
    ax.axis('off')
    if label != None:
        ax.set_title(label)

    ax.imshow(image)
    plt.show()
    return 0

# Draw an array of images.
def drawImages(images, labels=None):
    if images.shape[3] == 1:
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2])

    n_rows = math.floor(len(images)/2)
    n_columns = math.ceil(len(images)/2)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(8,8))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis('off')

        if i < len(images):
            image = images[i]
            ax.imshow(image)
            if labels is not None and i < len(labels):
                ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()

def createPath(full_model_path):
    paths = full_model_path.split('/')[:-1]
    full_path = ''

    for path in paths:
        full_path = os.path.join(full_path, path)
        if not os.path.isdir(full_path):
            os.mkdir(full_path)

def saveModel(model, path, model_name):
    full_path = os.path.join(path, model_name, model_name + '.h5')
    full_path = full_path.replace('\\', '/')
    createPath(full_path)
    model.save_weights(full_path)
    tf.saved_model.save(model, full_path)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = getMNISTDataset()
    n_images_to_draw = 6
    drawImages(x_train[:n_images_to_draw], y_train[:n_images_to_draw])