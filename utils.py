import math
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    return 0