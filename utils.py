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
        
def isFile(file_path):
    paths = file_path.split('/')
    current_path = ''
    
    for path in paths:
        current_path = os.path.join(current_path, path)
        if os.path.isdir(current_path) is False and os.path.isfile(current_path) is False:
            return False

        continue

    return True

def saveKerasModel(model, path, model_name):
    path = os.path.join(path, model_name, model_name + '.h5')
    path = path.replace('\\', '/')
    createPath(path)
    print('Saved model {}'.format(path))
    model.save(path)

def saveModel(model, path, model_name):
    path = os.path.join(path, model_name)
    path = path.replace('\\', '/')
    createPath(path)
    model_path = os.path.join(path, model_name + '_config.json')
    weights_path = os.path.join(path, model_name + '_weights.h5')
    model_path = path.replace('\\', '/')
    weights_path = path.replace('\\', '/')

    json_config = model.to_json()
    if json_config is not None:
        with open(model_path, 'w') as json_file:
            json_file.write(json_config)

    model.save_weights(weights_path)

def saveModelWeights(model, path, model_name):
    path = os.path.join(path, model_name)
    path = os.path.join(path, model_name + '_weights.h5')
    path = path.replace('\\', '/')
    createPath(path)
    print('Saved model weights {}'.format(path))
    model.save_weights(path)

def loadKerasModel(path, model_name):
    path = os.path.join(path, model_name, model_name + '.h5')
    path = path.replace('\\', '/')
    createPath(path)
    model = tf.keras.models.load_model(path)
    return model

def loadModelWeights(model, path, model_name):
    path = os.path.join(path, model_name)
    path = path.replace('\\', '/')
    createPath(path)
    model_path = os.path.join(path, model_name + '_config.json')
    weights_path = os.path.join(path, model_name + '_weights.h5')

    with open(model_path) as json_file:
        json_config = json_file.read()

    new_model = tf.keras.models.model_from_json(json_config)
    model.load_weights(weights_path)

    return 0

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = getMNISTDataset()
    n_images_to_draw = 6
    drawImages(x_train[:n_images_to_draw], y_train[:n_images_to_draw])