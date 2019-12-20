import pickle
import numpy as np
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
    for i in range(len(images)):
        image = images[i]
        fig, ax = plt.subplots()
        ax.axis('off')
        if labels != None and len(labels) >= len(images):
            ax.set_title(labels[i])
        ax.imshow(image)

    plt.show()
    return 0