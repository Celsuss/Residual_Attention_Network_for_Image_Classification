import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')

    return data

def getData(file):
    file_content = unpickle(file)
    features = file_content['data']
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
