import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        files = pickle.load(fo, encoding='bytes')
    return files

def getTrainData():
    files = ['data/cifar-10/cifar-10-batches-py/data_batch_1', 'data/cifar-10/cifar-10-batches-py/data_batch_2', 'data/cifar-10/cifar-10-batches-py/data_batch_3',
            'data/cifar-10/cifar-10-batches-py/data_batch_4', 'data/cifar-10/cifar-10-batches-py/data_batch_5']

    data = []
    for file in files:
        data.append(unpickle(file))
    return np.array(data)

def main():
    files = unpickle('./data/cifar-10/cifar-10-batches-py/data_batch_1')
    train_data = getTrainData()
    

    return 0

if __name__ == '__main__':
    main()