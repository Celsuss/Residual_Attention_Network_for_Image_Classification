import numpy as np
import utils

def main():
    test_files = ['data/cifar-10/cifar-10-batches-py/data_batch_1', 'data/cifar-10/cifar-10-batches-py/data_batch_2', 'data/cifar-10/cifar-10-batches-py/data_batch_3',
            'data/cifar-10/cifar-10-batches-py/data_batch_4', 'data/cifar-10/cifar-10-batches-py/data_batch_5']
    test_file = 'data/cifar-10/cifar-10-batches-py/test_batch'
    train_data, train_labels = utils.getTrainData(test_files)
    test_data, test_labels = utils.getTestData(test_file)
    
    train_batch_size = train_data[0].shape[0]

    return 0

if __name__ == '__main__':
    main()