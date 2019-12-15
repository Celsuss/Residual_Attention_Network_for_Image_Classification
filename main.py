import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import utils
from models.model import AttentionResNet

def train(model, x_train, y_train, x_test, y_test, loss_op, optimization):

    return 0

def main():
    learning_rate = 0.01
    epochs = 5

    test_files = ['data/cifar-10/cifar-10-batches-py/data_batch_1', 'data/cifar-10/cifar-10-batches-py/data_batch_2', 'data/cifar-10/cifar-10-batches-py/data_batch_3',
            'data/cifar-10/cifar-10-batches-py/data_batch_4', 'data/cifar-10/cifar-10-batches-py/data_batch_5']
    test_file = 'data/cifar-10/cifar-10-batches-py/test_batch'
    train_data, train_labels = utils.getTrainData(test_files)
    test_data, test_labels = utils.getTestData(test_file)
    
    model = AttentionResNet()
    loss_op = keras.losses.sparse_softmax_cross_entropy
    optimizer = keras.optimizer.Adam(lr=learning_rate)
    train(model, train_data, train_labels, test_data, test_labels, loss_op, optimizer)

    return 0

if __name__ == '__main__':
    main()