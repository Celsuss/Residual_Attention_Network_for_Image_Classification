import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import utils
import training
import dataProcessing
from models.model import AttentionResNet
from models.refModel import RefConvNet

print('Tensorflow version: {}'.format(tf.__version__))

def getData():
    x_train, y_train, x_test, y_test = utils.getCifar10Dataset()
    # x_train, y_train, x_test, y_test = utils.getMNISTDataset()

    x_train, y_train, x_test, y_test = dataProcessing.preprocessData(x_train, y_train, x_test, y_test, batch_size=128)

    return x_train, y_train, x_test, y_test

def main():
    learning_rate = 0.001
    epochs = 5
    batch_size = 128

    x_train, y_train, x_test, y_test = getData()

    IMG_HEIGHT = x_test.shape[1]
    IMG_WIDTH = x_test.shape[2]
    CHANNELS = x_test.shape[3]

    n_draw_images = 5
    utils.drawImages(x_train[:n_draw_images], y_train[:n_draw_images])

    # Reference model
    # model = RefConvNet(32, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    x_train, y_train = dataProcessing.createBatches(x_train, y_train, batch_size)

    # loss_op = keras.losses.CategoricalCrossentropy()
    # optimizer = keras.optimizers.Adam(lr=learning_rate)
    # train(model, x_train, y_train, x_test, y_test, loss_op, optimizer, epochs)
    
    # AttentionResNet
    model = AttentionResNet((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    loss_op = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    training.train(model, x_train, y_train, x_test, y_test, loss_op, optimizer, epochs)

    return 0

if __name__ == '__main__':
    main()