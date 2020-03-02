from models.model import AttentionResNet
from models.refModel import RefConvNet
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import dataProcessing
import training
import utils
import sys

print('Tensorflow version: {}'.format(tf.__version__))

def getData():
    x_train, y_train, x_test, y_test = utils.getCifar10Dataset()
    # x_train, y_train, x_test, y_test = utils.getMNISTDataset()

    x_train, y_train, x_test, y_test = dataProcessing.preprocessData(x_train, y_train, x_test, y_test, batch_size=128)

    return x_train, y_train, x_test, y_test

def readArgs(settings):
    args = sys.argv[1:]
    for arg in args:
        if arg == '-d' or arg == '--draw':
            settings['draw'] = True

    return settings

def main():
    settings = {'draw': False}
    settings = readArgs(settings)

    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    epochs = 5
    batch_size = 128

    x_train, y_train, x_test, y_test = getData()

    img_height = x_test.shape[1]
    img_width = x_test.shape[2]
    channels = x_test.shape[3]

    if settings['draw']:
        n_draw_images = 5
        utils.drawImages(x_train[:n_draw_images], y_train[:n_draw_images])

    x_train, y_train = dataProcessing.createBatches(x_train, y_train, batch_size)

    # Reference model
    model = RefConvNet(32, input_shape=(img_height, img_width, channels))
    loss_op = keras.losses.CategoricalCrossentropy()
    # optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True )
    optimizer = keras.optimizers.SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True )
    training.train(model, x_train, y_train, x_test, y_test, loss_op, optimizer, epochs, model_save_path='model_weights', model_name='ref_model')
    
    # AttentionResNet
    model = AttentionResNet((img_height, img_width, channels))
    loss_op = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    training.train(model, x_train, y_train, x_test, y_test, loss_op, optimizer, epochs, model_save_path='model_weights', model_name='AttentionResNet')

    return 0

if __name__ == '__main__':
    main()