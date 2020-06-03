from models.model import AttentionResNet
from models.refModel import RefConvNet
from models import refModel
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

    x_train, y_train, x_test, y_test = dataProcessing.preprocessData(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test

def readArgs(settings):
    args = sys.argv[1:]
    for arg in args:
        if arg == '-d' or arg == '--draw':
            settings['draw'] = True

    return settings

def trainModel(model, x_train, y_train, x_test, y_test, hyperparameters, name, save_weights=False, save_keras_model=False, model_save_path='model_weights'):
    learning_rate = hyperparameters['learning_rate']
    momentum = hyperparameters['momentum']
    weight_decay = hyperparameters['weight_decay']
    epochs = hyperparameters['epochs']
    
    loss_op = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)

    model = training.train(model, x_train, y_train, x_test, y_test, loss_op, optimizer, epochs, model_save_path=model_save_path, model_name=name, save_weights=save_weights, save_keras_model=save_keras_model)
    return model

def drawImages(x_train, y_train, settings):
    if settings['draw']:
        n_draw_images = 5
        utils.drawImages(x_train[:n_draw_images], y_train[:n_draw_images])

def main():
    settings = {'draw': False}
    settings = readArgs(settings)

    hyperparameters = {}
    hyperparameters['learning_rate'] = 0.001
    hyperparameters['momentum'] = 0.9
    hyperparameters['weight_decay'] = 0.0001
    hyperparameters['epochs'] = 50
    hyperparameters['batch_size'] = 128

    x_train, y_train, x_test, y_test = getData()
    data_shape = x_train.shape[1:]
    img_height = x_train.shape[1]
    img_width = x_train.shape[2]
    channels = x_train.shape[3]
    drawImages(x_train, y_train, settings)
    print('Input data has shape {}'.format(data_shape))

    x_train, y_train = dataProcessing.createBatches(x_train, y_train, hyperparameters['batch_size'])

    # Reference model
    # if utils.isFile('model_weights/ref_model/ref_model.h5'):
    #     model = utils.loadKerasModel('model_weights', 'ref_model')
    # else:
    #     model = refModel.getRefConvNet(input_channels=32, input_shape=data_shape)
    #     model = trainModel(model, x_train, y_train, x_test, y_test, hyperparameters, 'ref_model', save_keras_model=True)
    # training.testModel(model, x_test, y_test, 'ref model')
    
    # AttentionResNet
    model = AttentionResNet((img_height, img_width, channels))

    # if utils.isFile('model_weights\AttentionResNet\AttentionResNet_weights.h5'):
    #     model = utils.loadModelWeights(model, 'model_weights', 'AttentionResNet')
    # else:
    model = trainModel(model, x_train, y_train, x_test, y_test, hyperparameters, 'AttentionResNet', save_weights=True)
        
    training.testModel(model, x_test, y_test, 'AttentionResNet')

    return 0

if __name__ == '__main__':
    main()