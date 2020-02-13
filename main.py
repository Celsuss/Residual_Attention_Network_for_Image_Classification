import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import utils
import dataProcessing
from models.model import AttentionResNet
from models.refModel import RefConvNet

print('Tensorflow version: {}'.format(tf.__version__))

# TODO: Uncomment this, only for debuging
# @tf.function
def trainStep(model, x, y, loss_op, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_op(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)
    
# TODO: Uncomment this, only for debuging
# @tf.function
def testStep(model, x, y, loss_op, test_loss, test_accuracy):
    predictions = model(x)
    loss = loss_op(y, predictions)

    test_loss(loss)
    test_accuracy(y, predictions)

def train(model, x_train, y_train, x_test, y_test, loss_op, optimization, epochs):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    n_batches = len(x_train)

    for epoch in range(epochs):
        n_batch = 0
        for x, y in zip(x_train, y_train):
            n_batch+=1
            template = '[Epoch {}/{}, Batch {}/{}] Loss: {:.3f}, Accuracy: {:.2%}'
            print(template.format(epoch+1, epochs, n_batch, n_batches, train_loss.result(), train_accuracy.result()), end='\r')
            trainStep(model, x, y, loss_op, optimization, train_loss, train_accuracy)

        testStep(model, x_test, y_test, loss_op, test_loss, test_accuracy)

        template = '[Epoch {}] Loss: {:.3f}, Accuracy: {:.2%}, Test Loss: {:.3f}, Test Accuracy: {:.2%}'
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result(),
                test_loss.result(), test_accuracy.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    return model

def getData():
    # x_train, y_train, x_test, y_test = utils.getCifar10Dataset()
    x_train, y_train, x_test, y_test = utils.getMNISTDataset()

    x_train, y_train, x_test, y_test = dataProcessing.preprocessData(x_train, y_train, x_test, y_test, batch_size=128)

    return x_train, y_train, x_test, y_test

def drawTestData(data, n_images):
    for x_batch, y_batch in data:
        images = x_batch[:n_images]
        labels = y_batch[:n_images]
        break

    utils.drawImages(images, labels)
    return 0

def main():
    learning_rate = 0.001
    epochs = 5
    batch_size = 128

    # train_data, x_test, y_test = getData()
    x_train, y_train, x_test, y_test = getData()

    IMG_HEIGHT = x_test.shape[1]
    IMG_WIDTH = x_test.shape[2]
    CHANNELS = x_test.shape[3]

    # drawTestData(train_data, 5)

    # Reference model
    model = RefConvNet(32, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    x_train, y_train = dataProcessing.createBatches(x_train, y_train, batch_size)

    loss_op = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    train(model, x_train, y_train, x_test, y_test, loss_op, optimizer, epochs)
    
    # AttentionResNet
    # model = AttentionResNet()
    # loss_op = keras.losses.CategoricalCrossentropy()
    # optimizer = keras.optimizers.Adam(lr=learning_rate)
    # train(model, train_data, x_test, y_test, loss_op, optimizer, epochs)

    return 0

if __name__ == '__main__':
    main()