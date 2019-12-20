import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import utils
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
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(epochs):
        n_batch = 0
        for x, y in zip(x_train, y_train):
            n_batch+=1
            print('Batch {}/{}'.format(n_batch, len(x_train)), end='\r')
            trainStep(model, x, y, loss_op, optimization, train_loss, train_accuracy)
            continue

        testStep(model, x_test, y_test, loss_op, test_loss, test_accuracy)

        template = '[Epoch {}] Loss: {:.3f}, Accuracy: {:.2%}, Test Loss: {:.3f}, Test Accuracy: {:.2f}'
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result(),
                test_loss.result(), test_accuracy.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    return model

def main():
    learning_rate = 0.1 # 0.01
    epochs = 5

    test_files = ['data/cifar-10/cifar-10-batches-py/data_batch_1', 'data/cifar-10/cifar-10-batches-py/data_batch_2', 'data/cifar-10/cifar-10-batches-py/data_batch_3',
                'data/cifar-10/cifar-10-batches-py/data_batch_4', 'data/cifar-10/cifar-10-batches-py/data_batch_5']
    test_file = 'data/cifar-10/cifar-10-batches-py/test_batch'
    train_data, train_labels = utils.getTrainData(test_files)
    test_data, test_labels = utils.getTestData(test_file)

    IMG_HEIGHT = train_data[0].shape[1]
    IMG_WIDTH = train_data[0].shape[2]
    CHANNELS = train_data[0].shape[3]

    utils.drawImages(train_data[0][0:5], train_labels[0][0:5])

    # Reference model
    # model = RefConvNet()
    model = keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])

    loss_op = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    train(model, train_data, train_labels, test_data, test_labels, loss_op, optimizer, epochs)
    #
    
    # model = AttentionResNet()
    # # loss_op = keras.losses.sparse_softmax_cross_entropy
    # loss_op = keras.losses.SparseCategoricalCrossentropy()
    # optimizer = keras.optimizers.Adam(lr=learning_rate)
    # train(model, train_data, train_labels, test_data, test_labels, loss_op, optimizer, epochs)

    return 0

if __name__ == '__main__':
    main()