import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class AttentionResNet(keras.Model):
    def __init__(self):
        super(AttentionResNet, self).__init__()
        self.conv1 = layers.Conv2D(32, (7,7), strides=(2,2), padding='same')
        self.pool1 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')

        # Add res and attention blocks here
        #

        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = layers.BatchNormalization(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        # Add res and attention blocks here
        #

        x = layers.flatten(x)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = tf.nn.softmax(x)
        return x