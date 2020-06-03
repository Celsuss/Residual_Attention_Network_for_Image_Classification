import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from .attentionBlock import AttentionBlock
from .residualBlock import ResidualBlock

class AttentionResNet(keras.Model):
    def __init__(self, input_shape, channels=64):
        super(AttentionResNet, self).__init__()
        self.conv1 = layers.Conv2D(channels, (7,7), strides=(2,2), padding='same', input_shape=input_shape)
        self.pool1 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')

        self.batchNorm = layers.BatchNormalization()
        self.flatten = layers.Flatten()

        # Add res and attention blocks here
        self.res1 = ResidualBlock()
        self.attention1 = AttentionBlock(64)
        #

        self.dense1 = layers.Dense(4096) # 128
        self.dense2 = layers.Dense(10)

    def call(self, x):          # 32, 32, 3
        x = self.conv1(x)       # 16, 16, 64   
        x = self.batchNorm(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)       # 8, 8, 64

        # Add res and attention blocks here
        x = self.res1(x)        # 8, 8, 64
        x = self.attention1(x)  # 8, 8, 64
        #

        x = self.flatten(x)     # 4096
        x = self.dense1(x)      # ? 128
        x = tf.nn.relu(x)
        x = self.dense2(x)      # 10
        x = tf.nn.softmax(x)
        return x

    def to_json(self):
        return None



