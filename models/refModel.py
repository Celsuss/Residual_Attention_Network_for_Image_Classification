
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class RefConvNet(keras.Model):
    def __init__(self, input_channels=32, input_shape=(32, 32, 3)):
        super(RefConvNet, self).__init__()
        self.dropout = layers.Dropout(0.2)

        self.conv1 = layers.Conv2D(input_channels, (1,1), padding='same', activation='relu', input_shape=input_shape)
        self.pool1 = layers.MaxPool2D()
        self.conv2 = layers.Conv2D(input_channels*2, (1,1), padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D()
        self.conv3 = layers.Conv2D(input_channels*4, (1,1), padding='same', activation='relu')
        self.pool3 = layers.MaxPool2D()

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.nn.softmax(x)

        return x