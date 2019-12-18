import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class ResidualBlock(keras.Model):
    def __init__(self, kernel_size=(3, 3), strides=(1, 1)):
        super(ResidualBlock, self).__init__()
        output_channels = x.getshape()[-1].value
        input_channels = output_channels // 4

        self.conv2D1 = layers.Conv2D(input_channels, kernel_size=(1,1))
        self.conv2D2 = layers.Conv2D(32, kernel_size=kernel_size, strides=strides, padding='same')
        self.conv2D3 = layers.Conv2D(output_channels, kernel_size=(1,1))
        self.batchNorm = layers.BatchNormalization()

    def call(self, x):
        input = x

        x = self.batchNorm(x)
        x = tf.nn.relu(x)
        x = self.conv2D1(x)

        x = self.batchNorm(x)
        x = tf.nn.relu(x)
        x = self.conv2D2(x)

        x = self.batchNorm(x)
        x = tf.nn.relu(x)
        x = self.conv2D3(x)

        x = layers.Add([x, input])
        return x