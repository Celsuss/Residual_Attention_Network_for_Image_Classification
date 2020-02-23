import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class ResidualBlock(keras.Model):
    def __init__(self, input_channels=64, output_channels=64, kernel_size=(3, 3), strides=(1, 1)):
        super(ResidualBlock, self).__init__()
        # output_channels = x.getshape()[-1].value
        # input_channels = output_channels // 4

        self.skip_add = layers.Add()

        self.conv2D1 = layers.Conv2D(input_channels, (1,1))
        self.conv2D2 = layers.Conv2D(input_channels, kernel_size, padding='same', strides=strides)
        self.conv2D3 = layers.Conv2D(output_channels, (1,1), padding='same')
        self.batchNorm = layers.BatchNormalization()

        if input_channels != output_channels:
            self.conv2D4 = layers.Conv2D(output_channels, (1, 1), padding='same', strides=strides)
        else:
            self.conv2D4 = None

    def call(self, x):
        input_shape = x.shape
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

        if self.conv2D4 is not None:
            input = self.conv2D4(input)

        assert input_shape == x.shape
        x = self.skip_add([x, input])
        return x