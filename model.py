import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class RefNet(keras.Model):
    def __init__(self):
        super(RefNet, self).__init__()

    def call(self, x):
        
        return x