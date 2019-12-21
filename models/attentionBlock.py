import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from .residualBlock import ResidualBlock

class AttentionBlock(keras.Model):
    def __init__(self, channels=64, p=1, t=2, r=1):
        super(AttentionBlock, self).__init__()
        """
        Hyperparameters p, t and r.
        p: Nr of pre-processing Residual units
        t: Nr of Residual units in trunk branch
        r: Nr of Residual units between adjacent pooling layer in the mask branch
        """
        self.p = p
        self.t = t
        self.r = r
        self.p_residual_units = []
        self.t_residual_units = []
        self.r_residual_units = []
        
        for i in range(p):
            self.p_residual_units.append(ResidualBlock(channels, channels))
        for i in range(t):
            self.t_residual_units.append(ResidualBlock(channels, channels))
        for i in range(r):
            self.r_residual_units.append(ResidualBlock(channels, channels))

    def call(self, x, input_channels=None):
        """
        Mask branch and trunk branch.
        
        Trunk branch performs feature processing, can be adapted by any
        state-of-the-art network structures.

        Mask branch uses bottom-up top-down structure to learn same size 
        mask M(x) that soft weight output features T(x). 

        Output Hi,c(x) = (1 + Mi,c(x)) ∗ Fi,c(x)
        """

        if input_channels is None:
            input_channels = x.get_shape()[-1]
            output_channels = input_channels // 4

        for res_unit in self.p_residual_units:
            x = res_unit(x)

        x_trunk = x
        x_trunk = self.trunkBranch(x_trunk)

        x_mask = x
        x_mask = self.maskBranch(x_mask)

        return x

    def trunkBranch(self, x):


        return x

    def maskBranch(self, x):

        return x