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

        # Mask branch layers
        self.down_sampling_units = []
        self.up_sampling_units = []
        self.skip_sampling_units = []

        # Always up-sample the same amount as down-sampling
        n_down_sampling_units = 2
        n_up_sampling_units = n_down_sampling_units
        n_skip_sampling_units = n_down_sampling_units

        for i in range(n_down_sampling_units):
            self.down_sampling_units.append(layers.MaxPool2D(padding='same'))
        for i in range(n_up_sampling_units):
            self.up_sampling_units.append(layers.UpSampling2D())
        for i in range(n_skip_sampling_units):
            self.skip_sampling_units.append(layers.Add())

        self.conv2D1 = layers.Conv2D(channels, (1,1))
        self.conv2D2 = layers.Conv2D(channels, (1,1))
        self.sigmoid1 = layers.Activation('sigmoid')

        # End of mask branch layers

        self.lambd = layers.Lambda(lambda x: x + 1)
        self.multiply = layers.Multiply()

    def call(self, x, input_channels=None):
        """
        Mask branch and trunk branch.
        
        Trunk branch performs feature processing, can be adapted by any
        state-of-the-art network structures.

        Mask branch uses bottom-up top-down structure to learn same size 
        mask M(x) that soft weight output features T(x). 

        Output Hi,c(x) = (1 + Mi,c(x)) ∗ Fi,c(x)
        """

        # if input_channels is None:
        #     input_channels = x.get_shape()[-1]
        #     output_channels = input_channels // 4

        for res_unit in self.p_residual_units:
            x = res_unit(x)

        x_trunk = x
        x_trunk = self.trunkBranch(x_trunk)

        x_mask = x
        x_mask = self.maskBranch(x_mask)

        # Hi,c(x) = (1 + Mi,c(x)) ∗ Fi,c(x)
        x_mask = self.lambd(x_mask)

        x = self.multiply([x_mask, x_trunk])

        return x

    def trunkBranch(self, x):
        # Convolutions
        # Use the residual block?

        for res_unit in self.t_residual_units:
            x = res_unit(x)

        return x

    def maskBranch(self, x):
        # feed-forward sweep and top-down feedback
        input_shape = x.shape

        # Down sampling
        skip_outputs = []
        for down_unit in self.down_sampling_units:
            skip_outputs.append(x)
            x = down_unit(x)

            for res_unit in self.r_residual_units:
                x = res_unit(x)

        # Up sampling with skip connections
        skip_outputs = list(reversed(skip_outputs))

        for i in range(len(self.up_sampling_units)-1):
            up_unit = self.up_sampling_units[i]
            x = up_unit(x)

            for res_unit in self.r_residual_units:
                x = res_unit(x)

            skip_unit = self.skip_sampling_units[i]
            x = skip_unit([x, skip_outputs[i]])

        # Last up sampling, no res units after this up sampling.
        x = self.up_sampling_units[-1](x)

        # The number of bilinear interpolation is the same
        # as max pooling to keep the output size the same as the input
        # feature map. Then a sigmoid layer normalizes the output
        # range to [0, 1] after two consecutive 1 × 1 convolution layers. We also added skip connections between bottom-up
        # and top-down parts to capture information from different scales

        assert input_shape == x.shape

        x = self.conv2D1(x)
        x = self.conv2D2(x)
        x = self.sigmoid1(x)

        assert input_shape == x.shape

        return x