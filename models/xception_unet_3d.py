# https://github.com/simeon-spasov/MCI/blob/master/utils/sepconv3D.py

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import InputSpec, Conv3D

from tensorflow.keras import layers


class SeparableConv3D (Conv3D):
        """A custom implementation of 3D Separable Convolutions
        The layer takes activations with the shape batch x N_1 x N_2 x N_3 x N
        (batch is the batch_size, N_k is the k-th dimension, N - number of channels)
        First each of the N channels is convolved separately producing a single
        output feature map, i.e. depth multiplier is 1 (depthwise procedure)
        Then we apply 1x1x1 convolutions with N output channels on the output of the
        depthwise procedure (pointwise step)
    
        Module has only been used (and tested) with a depth multiplier of 1 but
        support for higher depth multipliers is built-in
    
        """
    
        def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 depth_multiplier=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 pointwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 pointwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 pointwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
            
            super(SeparableConv3D, self).__init__(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    bias_constraint=bias_constraint,
                    **kwargs)
            
            self.depth_multiplier = depth_multiplier
            self.depthwise_initializer = initializers.get(depthwise_initializer)
            self.pointwise_initializer = initializers.get(pointwise_initializer)
            self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
            self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
            self.depthwise_constraint = constraints.get(depthwise_constraint)
            self.pointwise_constraint = constraints.get(pointwise_constraint)
            
            

    
        def build(self, input_shape):
            if len(input_shape) < 5:
                raise ValueError('Inputs to `SeparableConv3D` should have rank 5. '
                                 'Received input shape:', str(input_shape))
            if self.data_format == 'channels_first':
                    self.channel_axis = 1
            else:
                self.channel_axis = 4
            if input_shape[self.channel_axis] is None:
                raise ValueError('The channel dimension of the inputs to '
                                 '`SeparableConv3D` '
                                 'should be defined. Found `None`.')
            self.input_dim = int(input_shape[self.channel_axis])
            
            depthwise_kernel_shape = (self.kernel_size[0],
                                      self.kernel_size[1],
                                      self.kernel_size[2],
                                      self.input_dim,
                                      self.depth_multiplier)
            pointwise_kernel_shape = (1, 1, 1,
                                      self.depth_multiplier * self.input_dim,
                                      self.filters)

            self.depthwise_kernel = self.add_weight(
                    shape=depthwise_kernel_shape,
                    initializer=self.depthwise_initializer,
                    name='depthwise_kernel',
                    regularizer=self.depthwise_regularizer,
                    constraint=self.depthwise_constraint)
            
            self.pointwise_kernel = self.add_weight(
                    shape=pointwise_kernel_shape,
                    initializer=self.pointwise_initializer,
                    name='pointwise_kernel',
                    regularizer=self.pointwise_regularizer,
                    constraint=self.pointwise_constraint)

            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            else:
                self.bias = None
            # Set input spec.
            self.input_spec = InputSpec(ndim=5, axes={self.channel_axis: self.input_dim})
            self.built = True

    


        def call(self, inputs):
            depthwise_conv_on_filters = []
            sliced_inputs = [sliced for sliced in tf.split(inputs, self.input_dim, self.channel_axis)]
            sliced_kernels = [sliced for sliced in tf.split(self.depthwise_kernel, self.input_dim, 3)]
            #See https://www.tensorflow.org/versions/r0.12/api_docs/python/array_ops/slicing_and_joining
            for i in range(self.input_dim):
               depthwise_conv_on_filters.append (  K.conv3d(sliced_inputs[i], 
                                                     sliced_kernels[i],
                                                     strides=self.strides,
                                                     padding=self.padding,
                                                     data_format=self.data_format,
                                                     dilation_rate=self.dilation_rate)   )

            depthwise_conv = K.concatenate(depthwise_conv_on_filters)
            pointwise_conv = K.conv3d(depthwise_conv, self.pointwise_kernel,
                                     strides = (1, 1, 1), padding = self.padding,
                                     data_format = self.data_format,
                                     dilation_rate=self.dilation_rate)
            
            outputs = pointwise_conv
            
            if self.use_bias:
                outputs = K.bias_add(
                        outputs,
                        self.bias,
                        data_format=self.data_format)

            if self.activation is not None:
                return self.activation(outputs)
            return outputs

        def get_config(self):
            config = super(SeparableConv3D, self).get_config()
            config.pop('kernel_initializer')
            config.pop('kernel_regularizer')
            config.pop('kernel_constraint')
            config['depth_multiplier'] = self.depth_multiplier
            config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
            config['pointwise_initializer'] = initializers.serialize(self.pointwise_initializer)
            config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
            config['pointwise_regularizer'] = regularizers.serialize(self.pointwise_regularizer)
            config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
            config['pointwise_constraint'] = constraints.serialize(self.pointwise_constraint)
            return config



def get_model(input_shape, num_classes):
    inputs = tf.keras.Input(input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = SeparableConv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = SeparableConv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling3D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling3D(2)(x)

        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv3D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model