from JuNet.neuralnetwork.builder import layers
import tensorflow as tf


def parametric_relu(x, name="P_relu"):
    with tf.variable_scope(name):
        alpha = tf.get_variable('a', x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)


def pRelu(x, name="P_relu"):
    return parametric_relu(x, name)


class build_model(object):
    def __init__(self, is_training, **model_params):
        self.is_training = is_training
        self.activation_fn = pRelu
        self.num_class = model_params['num_class']
        self.num_kernel = 16
        self.pool_kernel = 4

    def res_act(self, net, name='res_act'):
        net = layers.batch_norm(net, self.is_training, name=name + '_bn')
        net = self.activation_fn(net, name=name + '_act')
        return net

    def res_block(self, net, output_dim, is_downsizing=True, name="res_block"):
        with tf.variable_scope(name):
            net = self.res_act(net, name='act1')

            if is_downsizing:
                net = layers.conv2d(net, output_dim, k_size=(2, 2), activation_fn=None, normalization=None, name='conv_1')
                btl = layers.bottleneck(net, output_dim, k_size=(2, 2), name='skip1')
            else:
                net = layers.conv2d(net, output_dim, k_size=(1, 1), activation_fn=None, normalization=None, name='conv_1')
                btl = layers.bottleneck(net, output_dim, k_size=(1, 1), name='skip1')

            net = layers.conv2d(net, output_dim, activation_fn=None, normalization=None, name='conv_2')
            skip = tf.add(btl, net, name='res1')

            net = self.res_act(skip, name='act2')
            net = layers.conv2d(net, output_dim, strides=(1, 1), activation_fn=self.activation_fn, normalization=True, name='conv_3')
            net = layers.conv2d(net, output_dim, strides=(1, 1), activation_fn=None, normalization=None, name='conv_4')

            net = tf.add(skip, net, name='res2')

            return net

    def inference(self, net):
        net = layers.conv2d(net, self.num_kernel, name='conv1')

        net = self.res_block(net, self.num_kernel * 2, is_downsizing=False, name='res_block1')
        net = self.res_block(net, self.num_kernel * 4, is_downsizing=True, name='res_block2')
        net = self.res_block(net, self.num_kernel * 8, is_downsizing=True, name='res_block3')

        net = self.res_act(net)
        net = layers.avg_pool(net, k_size=(4, 4), strides=(1, 1), name='pool')
        net = layers.flatten(net, 'flat')
        net = layers.linear(net, self.num_class, name='linear')
        return net
