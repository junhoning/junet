from JuneNet.model.classification import ops
from JuneNet.model.builder import layers
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

    def res_act(self, input_, name='res_act'):
        bn = layers.batch_norm(input_, self.is_training, name=name + '_bn')
        act = self.activation_fn(bn, name=name + '_act')
        return act

    def res_block(self, input_, output_dim, is_downsizing=True, name="res_block"):
        with tf.variable_scope(name):
            act1 = self.res_act(input_, name='act1')

            if is_downsizing:
                skip2 = ops.layers.bottleneck_layer(act1, output_dim, d_h=2, d_w=2, name='skip1')
                _, conv1 = ops.layers.conv2d_same_act(act1, output_dim, d_h=2, d_w=2,
                                                  activation_fn=self.activation_fn, with_logit=True, name='conv1')
            else:
                skip2 = ops.layers.bottleneck_layer(act1, output_dim, d_h=1, d_w=1, name='skip1')
                _, conv1 = ops.layers.conv2d_same_act(act1, output_dim, d_h=1, d_w=1,
                                                  activation_fn=self.activation_fn, with_logit=True, name='conv1')
            conv2 = ops.layers.conv2d_same(conv1, output_dim, name='conv2')
            res1 = tf.add(skip2, conv2, name='res1')

            act2 = self.res_act(res1, name='act2')
            _, conv3 = ops.layers.conv2d_same_repeat(act2, output_dim, num_repeat=2, d_h=1, d_w=1,
                                                 activation_fn=self.activation_fn, with_logit=True, name='conv3')
            res2 = tf.add(res1, conv3, name='res2')

            return res2

    def inference(self, input_):
        conv1 = ops.layers.conv2d_same(input_, self.num_kernel, name='conv1')

        res_block1 = self.res_block(conv1, self.num_kernel * 2, is_downsizing=False, name='res_block1')
        res_block2 = self.res_block(res_block1, self.num_kernel * 4, is_downsizing=True, name='res_block2')
        res_block3 = self.res_block(res_block2, self.num_kernel * 8, is_downsizing=True, name='res_block3')

        act = self.res_act(res_block3)
        pool = ops.layers.avg_pool(act, k_h=self.pool_kernel, k_w=self.pool_kernel, d_h=1, d_w=1, name='pool')
        flat = ops.layers.flatten(pool, 'flat')

        linear = ops.layers.linear(flat, self.num_class, name='linear')
        return linear