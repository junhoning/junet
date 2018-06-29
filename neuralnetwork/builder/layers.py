from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm as batch_normalization

from ..builder import acts


initializer = tf.contrib.layers.xavier_initializer()
act_fn = acts.pRelu


def conv2d(layer_input, output_dim, k_size=(3, 3), strides=(1, 1), activation_fn=act_fn, normalization=True, is_training=True, name='conv2d'):
    with tf.variable_scope(name):
        weight = tf.get_variable('w', list(k_size) + [layer_input.get_shape()[-1], output_dim], initializer=initializer)
        conv = tf.nn.conv2d(layer_input, weight, strides=[1] + list(strides) + [1], padding='SAME', name='conv')
        bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
        conv = tf.nn.bias_add(conv, bias)

        if normalization is not None:
            conv = batch_norm(conv, train_phase=is_training, name='batch_norm')
        if activation_fn is not None:
            conv = activation_fn(conv, 'act')

        return conv


def conv2d_trans(input_, output_shape,  k_size=(5, 5), strides=(2, 2), record_tboard=True, name="conv2dTrans"):
    with tf.variable_scope(name):
        deconv_shape = tf.stack([tf.shape(input_)[0], input_.get_shape()[1] * 2, input_.get_shape()[2] * 2, output_shape[-1]])
        w = tf.get_variable('w', list(k_size) + [output_shape[-1], input_.get_shape()[-1]], initializer=initializer)
        conv = tf.nn.conv2d_transpose(input_, w, output_shape=deconv_shape, strides=[1] + list(strides) + [1], padding="SAME")
        bias = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0))
        conv_b = tf.nn.bias_add(conv, bias)
        if record_tboard:
            tf.summary.histogram("weight_trans", w)
        return conv_b


def conv3d(self, input, out_channel, k_size=(3, 3, 3), strides=(1, 1, 1), padding='VALID', name='conv3d_layer'):
    with tf.variable_scope(name):
        in_channel = input.get_shape()[-1]
        w = tf.get_variable('w',  list(k_size) + [in_channel, out_channel], initializer=initializer)
        conv = tf.nn.conv3d(input, w, strides=[1] + list(strides) + [1], padding=padding, name='conv')
        bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0))
        conv_b = tf.nn.bias_add(conv, bias)

        if self.record_tboard:
            tf.summary.histogram("weight", w)

        return conv_b


def conv3d_trans(self, input, k_size=(2, 2, 2), strides=(2, 2, 2), name="conv3d_trans"):
    with tf.variable_scope(name):
        deconv_shape = tf.stack(
            [tf.shape(input)[0], input.get_shape()[1] * 2, input.get_shape()[2] * 2, input.get_shape()[3] * 2, input.get_shape()[-1]])
        w = tf.get_variable('w', list(k_size) + [input.get_shape()[-1], input.get_shape()[-1]], initializer=initializer)
        conv = tf.nn.conv3d_transpose(input, w, output_shape=deconv_shape, strides=[1] + list(strides) + [1], padding="SAME")
        bias = tf.get_variable('biases', [input.get_shape()[-1]], initializer=tf.constant_initializer(0))
        conv_b = tf.nn.bias_add(conv, bias)

        if self.record_tboard:
            tf.summary.histogram("weight_trans", w)
        return conv_b


def batch_norm(inputs, train_phase=True, name="batch_norm"):
    if type(train_phase) == bool:
        train_phase = tf.constant(train_phase, dtype=tf.bool, shape=[])

    train_bn = batch_normalization(inputs=inputs, decay=0.9, updates_collections=None, zero_debias_moving_mean=True,
                                   is_training=True, reuse=None, scale=True, epsilon=1e-5, trainable=True, scope=name)
    test_bn = batch_normalization(inputs=inputs, decay=0.9, updates_collections=None, zero_debias_moving_mean=True,
                                  is_training=False, reuse=True, scale=True, epsilon=1e-5, trainable=True, scope=name)

    return tf.cond(train_phase, lambda: train_bn, lambda: test_bn)


def bottleneck(input_, output_dim, k_size=(1, 1), strides=(1, 1), is_training=True, name='bottleneck'):
    return conv2d(input_, output_dim, k_size, strides,
                  activation_fn=act_fn,
                  is_training=is_training,
                  name=name)


def linear(input_, output_size, stddev=0.02, bias=0.0, name="linear"):
    if len(input_.get_shape()) > 2:
        input_ = flatten(input_)

    input_size = input_.get_shape().as_list()[1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [input_size, output_size], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(bias))
        logits = tf.nn.xw_plus_b(input_, w, biases, name="logits")
        return logits


def flatten(input_, name="flatten"):
    vec_dim = input_.get_shape()[1:]
    n = vec_dim.num_elements()
    with tf.name_scope(name):
        return tf.reshape(input_, [-1, n])


def drop_out(x, keep_prob, is_training, name="drop_out"):
    with tf.variable_scope(name):
        dropout = tf.nn.dropout(x, keep_prob)
        full_dropout = tf.nn.dropout(x, 1.)
        return tf.cond(is_training, lambda: dropout, lambda: full_dropout)


def max_pool(input_, k_size=(2, 2), strides=(2, 2), padding='SAME', name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(input_, ksize=[1] + list(k_size) + [1], strides=[1] + list(strides) + [1], padding=padding)


def avg_pool(input_, k_size=(2, 2), strides=(2, 2), padding='SAME', name="pool"):
    with tf.name_scope(name):
        return tf.nn.avg_pool(input_, ksize=[1] + list(k_size) + [1], strides=[1] + list(strides) + [1], padding=padding)


def fully_connected(net, out_dim):
    net = tf.reduce_mean(net, [1, 2], name='avg_pool')
    final_dim = net.get_shape()[1:].num_elements()
    net = tf.reshape(net, shape=[-1, final_dim])
    weight = tf.get_variable('DW', [final_dim, out_dim],
                             initializer=tf.initializers.variance_scaling(1.0))
    bias = tf.get_variable('biases', [out_dim],
                           initializer=tf.constant_initializer())
    net = tf.nn.xw_plus_b(net, weight, bias)
    return net


########## Block Series ##########
def residual_block(input_, output_dim, activation_fn=act_fn, is_training=True, name="residual_block"):
    with tf.variable_scope(name):
        conv1 = conv2d(input_, output_dim, activation_fn=activation_fn, name="conv1")
        with tf.variable_scope('inside_block'):
            conv2 = conv2d(conv1, output_dim, activation_fn=activation_fn, is_training=is_training, name='conv_2')
            conv3 = conv2d(conv2, output_dim, activation_fn=activation_fn, is_training=is_training, name='conv_3')
            conv4 = conv2d(conv3, output_dim, activation_fn=None, is_training=is_training, name='conv_4')
            skip_connect = tf.add(conv1, conv4, name='skip_connection')
        conv5 = conv2d(skip_connect, output_dim, activation_fn=activation_fn, name="conv_5")
        return conv5


def dense_block(input_, output_dim, dense_repeat_num=3, activation_fn=act_fn, is_training=True, name='dense_block'):
    with tf.variable_scope(name):
        conv1 = conv2d(input_, output_dim, activation_fn=act_fn, is_training=is_training, name='conv_1')
        nodes = [conv1]
        for i in range(dense_repeat_num):
            if i == dense_repeat_num-1:
                activation_fn = None
            conv_name = "conv_%d" % (i+2)
            conv = conv2d(tf.concat(nodes, axis=3), output_dim, activation_fn=activation_fn, is_training=is_training, name=conv_name)
            nodes.append(conv)
        conv = conv2d(conv, output_dim, strides=(2, 2), activation_fn=act_fn, is_training=is_training,
                      name='conv_%d' % (dense_repeat_num+2))
        return conv


########### Custom Layer #############
def median_pool(inputs, k_size):
    patches = tf.extract_image_patches(inputs, [1, k_size, k_size, 1], ...)
    m_idx = int(k_size * k_size / 2 + 1)
    top = tf.nn.top_k(patches, m_idx, sorted=True)
    median = tf.slice(top, [0, 0, 0, m_idx - 1], [-1, -1, -1, 1])
    return median

