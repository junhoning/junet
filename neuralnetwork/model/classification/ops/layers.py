from __future__ import absolute_import

import tensorflow as tf

default_act_fn = tf.nn.relu


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME', name="conv")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)
        return conv_b


def conv2d_valid(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID', name="conv")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)
        return conv_b


def conv2d_valid_act(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0,
                     activation_fn=default_act_fn, with_logit=False, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID', name="conv")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)

        bn = batch_norm(conv_b)

        act = activation_fn(bn, "act")

        if with_logit:
            return act, conv_b
        else:
            return act


def conv2d_valid_repeat(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, num_repeat=3, bias=0.0,
                        activation_fn=default_act_fn, name="conv_block"):
    with tf.variable_scope(name):
        for i in range(num_repeat):
            name = "conv_%d" % i
            output = conv2d_valid_act(input_, output_dim, k_h, k_w, d_h, d_w, bias,
                                      activation_fn=activation_fn, name=name)
            input_ = output
        return output


def conv2d_same(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME', name="conv")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)
        return conv_b


def conv2d_same_act(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0,
                    activation_fn=default_act_fn, with_logit=False, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME', name="conv")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)

        bn = batch_norm(conv_b)

        act = activation_fn(bn, "act")

        if with_logit:
            return act, conv_b
        else:
            return act


def conv2d_same_repeat(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, num_repeat=3,
                       activation_fn=default_act_fn, with_logit=False, name="conv_block"):
    with tf.variable_scope(name):
        output, logit = 0, 0
        for i in range(num_repeat):
            name = "conv_%d" % i
            output, logit = conv2d_same_act(input_, output_dim, k_h, k_w, d_h, d_w,
                                            activation_fn=activation_fn, with_logit=True, name=name)
            input_ = output

        if with_logit:
            return output, logit
        else:
            return output


def conv2dTrans(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0, name="conv2dTrans"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)

        return conv_b


def conv2dTrans_valid(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0, name="conv2dTrans"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding="VALID")
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)

        return conv_b


def conv2dTrans_same(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0, name="conv2dTrans"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding="SAME")
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)

        return conv_b


def conv2dTrans_valid_act(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, bias=0.0,
                          activation_fn=default_act_fn, with_logit=False, name="conv2dTrans"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding="VALID")
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        conv_b = tf.nn.bias_add(conv, biases)

        bn = batch_norm(conv_b)

        act = activation_fn(bn)

        if with_logit:
            return act, conv_b
        else:
            return act


def conv2dTrans_same_act(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, bias=0.0,
                         activation_fn=default_act_fn, with_logit=False, name="conv2dTrans"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding="SAME")
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        conv_b = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        #conv_b = tf.nn.bias_add(conv, biases)

        bn = batch_norm(conv_b)

        act = activation_fn(bn)

        if with_logit:
            return act, conv_b
        else:
            return act


def bottleneck_layer(input_, output_dim, d_h=1, d_w=1, name="bottleneck"):
    return conv2d_same(input_, output_dim, k_h=1, k_w=1, d_h=d_h, d_w=d_w, bias=0.0, name=name)


def linear(input_, output_size, stddev=0.02, bias=0.0, name="linear"):
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


def batch_norm(x, epsilon=1e-5, momentum=0.9, train=True, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x, decay=momentum,
                                            updates_collections=None, epsilon=epsilon, scale=True, scope=name)


def drop_out(x, prob, name="drop_out"):
    with tf.variable_scope(name):
        return tf.nn.dropout(x, prob)


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    return cross_entropy_mean


def eval(logits, labels, top=1):
    predictions = tf.argmax(logits, 1)
    correct = tf.equal(predictions, tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float64), name="accuracy")


def max_pool(input_, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding=padding)


def avg_pool(input_, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name="pool"):
    with tf.name_scope(name):
        return tf.nn.avg_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding=padding)