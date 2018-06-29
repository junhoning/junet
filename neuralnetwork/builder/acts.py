import tensorflow as tf


def pRelu(x, name="P_relu"):
    with tf.name_scope(name):
        alpha = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)


def maxout_(x, num_param=5, name="maxout"):
    with tf.name_scope(name):
        output = []
        # print(x.get_shape(), x.get_shape().as_list(), x.get_shape()[-1])
        for i in range(num_param):
            name = 'w_%d' % i
            w = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(1.0 * (i-num_param/2)))
            name = 'b_%d' % i
            b = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(i-num_param/2))
            out = x*w + b
            # print(out.get_shape())
            output.append(out)

        ret = tf.reduce_max(output, 0)
        print(ret.get_shape())
        return ret


def maxout(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def leaky_relu(x, leak=0.01, name="leaky_relu"):
    with tf.name_scope(name):
        # return tf.maximum(x, leak * x)
        return tf.maximum(0, x) + leak * tf.minimum(0, x)


def lRelu(x, leak=0.01, name="leaky_relu"):
    return leaky_relu(x, leak, name)


def elu(x, name="softplus"):
    with tf.name_scope(name):
        return tf.nn.elu(x)


def absolute_relu(x, name="absolute_relu"):
    with tf.name_scope(name):
        return tf.maximum(x, -x)


# SWISH: A SELF-GATED ACTIVATION FUNCTION
# https://arxiv.org/pdf/1710.05941.pdf
def swish(x, name='swish_act'):
    with tf.name_scope(name):
        return x * tf.nn.relu(x)

