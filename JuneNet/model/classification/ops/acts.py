import tensorflow as tf

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


def aRelu(x, name="absolute_relu"):
    return absolute_relu(x, name)


def square_act(x, name="square_act"):
    with tf.name_scope(name):
        return x*x


def square(x, name="square_act"):
    return square_act(x, name)


def linear_act(x, name="linear_act"):
    with tf.name_scope(name):
        return x


def linear(x, name="linear_act"):
    return linear_act(x, name)


def parametric_relu(x, name="P_relu"):
    with tf.variable_scope(name):
        alpha = tf.get_variable('a', x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)


def pRelu(x, name="P_relu"):
    return parametric_relu(x, name)


def maxout(x, num_param=5, name="maxout"):
    with tf.variable_scope(name):
        output = []
        for i in range(num_param):
            name = 'w_%d' % i
            w = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(1.0 * (i-num_param/2)))
            name = 'b_%d' % i
            b = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(i-num_param/2))
            out = x*w + b
            output.append(out)

        ret = tf.reduce_max(output, 0)
        return ret


def th_relu(x, th=0.01, name="th_relu"):
    with tf.name_scope(name):
        return tf.select(tf.less(x, th), tf.zeros_like(x), x)


def tRelu(x, th=0.01, name="the_relu"):
    return th_relu(x, th, name)
