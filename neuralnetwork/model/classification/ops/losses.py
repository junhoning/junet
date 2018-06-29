import tensorflow as tf
from JuneNet.model.classification.ops import layers


def l1_loss(input_, target_, lamb=1.0, name="l1_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        loss = tf.mul(tf.reduce_mean(tf.abs(input_ - target_)), lamb, name="loss")
        return loss


def l2_loss(input_, target_, lamb=1.0, name="l2_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        loss = tf.mul(tf.reduce_mean(tf.square(input_ - target_)), lamb, name="loss")
        return loss


def cross_entropy_loss(logits, labels, lamb=1.0, name="ce_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        loss = tf.mul(lamb, tf.reduce_mean(cross_entropy), name="loss")
        return loss


def binomial_cross_entropy_loss(logits, labels, lamb=1.0, name="bi-ce_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.multiply(lamb, tf.reduce_mean(cross_entropy), name="loss")
        return loss


def pixel_wise_l1_loss(input_, target_, lamb=1.0, name="pixel_l1"):
    num_class = input_.get_shape().as_list()[-1]
    input_reshape = tf.reshape(input_, shape=[-1, num_class])
    target_reshape = tf.reshape(target_, shape=[-1, num_class])
    return l1_loss(input_reshape, target_reshape, lamb, name)


def pixel_wise_l2_loss(input_, target_, lamb=1.0, name="pixel_l2"):
    num_class = input_.get_shape().as_list()[-1]
    input_reshape = tf.reshape(input_, shape=[-1, num_class])
    target_reshape = tf.reshape(target_, shape=[-1, num_class])
    return l2_loss(input_reshape, target_reshape, lamb, name)


def pixel_wise_cross_entropy(input_, target_, lamb=1.0, name="pixel_ce"):
    num_class = input_.get_shape().as_list()[-1]
    input_reshape = tf.reshape(input_, shape=[-1, num_class])
    target_reshape = tf.reshape(target_, shape=[-1, num_class])
    return cross_entropy_loss(input_reshape, target_reshape, lamb, name)
