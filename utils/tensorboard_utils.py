import re

import tensorflow as tf
import numpy as np


def save_scalar_summary(scope, value, name):
    tf.add_to_collection(name, value)
    tf.add_n(tf.get_collection(name), name='total_%s' % name)

    value_collection = tf.get_collection(name, scope)
    total_value = tf.add_n(value_collection, name='total_%s' % name)

    for scalar in value_collection + [total_value]:
        valid_name = re.sub('%s_[0-9]*/' % 'model', '', scalar.op.name)
        tf.summary.scalar(valid_name, scalar)

    return total_value


def save_image_summary(image, name='image'):
    max_outputs = 3

    tf.summary.histogram(name, image)

    if name != 'image':
        sliced_image = tf.cast(image, tf.float32)
        image_maximum = tf.ones_like(sliced_image, tf.float32) * tf.reduce_max(sliced_image)
        image = sliced_image * (255 / tf.cast(image_maximum, tf.float32))

    tf.summary.image(name=name, tensor=image, max_outputs=max_outputs)


def save_image_summary_3d(image, name='image'):
    max_outputs = 3

    tf.summary.histogram(name, image)
    if len(image.get_shape().as_list()) < 4:
        image = tf.reshape(image, tf.shape(image)[0] + image.get_shape().as_list()[1:] + [1])

    image_shape = image.get_shape().as_list()
    image_shape[0] = max_outputs
    sliced_image = tf.slice(tf.cast(image, tf.uint8),
                            [0 for _ in image_shape], image_shape,
                            name='sliced_%s' % name)

    if name != 'image':
        sliced_image = tf.cast(sliced_image, tf.float32)
        image_maximum = tf.ones_like(sliced_image, tf.float32) * tf.reduce_max(sliced_image)
        sliced_image = sliced_image * (255 / tf.cast(image_maximum, tf.float32))
    tf.summary.image(name=name, tensor=tf.cast(sliced_image, tf.uint8), max_outputs=max_outputs)


def save_all_gradients(summaries, grads):
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))


# Add histograms for trainable variables.
def save_all_variables(summaries):
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))


def save_summary_scalar(summaries, scalar, name):
    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar(name, scalar)) # 'learning_rate', lr