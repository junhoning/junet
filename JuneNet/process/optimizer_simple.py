import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from JuneNet.model.builder import losses
from JuneNet.model.segmentation import FusionNet_3D

model = {'fusionnet': FusionNet_3D}
loss_fn = {'dice': losses.dice_coef}
optm_fn = {'adam': tf.train.AdamOptimizer}


MOVING_AVERAGE_DECAY = 0.9999


def tower_loss(scope, images, labels):
    logits = model[FLAGS.model_name].build_model().inference(images)

    loss = loss_fn[FLAGS.loss_name](predictions=logits, groundtruth=labels)

    loss_collection = tf.get_collection('losses', scope)

    total_loss = tf.add_n(loss_collection, name='total_loss')

    for l in loss_collection + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % FLAGS.model_name, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expaneded_g = tf.expand_dims(g, 0)
            grads.append(expaneded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


def train(project_title='temp', data_manager=None, learning_rate=0.0001, num_epochs=50, batch_size=4, optimizer='dice',
          loss_name='ce', aug_list={}, model_name='unet', **params):
    if data_manager is not None:
        datamanager = data_manager
    else:
        datamanager = data_manager.DataInput(batch_size=batch_size, project_title=project_title, **params)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = int(datamanager.batches_per_epoch * params['num_epochs_per_decay'])
        lr = tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        decay_steps,
                                        params['lr_decay_factor'],
                                        staircase=True)

        opt = optm_fn[optimizer](lr)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(datamanager.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model_name, i)) as scope:
                        images = datamanager.image
                        labels = datamanager.label
                        loss = tower_loss(scope, images, labels)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        # We must calculate the mean of each gradient. Note that this is the
                        # synchronization point across all towers.

        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=params['log_device_placement'])

        with tf.Session(config=sess_config) as sess:
            sess.run(init)

            summary_writer = tf.summary.FileWriter(datamanager.work_dir, sess.graph)

            for epoch in range(num_epochs):
                for step in range(datamanager.batches_per_epoch):
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_examples_per_step = batch_size * datamanager.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / datamanager.num_gpus

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                                            examples_per_sec, sec_per_batch))

                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(datamanager.work_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpus', type=str, default=None,
                        help='Number of GPUs')

    parser.add_argument('--project_title', type=str, default='SegEngine',
                        help='Title of Project')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of Images to process in Batch')

    parser.add_argument('--grid_n', type=int, default=1,
                        help='Number of Grid for cropping images')

    parser.add_argument('--input_shape', nargs='+', type=int, default=(64, 64, 64),
                        help='Input Size You want to crop from the Original Size')

    parser.add_argument('--data_ext', type=str, default='tfrecords',
                        help='File Ext of the target data file')

    parser.add_argument('--aug_list', nargs='+', type=str, default='',
                        help='Augmentation List')

    parser.add_argument('--inbound_shape', nargs='+', type=int, default=parser.parse_args().input_shape,
                        help='Inbound Range to Limit the Grid from Original Data')

    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial Learning Rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of Epochs')
    parser.add_argument('--model_name', type=str, default='segmentation',
                        help='Name of Model')
    parser.add_argument('--loss_name', type=str, default='dice',
                        help='Name of Loss Function')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Name of Optimization')
    parser.add_argument('--num_epochs_per_decay', type=int, default=3,
                        help='Number of Epochs per Decay')
    parser.add_argument('--lr_decay_factor', type=float, default=1,
                        help='Learning Rate decay factor')

    parser.add_argument('--loss-name', type=str, default='ce',
                        help='Name of Loss Name')

    FLAGS = parser.parse_args()

    params = {'grid_n': FLAGS.grid_n,
              'input_shape': FLAGS.input_shape,
              'inbound_shape': FLAGS.inbound_shape,
              'lr_decay_factor': FLAGS.lr_decay_factor,
              'num_epochs_per_decay': FLAGS.num_epochs_per_decay,
              'weight_decay': FLAGS.weight_decay,
              'data_ext': FLAGS.data_ext,
              'work_dir': FLAGS.work_dir,
              'data_dir': FLAGS.data_dir,
              'num_gpus': FLAGS.num_gpus,
              'log_device_placement': FLAGS.log_device_placement}

    train(learning_rate=FLAGS.learning_rate, num_epochs=FLAGS.num_gpus, batch_size=FLAGS.batch_size,
          project_title=FLAGS.project_title, optimizer=FLAGS.optimizer, loss_name=FLAGS.loss_name,
          aug_list=FLAGS.aug_list, model_name=FLAGS.model_name, **params)