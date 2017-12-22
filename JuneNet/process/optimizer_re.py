import time
from datetime import datetime
import os
import re

import numpy as np
import tensorflow as tf
from JuneNet.process.data.data_manager import DataManager

from JuneNet.model.builder import losses, evaluate
from JuneNet.model.segmentation import FusionNet_3D, UNet_2D
from JuneNet.model.classification import vgg
from JuneNet.process.neuralnetwork import Trainer
from JuneNet.process import utils

segmentation_model = {'fusionnet': FusionNet_3D,
                      'unet': UNet_2D}

classification_model = {'vgg': vgg}


loss_fn = {'dice': losses.dice_coef,
           'ce': losses.cross_entropy_loss}
optimizers = {'adam': tf.train.AdamOptimizer}
evaluates = {'dice', evaluate.eval_dice_coef}

MOVING_AVERAGE_DECAY = 0.9999


class NeuralNetwork(DataManager):
    def __init__(self, project_title='temp', work_dir='c://workspace/', data_ext='tfrecords', num_gpus=None,
                 tboard_record=True, log_device_placement=False):
        self.project_title = project_title
        self.work_dir = work_dir
        self.num_gpus = num_gpus
        self.data_ext = data_ext
        self.tboard_record = tboard_record
        self.log_device_placement = log_device_placement
        print('Segmentation initial loaded')

    def optimizer(self, mode='train', learning_rate=0.0001, decay_step=3, lr_decay_factor=1.,
                  num_epochs=50, batch_size=4, input_shape=None, inbound_shape=None, grid_n=1,
                  model_name='vgg', loss_name='ce', optimizer='adam', evaluator='dice'):

        print("Optimizer Get Ready")
        is_training = mode == 'train'
        DataManager.__init__(self, self.project_title, batch_size, input_shape, inbound_shape,
                             grid_n, self.num_gpus, self.data_ext, self.work_dir)

        if self.num_gpus == 0:
            return self.cpu_optimizer(is_training, learning_rate, decay_step, lr_decay_factor,
                                      model_name, loss_name, optimizer, evaluator, self.log_device_placement)
        else:
            return self.multi_gpu_optimizer(is_training, learning_rate, decay_step, lr_decay_factor,
                                            model_name, loss_name, optimizer, evaluator, self.log_device_placement)

    def cpu_optimizer(self, is_training, learning_rate, decay_step, lr_decay_factor, model_name,
                      loss_name, optimizer, evaluator, log_device_placement, **model_params):
            logits = classification_model[model_name].build_model(is_training, **model_params).inference(self.images)
            loss = loss_fn[loss_name](predictions=logits, groundtruth=self.labels)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            with tf.variable_scope("cost"):
                self.cost = loss_fn[loss_name](logits, self.labels)
                if self.tboard_record:
                    summaries.append(tf.summary.scalar("cost", self.cost))

            with tf.variable_scope("corr"):
                # predmax = tf.argmax(logits, self.data_dim+1)
                # ymax = tf.argmax(self.labels, self.data_dim+1)
                predmax = tf.argmax(logits, -1)
                ymax = tf.argmax(self.labels, -1)
                corr = tf.equal(predmax, ymax)

            with tf.variable_scope("evaluation"):
                accr = tf.reduce_mean(tf.cast(corr, 'float'))
                # dice = evaluates[evaluator].eval_dice_coef(logits, self.labels)

                if self.tboard_record:
                    summaries.append(tf.summary.scalar('accuracy', accr))
                    # tf.summary.scalar('dice', dice)

            with tf.variable_scope("Optimizer"):
                # Optimizer
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
                var_list = tf.trainable_variables()
                decay_steps = int(self.batches_per_epoch * decay_step)
                lr = tf.train.exponential_decay(learning_rate,
                                                global_step,
                                                decay_steps,
                                                lr_decay_factor,
                                                staircase=True)
                optimizer = optimizers[optimizer](lr)
                grads_and_vars = optimizer.compute_gradients(loss=self.cost, var_list=var_list)
                apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step)
                print("Optimization Ready")

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            self.train_op = tf.group(apply_gradient_op, variables_averages_op)
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.saver = tf.train.Saver(tf.global_variables())
            self.summary_op = tf.summary.merge(summaries)
            self.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)

    def multi_gpu_optimizer(self, is_training, learning_rate, decay_step, lr_decay_factor, model_name,
                            loss_name, optimizer, evaluator, log_device_placement, **model_params):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            decay_steps = int(self.batches_per_epoch * decay_step)
            lr = tf.train.exponential_decay(learning_rate,
                                            global_step,
                                            decay_steps,
                                            lr_decay_factor,
                                            staircase=True)
            opt = optimizers[optimizer](lr)

            train_model = classification_model[model_name].build_model(is_training, **model_params)
            loss_function = loss_fn[loss_name]

            def single_cpu_optimizer(train_model, loss_function, images, labels):
                logits = train_model.inference(images)
                loss = loss_function(predictions=logits, groundtruth=labels)

                var_list = tf.trainable_variables()
                grads_and_vars = opt.compute_gradients(loss=self.cost, var_list=var_list)
                apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step)

            def multi_gpu_optimizer(train_model, loss_function, images, labels, opt):
                tower_grads = []
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('%s_%d' % (model_name, i)) as scope:
                                logits = train_model.inference(images)
                                _ = loss_function(predictions=logits, groundtruth=labels)
                                loss_collection = tf.get_collection('losses', scope)
                                total_loss = tf.add_n(loss_collection, name='total_loss')
                                grads = opt.compute_gradients(total_loss)

                                tf.get_variable_scope().reuse_variables()
                                tower_grads.append(grads)

                                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                                for l in loss_collection + [total_loss]:
                                    loss_name = re.sub('%s_[0-9]*/' % 'model', '', l.op.name)
                                    tf.summary.scalar(loss_name, l)

                grads = utils.average_gradients(tower_grads)
                apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
                for grad, var in grads:
                    if grad is not None:
                        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                return total_loss, summaries, apply_gradient_op

            self.loss, summaries, apply_gradient_op = multi_gpu_optimizer(train_model, loss_function, opt)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            self.train_op = tf.group(apply_gradient_op, variables_averages_op)
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.saver = tf.train.Saver(tf.global_variables())
            self.summary_op = tf.summary.merge(summaries)
            self.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)

    def train(self, subject_title='temp', num_epochs=50):
        checkpoint_dir = os.path.join(self.work_dir, self.project_title, subject_title)
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        with tf.Session(config=self.sess_config) as sess:
            sess.run(self.init_op)

            summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
            print('Training Start at', datetime.now())
            for epoch in range(num_epochs):
                for step in range(self.batches_per_epoch):
                    start_time = time.time()
                    _, loss_value = sess.run([self.train_op, self.cost])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_device = 1 if self.num_gpus == 0 else self.num_gpus
                        num_examples_per_step = self.batch_size * num_device
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / (num_device)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                                            examples_per_sec, sec_per_batch))

                    if step % 20 == 0:
                        summary_str = sess.run(self.summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 500 == 0 or (step + 1) == (num_epochs * self.batches_per_epoch):
                        self.saver.save(sess, checkpoint_path, global_step=step)
