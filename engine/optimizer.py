from datetime import datetime
import os

import numpy as np
import tensorflow as tf

from ..utils.tensorboard_utils import *
from .training_arrays import training_loop
from .network import Network

from JuNet.data.data_manager import DataManager
from JuNet.neuralnetwork.builder import losses, evaluate
from JuNet.neuralnetwork.model.segmentation import FusionNet_3D, UNet_3D
from JuNet.neuralnetwork.model.classification import vgg, WideResNet
from JuNet.neuralnetwork import reporter

models = {'vgg': vgg,
          'wideresnet': WideResNet,
          'fusionnet': FusionNet_3D,
          'unet': UNet_3D}
loss_fn = {'dice': losses.loss_dice_coef,
           'ce': losses.cross_entropy_loss}
optimizers = {'adam': tf.train.AdamOptimizer}
evaluates = {'dice', evaluate.eval_dice_coef}


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expaneded_g = tf.expand_dims(g, axis=0)
            grads.append(expaneded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


class Optimizer(Network):
    def __init__(self, project_title='temp', work_dir='c://workspace/', data_ext='tfrecords'):
        self.project_title = project_title
        self.work_dir = work_dir
        self.project_dir = os.path.join(work_dir, project_title)
        self.data_ext = data_ext
        print('Segmentation initial loaded')

    def compile(self, subject_title='temp', mode='train', learning_rate=0.0001, decay_step=3, lr_decay_factor=1.,
                      num_epochs=50, batch_size=4, input_shape=None, inbound_shape=None, grid_n=1, valid_rate=0.1,
                      model_name='vgg', loss_name='ce', optimizer='adam', evaluator='dice', **model_params):
        print("Optimizer Setting is Ready")

        self.train_params = locals()
        del self.train_params['self']
        print(self.train_params)

        self.MOVING_AVERAGE_DECAY = 0.9999

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.inbound_shape = inbound_shape
        self.grid_n = grid_n
        self.valid_rate = valid_rate
        self.subject_title = subject_title

        self.is_training = mode == 'train'
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.lr_decay_factor = lr_decay_factor
        self.num_epochs = num_epochs

        self.model_name = model_name
        self.loss_function = loss_fn[loss_name]
        self.optimizer = optimizers[optimizer]
        # self.evaluator = evaluates[evaluator]
        self.model_params = model_params

        # Start Training
        self.reports = reporter.Reporter(self.project_dir, self.subject_title, self.train_params, self.model_params,
                                         self.model_params)

    def fit(self, report_per_epoch=100, save_per_epoch=1, valid_per_epoch=50, evals_per_epoch=1, verbosity=3, **train_params):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            self.model_params['num_class'] = self.num_class
            is_training = tf.placeholder_with_default(tf.constant(True), None, name='is_training')

            handle = tf.placeholder(tf.string, shape=[], name='handle')
            images, labels, datainfo = self.get_data(handle)
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            decay_steps = int(self.batches_per_epoch * self.decay_step)
            lr = tf.train.exponential_decay(self.learning_rate,
                                            global_step,
                                            decay_steps,
                                            self.lr_decay_factor,
                                            staircase=True)
            opt = self.optimizer(lr)

            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_device):
                    with tf.device('/%s:%d' % (self.train_device, i)):
                        with tf.name_scope('%s_%d' % (self.model_name, i)) as scope:
                            train_model = models[self.model_name].build_model(is_training, self.model_params)
                            logits = train_model.inference(images[i])
                            model_loss = self.loss_function(predictions=logits, groundtruth=labels[i])
                            model_accr = tf.reduce_mean(
                                tf.metrics.accuracy(tf.argmax(labels[i], -1), tf.argmax(logits, -1)), name='accuracy')

                            logit_value = tf.argmax(logits, -1)
                            if len(labels[i].get_shape().as_list()) > 3:
                                save_image_summary(images[i], name='image')

                                save_image_summary(logit_value, name='logit')
                                save_image_summary(tf.where(tf.equal(logit_value, 0),
                                                            tf.cast(logit_value, tf.float32),
                                                            tf.cast(tf.reshape(images[i],
                                                                               images[i].get_shape().as_list()[:-1]),
                                                                    tf.float32)),
                                                   name='masked_image')
                                save_image_summary(tf.argmax(labels[i], -1), name='label')

                            loss = save_scalar_summary(scope, model_loss, 'loss')
                            accr = save_scalar_summary(scope, model_accr, 'accr')

                            tf.get_variable_scope().reuse_variables()
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_gradient_op, variables_averages_op)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            # saver = tf.train.Saver(tf.global_variables())
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=False,
                                         gpu_options=tf.GPUOptions(
                                             force_gpu_compatible=True,
                                             allow_growth=True))

            checkpoint_dict = {}
            checkpoint_dict['checkpoint_dir'] = os.path.join(self.reports.save_path)
            checkpoint_dict['checkpoint_path'] = os.path.join(self.reports.save_path, 'model.ckpt')

        return training_loop(train_op,
                             init_op,
                             sess_config,
                             checkpoint_dict,
                             handle)
