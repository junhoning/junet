from datetime import datetime
import os
import re

import numpy as np
import tensorflow as tf

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
    if len(image.get_shape().as_list()) < 4:
        image = tf.reshape(image, image.get_shape().as_list() + [1])

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


class Optimizer(DataManager):
    def __init__(self, project_title='temp', work_dir='c://workspace/', data_ext='tfrecords', num_gpus=None,
                 tboard_record=True):
        self.project_title = project_title
        self.work_dir = work_dir
        self.project_dir = os.path.join(work_dir, project_title)
        self.num_gpus = num_gpus
        self.data_ext = data_ext
        self.tboard_record = tboard_record
        print('Segmentation initial loaded')

    def set_optimizer(self, subject_title='temp', mode='train', learning_rate=0.0001, decay_step=3, lr_decay_factor=1.,
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

    def optimization_load(self):
        return NotImplemented

    def train(self, report_per_epoch=100, save_per_epoch=1, valid_per_epoch=50, evals_per_epoch=1, verbosity=3, **train_params):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            DataManager.__init__(self, self.project_title, self.batch_size, self.input_shape, self.inbound_shape,
                                 self.grid_n, 'GZIP', self.valid_rate, self.num_gpus, self.data_ext, self.work_dir, **train_params)
            self.model_params['num_class'] = self.num_class
            is_training = tf.placeholder_with_default(tf.constant(True), None, name='is_training')

            handle = tf.placeholder(tf.string, shape=[])
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

            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_gradient_op, variables_averages_op)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            summary_op = tf.summary.merge(summaries)
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=False,
                                         gpu_options=tf.GPUOptions(
                                             force_gpu_compatible=True,
                                             allow_growth=True))

            checkpoint_dir = os.path.join(self.reports.save_path)
            checkpoint_path = os.path.join(self.reports.save_path, 'model.ckpt')

            print('Number of Datas : Train %d / Validation %d' % (len(self.train_list), len(self.valid_list)))
            print("Start Training at ", datetime.now())
            with tf.Session(config=sess_config) as sess:
                sess.run(init_op)
                summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
                eval_summary_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval/'), sess.graph)

                train_handle = sess.run(self.train_iterator.string_handle())
                valid_handle = sess.run(self.valid_iterator.string_handle())
                glob_step = -1
                for epoch in range(self.num_epochs):
                    sess.run(self.valid_iterator.initializer)
                    loss_sum = 0; avg_step = 0; accr_sum = 0
                    for step in range(self.batches_per_epoch):
                        glob_step += 1
                        train_feed = {handle: train_handle}
                        sess.run(train_op, feed_dict=train_feed)

                        # Write Report State and Print Loss and Accuracy
                        if step in np.linspace(1, self.batches_per_epoch-1, report_per_epoch).astype(int):
                            state = {}
                            if verbosity > 1:
                                loss_value, accr_value = sess.run([loss, accr], train_feed)
                                loss_sum += loss_value; accr_sum += accr_value
                                avg_step += 1
                                # 'dice': dice_sum/avg_step, 'test_dice': test_dice_sum/avg_step,
                                state['cost'] = loss_sum/avg_step
                                state['accr'] = accr_sum/avg_step

                            self.reports.report_state(state, glob_step, self.num_epochs, epoch, self.batches_per_epoch, step)

                            summary_str = sess.run(summary_op, train_feed)
                            summary_writer.add_summary(summary_str, glob_step)

                        if step in np.linspace(0, self.batches_per_epoch-1, valid_per_epoch+1)[1:].astype(int):
                            valid_feed = {handle: valid_handle, is_training: False}
                            _, _, eval_summary = sess.run([loss, accr, summary_op], valid_feed)
                            eval_summary_writer.add_summary(eval_summary, glob_step)
                            # loss_value, accr_value = sess.run([loss, accr], valid_feed)

                        # if step in np.linspace(0, self.batches_per_epoch-1, save_per_epoch+1)[1:].astype(int):
                        #     saver.save(sess, checkpoint_path, 0)

                        if step in np.linspace(0, self.batches_per_epoch - 1, evals_per_epoch + 1)[1:].astype(int):
                            print("\nStart Evaluating")
                            sess.run(self.valid_iterator.initializer)

                            import scipy.misc
                            save_path = self.reports.save_path
                            valid_feed = {handle: valid_handle, is_training: False}
                            target, groundtruth, logit_result = sess.run([images[i], labels[i], logits], valid_feed)
                            result = np.argmax(logit_result, 3)
                            label = np.argmax(groundtruth, 3)
                            for n in range(8):
                                scipy.misc.toimage(target[n, :, :, 0], cmin=0, cmax=255).save(
                                    os.path.join(save_path, 'image_%d_%d.jpg' % (epoch, n)))
                                scipy.misc.toimage(result[n], cmin=0, cmax=1).save(
                                    os.path.join(save_path, 'result_%d_%d.jpg' % (epoch, n)))
                                scipy.misc.toimage(label[n], cmin=0, cmax=1).save(
                                    os.path.join(save_path, 'label_%d_%d.jpg' % (epoch, n)))

                print("Done training at : ", datetime.now(), "\n")
