import os
import numpy as np
import tensorflow as tf
import shutil

from .. import losses
from .training_generator import training_loops
from ..utils import tensorboard_utils
from .network import Network


class Model(Network):
    def compile(self, subject_title, dataset, loss_fn, metric_fn):
        self.subject_title = subject_title
        self.dataset = dataset
        self.loss_fn = loss_fn
        if not isinstance(metric_fn, list):
            metric_fn = [metric_fn]
        self.metric_fn = metric_fn

    def fit(self, model, learning_rate, epochs=None, mode='train'):
        # g = tf.Graph()
        # # with g.as_default():
        with tf.name_scope('model') as scope:
            is_training = tf.placeholder_with_default(tf.constant(True), None, name='is_training')

            image = tf.placeholder(dtype=tf.float32, shape=[None] + self.dataset.input_shape + [3], name='image')
            label = tf.placeholder(dtype=tf.float32, shape=[None] + self.dataset.input_shape + [2], name='label')
            logit = model(is_training, self.dataset.num_class).inference(image)

            cost = losses.LossFunction(n_class=self.dataset.num_class, loss_fn=self.loss_fn).layer_op(logit, tf.argmax(label, 3))
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            accr = [fn(logit, label) for fn in self.metric_fn][0]

            # # Add histograms for trainable variables.
            # for var in tf.trainable_variables():
            #     summaries.append(tf.summary.histogram(var.op.name, var))

            if mode == 'debug':
                if os.path.exists(self.subject_title):
                    shutil.rmtree(self.subject_title)

            tensorboard_utils.save_image_summary(image, 'image')
            tensorboard_utils.save_image_summary(tf.expand_dims(tf.argmax(label, -1), -1), 'label')
            tensorboard_utils.save_image_summary(tf.expand_dims(tf.argmax(logit, -1), -1), 'logit')

            tensorboard_utils.save_scalar_summary(scope, cost, 'cost')
            tensorboard_utils.save_scalar_summary(scope, accr, 'accr')

            tf.add_to_collection('metric_ops', accr)
            tf.add_to_collection('metric_ops', cost)

            self.input_tensor_name = image.name.split(":")[0]
            self.output_tensor_name = logit.name.split(":")[0]
            print(self.input_tensor_name, self.output_tensor_name)

            return training_loops(scope, self.dataset, image, label, self.subject_title, epochs, train_op)

    def eval(self, subject_title, image, input_node_name, output_node_name):
        with tf.Session() as sess:
            # load model with saved_model API from args.model_dir
            tf.saved_model.loader.load(sess, ['serve'], os.path.join(subject_title, 'model'))

            # default parameters to the graph
            init_graph = sess.graph
            input_node = init_graph.get_tensor_by_name(input_node_name + ':0')
            output_node = init_graph.get_tensor_by_name(output_node_name + ':0')
            is_training = init_graph.get_tensor_by_name('model/is_training:0')

            input_shape = input_node.get_shape().as_list()

            patch_nums = np.ceil(np.array(image.shape[1:]) / np.array(input_shape[1:]))

            coord_x = np.linspace(0, image.shape[1] - input_shape[1], patch_nums[0], dtype=np.uint32)
            coord_y = np.linspace(0, image.shape[2] - input_shape[2], patch_nums[1], dtype=np.uint32)

            output_shape = list(image.shape)
            output_shape[-1] = output_node.get_shape().as_list()[-1]
            result = np.zeros(output_shape)
            for x in coord_x:
                for y in coord_y:
                    input_image = image[:, x:x + input_shape[1], y:y + input_shape[2], :]

                    session_result = sess.run(output_node, feed_dict={input_node: input_image,
                                                                      is_training: tf.Variable(False)})

                    result[:, x:x + input_shape[1], y:y + input_shape[2], :] += session_result.astype(image.dtype)

            return np.argmax(result, -1)

    def to_pb(self, subject_title, input_tensor_name, output_tensor_name):
        output_dir = os.path.join(subject_title)
        from ..utils.save_model import create_saved_model
        create_saved_model([os.path.join(subject_title, 'checkpoint')],
                           input_tensor_name,
                           output_tensor_name,
                           tag_constants=['serve'],
                           output_dir=output_dir)
