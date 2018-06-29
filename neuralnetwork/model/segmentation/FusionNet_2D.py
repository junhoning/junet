import tensorflow as tf
import sys

from JuNet.neuralnetwork.builder import acts
from JuNet.neuralnetwork.builder.layers import conv2d, batch_norm

act_fn_1 = acts.pRelu
act_fn_2 = acts.pRelu
initializer = tf.contrib.layers.xavier_initializer()


class build_model(object):
    def __init__(self, train_setting, is_training):
        self.growth_rate = 1
        self.padding = 'SAME'

        self.remove_label = train_setting['remove_label']  # train_setting.pop('remove_label', [])
        self.output_dim = 9 - len(self.remove_label)
        self.is_training = is_training
        self.record_tboard = train_setting['record_tboard']
        self.log = False

        self.train_setting = train_setting
        self.kernel_num = 64
        self.n_block = 4

        self.connected_cov = self.dense_block
        self.skip_list = []
        print("Model is Loading...")
        sys.stdout.flush()

    def bottleneck(self, input_, output_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='bottleneck'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME', name='conv')
            biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            conv_b = tf.nn.bias_add(conv, biases)
            if self.record_tboard:
                tf.summary.histogram("weight", w)
            return conv_b

    def conv_block(self, input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', activation_fn=act_fn_1, name='conv2d'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding, name='conv')
            bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)

            output = batch_norm(conv_b, train_phase=self.is_training, name='batch_norm')

            if activation_fn is not None:
                return activation_fn(output, 'act')
            else:
                return output

    def dense_block(self, conv, output_dim, dense_repeat_num=3, activation_fn=act_fn_1, padding='SAME', name='dense_block'):
        with tf.variable_scope(name):
            # conv = self.conv_block(conv, output_dim, activation_fn=activation_fn, name='conv_1')
            nodes = [conv]
            for i in range(dense_repeat_num):
                if i == (dense_repeat_num - 1):
                    activation_fn = None
                conv_name = "conv_%d" % (i + 2)
                conv = self.conv_block(tf.concat(nodes, axis=3), output_dim, activation_fn=activation_fn, name=conv_name)
                nodes.append(conv)
            conv = self.conv_block(conv, output_dim, activation_fn=activation_fn, padding=padding, name='conv_%d' % (dense_repeat_num + 2))
            return conv

    def residual_block(self, input_, output_dim, activation_fn, padding='SAME', name="residual_block"):
        with tf.variable_scope(name):
            conv1 = self.conv_block(input_, output_dim, activation_fn=activation_fn, name="conv1")
            with tf.variable_scope('conv_block'):
                conv2 = self.conv_block(conv1, output_dim, activation_fn=activation_fn, name='conv_2')
                conv3 = self.conv_block(conv2, output_dim, activation_fn=activation_fn, name='conv_3')
                conv4 = self.conv_block(conv3, output_dim, activation_fn=None, name='conv_4')
                skip_connect = tf.add(conv1, conv4, name='skip_connection')
            conv5 = self.conv_block(skip_connect, output_dim, activation_fn=None, padding=padding, name="conv_5")
            return conv5

    def conv2d(self, input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv2d'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding, name='conv')
            bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            conv_b = tf.nn.bias_add(conv, bias)
            return conv_b

    def conv2d_trans(self, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, activation_fn=tf.nn.relu, name="conv3d_Trans"):
        with tf.variable_scope(name):
            output_shape[0] = tf.shape(input_)[0]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=initializer)
            conv = tf.nn.conv2d_transpose(input_, w, output_shape=tf.stack(output_shape), strides=[1, d_h, d_w, 1], padding="SAME")
            bias = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            conv_b = tf.nn.bias_add(conv, bias)

            bn = batch_norm(conv_b, train_phase=self.is_training, name='batch_norm')
            act = activation_fn(bn, 'act')
            if self.record_tboard:
                tf.summary.histogram("weight_trans", w)
            return act

    def _print_name(self, layer):
        if self.log:
            print(layer.name.split('/')[1], '\t', layer.get_shape())

    def encoder(self, input_):
        self.down1 = self.connected_cov(input_, self.kernel_num, activation_fn=act_fn_1, name="down1")
        pool1 = tf.nn.max_pool(self.down1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='pool1')

        self.down2 = self.connected_cov(pool1, self.kernel_num * 2, activation_fn=act_fn_1, name="down2")
        pool2 = tf.nn.max_pool(self.down2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='pool2')

        self.down3 = self.connected_cov(pool2, self.kernel_num * 4, activation_fn=act_fn_1, name="down3")
        pool3 = tf.nn.max_pool(self.down3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='pool3')

        self.down4 = self.connected_cov(pool3, self.kernel_num * 8, activation_fn=act_fn_1, name="down4")
        pool4 = tf.nn.max_pool(self.down4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='pool4')

        return pool4

    def decoder(self, input_):
        conv_trans4 = self.conv2d_trans(input_, self.down4.get_shape().as_list(), activation_fn=act_fn_2, name="unpool4")
        res4 = tf.divide(tf.add(conv_trans4, self.down4, name='skip_up_4'), 2)
        up4 = self.connected_cov(res4, self.kernel_num * 8, activation_fn=act_fn_2, name="up4")

        conv_trans3 = self.conv2d_trans(up4, self.down3.get_shape().as_list(), activation_fn=act_fn_2, name="unpool3")
        res3 = tf.divide(tf.add(conv_trans3, self.down3, name='skip_up_3'), 2)
        up3 = self.connected_cov(res3, self.kernel_num * 4, activation_fn=act_fn_2, name="up3")

        conv_trans2 = self.conv2d_trans(up3, self.down2.get_shape().as_list(), activation_fn=act_fn_2, name="unpool2")
        res2 = tf.divide(tf.add(conv_trans2, self.down2, name='skip_up_2'), 2)
        up2 = self.connected_cov(res2, self.kernel_num * 2, activation_fn=act_fn_2, name="up2")

        conv_trans1 = self.conv2d_trans(up2, self.down1.get_shape().as_list(), activation_fn=act_fn_2, name="unpool1")
        res1 = tf.divide(tf.add(conv_trans1, self.down1, name='skip_up_1'), 2)
        up1 = self.connected_cov(res1, self.kernel_num, activation_fn=act_fn_2, name="up1")

        return up1

    def inference(self, image):
        encodes = self.encoder(image)
        bridge = self.connected_cov(encodes, self.kernel_num * 16, activation_fn=act_fn_1, name="bridge")
        decodes = self.decoder(bridge)
        logit = self.bottleneck(decodes, self.output_dim, name='bottleneck')
        self._print_name(logit)
        print("Model Loaded Successfully")

        return logit