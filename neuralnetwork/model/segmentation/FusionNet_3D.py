import tensorflow as tf

from JuNet.neuralnetwork.builder import acts
from JuNet.neuralnetwork.builder.layers import batch_norm

act_fn_1 = acts.pRelu
act_fn_2 = acts.pRelu
initializer = tf.contrib.layers.xavier_initializer()


class build_model(object):
    def __init__(self, train_setting):
        self.kernel_num = 16
        self.growth_rate = 1
        self.n_block = 4
        self.padding = 'SAME'

        self.remove_label = train_setting['remove_label']
        self.output_dim = 9 - len(self.remove_label)
        self.is_training = True
        self.record_tboard = train_setting['record_tboard']
        self.log = False

        self.train_setting = train_setting
        if train_setting['model_setting']['setting_on']:
            self.kernel_num = train_setting['model_setting']['kernel_num']
            self.n_block = train_setting['model_setting']['n_block']


        self.skip_list = []
        print("Model is Loading...")

    def bottleneck(self, input_, output_dim, k_h=1, k_w=1, k_d=1, d_h=1, d_w=1, d_d=1, name='bottleneck'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding='SAME', name='conv')
            biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, biases)
            if self.record_tboard:
                tf.summary.histogram("weight", w)
            return conv_b

    def conv_block(self, input_, output_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, padding='SAME', activation_fn=tf.nn.relu, name='conv3d'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding=padding, name='conv')
            bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)

            output = batch_norm(conv_b, train_phase=self.is_training, name='batch_norm')
            if self.record_tboard:
                tf.summary.histogram("weight", w)

            if activation_fn is not None:
                return activation_fn(output, 'act')
            else:
                return output

    def transition_layer(self, input_, output_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, activation_fn=act_fn_1, name='transition_layer'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding='SAME', name='conv')
            bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)
            bn = batch_norm(conv_b, train_phase=self.is_training, name='batch_norm')
            act = activation_fn(bn)
            return act

    def dense_block(self, conv, output_dim, dense_repeat_num=3, activation_fn=act_fn_1, padding='SAME', name='dense_block'):
        with tf.variable_scope(name):
            nodes = [conv]
            for i in range(dense_repeat_num):
                if i == dense_repeat_num - 1:
                    activation_fn = None
                conv_name = "conv_%d" % (i + 2)
                conv = self.conv_block(tf.concat(nodes, axis=4), output_dim, activation_fn=activation_fn, name=conv_name)
                nodes.append(conv)
            conv = self.conv_block(conv, output_dim, d_h=1, d_w=1, d_d=1, activation_fn=activation_fn, padding=padding, name='conv_%d' % (dense_repeat_num + 2))
            return conv

    def residual_block(self, input_, output_dim, activation_fn, padding='SAME', name="residual_block"):
        with tf.variable_scope(name):
            conv1 = self.conv_block(input_, output_dim, activation_fn=activation_fn, name="conv1")
            with tf.variable_scope('conv_block'):
                conv2 = self.conv_block(conv1, output_dim, activation_fn=activation_fn, name='conv_2')
                conv3 = self.conv_block(conv2, output_dim, activation_fn=activation_fn, name='conv_3')
                conv4 = self.conv_block(conv3, output_dim, activation_fn=None, name='conv_4')
                skip_connect = tf.add(conv1, conv4, name='skip_connection')
            conv5 = self.conv_block(skip_connect, output_dim, activation_fn=activation_fn, padding=padding, name="conv_5")
            return conv5

    def conv3d_layer(self, input_, output_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv3d'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding='SAME', name='conv')
            bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)
            return conv_b

    def conv3d_trans(self, input_, output_shape, k_h=5, k_w=5, k_d=5, d_h=2, d_w=2, d_d=2, activation_fn=tf.nn.relu, name="conv3d_Trans"):
        with tf.variable_scope(name):
            output_shape[0] = tf.shape(input_)[0]
            w = tf.get_variable('w', [k_h, k_w, k_d, output_shape[-1], input_.get_shape()[-1]], initializer=initializer)
            conv = tf.nn.conv3d_transpose(input_, w, output_shape=tf.stack(output_shape), strides=[1, d_h, d_w, d_d, 1], padding="SAME")
            bias = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)

            bn = batch_norm(conv_b, train_phase=self.is_training, name='batch_norm')
            act = activation_fn(bn, 'act')
            if self.record_tboard:
                tf.summary.histogram("weight_trans", w)
            return act

    def _print_name(self, layer):
        if self.log:
            print(layer.name.split('/')[1], '\t', layer.get_shape())

    def encoder(self, net):
        for i in range(1, self.n_block+1):
            net = self.dense_block(net, self.kernel_num * self.growth_rate, activation_fn=act_fn_1, padding=self.padding, name="down_%d" % i)
            self._print_name(net)
            self.skip_list.append(net)
            net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="VALID", name='pool_%d' % i)
            self._print_name(net)
            self.growth_rate *= 2
        return net

    def decoder(self, net):
        for i in range(self.n_block, 0, -1):
            self.growth_rate /= 2
            net = self.conv3d_trans(net, self.skip_list[i-1].get_shape().as_list(), activation_fn=act_fn_2, name="unpool_%d" % i)
            self._print_name(net)
            net = tf.concat([net, self.skip_list[i-1]], axis=-1, name='skip_up_%d' % i)
            net = self.dense_block(net, self.kernel_num * self.growth_rate, activation_fn=act_fn_2, padding=self.padding, name="up_%d" % i)
            self._print_name(net)
        return net

    def inference(self, image):
        encodes = self.encoder(image)
        bridge = self.dense_block(encodes, self.kernel_num * self.growth_rate, activation_fn=act_fn_1, name="bridge")
        self._print_name(bridge)
        decodes = self.decoder(bridge)
        logit = self.bottleneck(decodes, self.output_dim, name='bottleneck')
        self._print_name(logit)
        print("Model Loaded Successfully")

        return logit
