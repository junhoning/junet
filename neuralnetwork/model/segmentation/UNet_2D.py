import tensorflow as tf

from ...builder import acts, layers

act_fn = acts.pRelu
initializer = tf.contrib.layers.xavier_initializer()


class BuildModel(object):
    def __init__(self, is_training, num_class, **model_params):
        self.kernel_num = model_params.pop('kernel_num', 64)
        self.block_n = model_params.pop('block_n', 4)
        self.conv_kernel = 1
        self.output_dim = num_class
        self.padding = 'SAME'
        self.record_tboard = model_params.pop('record_tboard', True)

        self.model_log = model_params.pop('model_log', False)
        self.is_training = is_training
        print("2D U-Net is Loading")

    def conv2d_layer(self, input, out_channel, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv_layer'):
        with tf.variable_scope(name):
            in_channel = input.get_shape()[-1]
            w = tf.get_variable('w', [k_h, k_w, in_channel, out_channel], initializer=initializer)
            conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding=padding, name='conv')
            bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)

            return conv_b

    def conv2d_block(self, input, out_channel, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', activation_fn=act_fn, name='conv_block'):
        with tf.variable_scope(name):
            conv = self.conv2d_layer(input, out_channel, k_h, k_w, d_h, d_w, padding=padding)
            bn = layers.batch_norm(conv, train_phase=self.is_training, name='batch_norm')
            output = activation_fn(bn, 'act')
            return output

    def conv2d_trans(self, input, k_h=2, k_w=2, d_h=2, d_w=2, activation_fn=act_fn, name="conv2d_trans"):
        with tf.variable_scope(name):
            deconv_shape = tf.stack(
                [tf.shape(input)[0], input.get_shape()[1] * 2, input.get_shape()[2] * 2, input.get_shape()[-1]])
            w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], input.get_shape()[-1]], initializer=initializer)
            conv = tf.nn.conv2d_transpose(input, w, output_shape=deconv_shape, strides=[1, d_h, d_w, 1], padding="SAME")
            bias = tf.get_variable('biases', [input.get_shape()[-1]], initializer=tf.constant_initializer(0))
            conv_b = tf.nn.bias_add(conv, bias)

            bn = layers.batch_norm(conv_b, train_phase=self.is_training, name='batch_norm')
            act = activation_fn(bn, 'act')

            return act

    def max_pool(self, input, k_h=2, k_w=2, d_h=2, d_w=2, padding='SAME', name='pool_layer'):
        with tf.variable_scope(name):
            kernel_size = [1, k_h, k_w, 1]
            strides = [1, d_h, d_w, 1]
            pool = tf.nn.max_pool(input, ksize=kernel_size, strides=strides, padding=padding, name='pool')
            return pool

    def crop_concat(self, input_1, input_2):
        x1_shape = input_1.get_shape().as_list()
        x2_shape = input_2.get_shape().as_list()

        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(input_1, offsets, size)

        return tf.concat([x1_crop, input_2], -1)

    def _print_name(self, layer):
        if self.model_log:
            print(layer.name.split('/')[1], '\t', layer.get_shape())

    def inference(self, net):
        # Down Sampling
        skip_list = []
        for i in range(1, self.block_n+1):
            net = self.conv2d_block(net, self.kernel_num * self.conv_kernel, padding=self.padding, name='Conv2d_%d_1' % i)
            self._print_name(net)
            self.conv_kernel *= 2
            net = self.conv2d_block(net, self.kernel_num * self.conv_kernel, padding=self.padding, name='Conv2d_%d_2' % i)
            self._print_name(net)
            skip_list.append(net)
            net = self.max_pool(net, padding=self.padding, name='Pooling_%d' % i)
            self._print_name(net)

        # Bridge
        net = self.conv2d_block(net, self.kernel_num * self.conv_kernel, padding='SAME', name='bridge_1')
        self._print_name(net)
        self.conv_kernel *= 2
        net = self.conv2d_block(net, self.kernel_num * self.conv_kernel, padding='SAME', name='bridge_2')
        self._print_name(net)

        # Up Sampling
        for i in range(1, self.block_n+1):
            self.conv_kernel /= 2
            net = self.conv2d_trans(net, name='Conv3d_trans_%d' % (self.block_n+1 - i))
            self._print_name(net)
            net = self.conv2d_block(self.crop_concat(skip_list[-i], net), self.kernel_num * self.conv_kernel, padding='SAME', name='UpConv3d_%d_1' % (self.block_n+1 - i))
            self._print_name(net)
            net = self.conv2d_block(net, self.kernel_num * self.conv_kernel, padding='SAME', name='UpConv3d_%d_2' % (self.block_n+1 - i))
            self._print_name(net)

        # Output
        output = self.conv2d_layer(net, self.output_dim, k_h=1, k_w=1, d_h=1, d_w=1, padding='SAME', name='output')
        self._print_name(net)

        print("Model is successfully loaded")
        return output

