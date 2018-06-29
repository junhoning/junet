import tensorflow as tf

from JuNet.neuralnetwork.builder import acts, layers


initializer = tf.contrib.layers.xavier_initializer()
act_fn = acts.pRelu


class build_model(object):
    def __init__(self, is_training, **model_params):
        self.is_training = is_training
        self.keep_prob = model_params.pop('keep_prob', 0.7)
        self.num_class = model_params['num_class']

    def inference(self, layer):
        layer = layers.conv2d(layer, 64, is_training=self.is_training, name='conv1')
        layer = layers.conv2d(layer, 64, is_training=self.is_training, name='conv2')
        layer = layers.max_pool(layer, name='max_pool1')

        layer = layers.conv2d(layer, 128, is_training=self.is_training, name='conv3')
        layer = layers.conv2d(layer, 128, is_training=self.is_training, name='conv4')
        layer = layers.max_pool(layer, name='max_pool2')

        layer = layers.conv2d(layer, 256, is_training=self.is_training, name='conv5')
        layer = layers.conv2d(layer, 256, is_training=self.is_training, name='conv6')
        layer = layers.conv2d(layer, 256, is_training=self.is_training, name='conv7')
        layer = layers.max_pool(layer, name='max_pool3')

        layer = layers.conv2d(layer, 512, is_training=self.is_training, name='conv8')
        layer = layers.conv2d(layer, 512, is_training=self.is_training, name='conv9')
        layer = layers.conv2d(layer, 512, is_training=self.is_training, name='conv10')
        layer = layers.max_pool(layer, name='max_pool4')

        layer = layers.conv2d(layer, 512, is_training=self.is_training, name='conv11')
        layer = layers.conv2d(layer, 512, is_training=self.is_training, name='conv12')
        layer = layers.conv2d(layer, 512, is_training=self.is_training, name='conv13')
        net = layers.max_pool(layer, name='max_pool5')

        # Fully Connected
        net = layers.conv2d(net, 4096, k_size=(7, 7), name='fc1')
        net = layers.drop_out(net, keep_prob=self.keep_prob, is_training=self.is_training, name='dropout_1')
        net = layers.conv2d(net, 4096, k_size=(1, 1), name='fc2')

        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

        net = layers.drop_out(net, keep_prob=self.keep_prob, is_training=self.is_training, name='dropout_2')
        net = layers.conv2d(net, self.num_class, (1, 1), activation_fn=None, normalization=None, name='conv14')
        net = tf.squeeze(net, [1, 2], name='output')
        return net
