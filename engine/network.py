import tensorflow as tf


class Network(object):
    def __init__(self):
        self.__base_init(name=None)

    def __base_init(self, name):
        if not name:
            prefix = self.__class__.__name__.lower()
            name = tf.get_default_graph()

        self.name = name