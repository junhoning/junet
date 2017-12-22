import ast
import configparser
import os

import numpy as np
import tensorflow as tf

from JuneNet.process.data import readers, tfrecords_generator
from JuneNet.process.utils import count_gpus


def check_call_config(work_dir, project_title):
    config_path = os.path.join(work_dir, project_title, 'configuration.txt')

    config = configparser.ConfigParser()
    config.read_file(open(config_path))
    return config


def get_coords(config, input_shape, inbound_shape, grid_n):
    data_shape = tuple(ast.literal_eval(config.get('data attributes', "data_shape")))
    if isinstance(input_shape, int):
        input_shape = [input_shape] * len(data_shape)
    if isinstance(inbound_shape, int):
        inbound_shape = [inbound_shape] * len(data_shape)

    data_coords = []
    for i in range(len(input_shape)):
        pad_size = (data_shape[i] - inbound_shape[i]) // 2
        if grid_n == 1:
            coords = np.array(data_shape[i] // 2)
        else:
            coords = np.linspace(pad_size + (input_shape[i] // 2),
                                 data_shape[i] - pad_size - (input_shape[i] // 2),
                                 grid_n).astype(int)
        data_coords.append(coords)

    if len(data_shape) - len(input_shape):
        data_coords = list(range(data_shape[0])) + data_coords

    return data_coords


def get_data_reader(data_ext, data_dir, input_shape):
    data_reader = {'hdf5': readers.get_hdf5,
                  'tfrecords': readers.get_tfrecords}
    return data_reader[data_ext](data_dir, input_shape)


def produce_dataset(num_gpus, data_dir, data_ext, batch_size, batches_per_epoch, data_coords, input_shape):
    dataset = get_data_reader(data_ext, data_dir, input_shape)
    coords = tf.data.Dataset.from_tensor_slices(data_coords)
    # dataset = tf.data.Dataset.zip((dataset, data_coords))
    # dataset = dataset.map(deformation_tf.deformation(aug_list))
    dataset = dataset.shuffle(buffer_size=(int(0.4 * batches_per_epoch) + batch_size * 3))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    image_stacked, label_stacked = iterator.get_next()
#     return image_stacked, label_stacked
#
#
# def produce_dataset_stacked(num_gpus, data_dir, data_ext, batch_size, batches_per_epoch, data_coords, input_shape):
#     image_stacked, label_stacked = produce_dataset(data_dir, data_ext, batch_size, batches_per_epoch, data_coords, input_shape)

    with tf.device('/cpu:0'):
        image_batch = tf.unstack(image_stacked, num=batch_size, axis=0)
        label_batch = tf.unstack(label_stacked, num=batch_size, axis=0)
        # map_batch = tf.unstack(map_stacked, num=self.batch_size, axis=0)

        feature_shards = [[] for _ in range(num_gpus)]
        label_shards = [[] for _ in range(num_gpus)]
        # map_shards = [[] for _ in range(num_gpus)]

        for i in range(batch_size):
            idx = i % num_gpus
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
            # map_shards[idx].append(map_batch[i])

        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        # map_shards = [tf.parallel_stack(x) for x in map_shards]

        features_shards = []
        for i in range(len(feature_shards)):
            feature_dict = {}
            feature_dict['image'] = feature_shards[i]
            # feature_dict['map'] = map_shards[i]
            features_shards.append(feature_dict)

        return features_shards, label_shards


def generate_dataset(project_title, id_list, image_list, label_list, weight_list=None,
                     data_ext='tfrecords', project_dir='c://workspace/', save_dir=r'c://workspace/data'):
    from JuneNet.process.generator import hdf5_generator
    data_generator = {'tfrecords': tfrecords_generator.get_tfrecords,
                       'hdf5': hdf5_generator}
    data_generator[data_ext](project_title, id_list, image_list, label_list, weight_list, project_dir, save_dir)


def make_batch(num_gpus, data_dir, data_ext, batch_size, batches_per_epoch, data_coords, input_shape):
    if num_gpus > 0:
        features, label = produce_dataset_stacked(num_gpus, data_dir, data_ext,
                                                  batch_size, batches_per_epoch, data_coords, input_shape)
    else:
        features, label = produce_dataset(data_dir, data_ext,
                                          batch_size, batches_per_epoch, data_coords, input_shape)

    return {'features': features, 'label': label}


class DataManager(object):
    def __init__(self, project_title, batch_size=4, input_shape=None, inbound_shape=None, grid_n=1,
                 num_gpus=None, data_ext='tfrecords', work_dir='c://workspace/'):
        self.project_title = project_title
        self.data_ext = data_ext
        self.num_gpus = count_gpus(num_gpus)
        self.config = check_call_config(work_dir, self.project_title)
        self.data_dir = self.config.get('data directory', 'data_dir_%s' % self.data_ext)
        self.data_list = ast.literal_eval(self.config.get('data attributes', 'data_list'))

        self.batch_size = batch_size
        self.num_class = self.config.get('data attributes', 'num_class')
        self.batches_per_epoch = len(self.data_list) // self.batch_size

        self.data_shape = tuple(ast.literal_eval(self.config.get('data attributes', "data_shape")))
        self.input_shape = self.data_shape if input_shape is None else input_shape
        self.inbound_shape = self.input_shape if inbound_shape is None else inbound_shape
        self.data_dim = len(self.input_shape)
        self.grid_n = grid_n
        self.data_coords = get_coords(self.config, self.input_shape, self.inbound_shape, self.grid_n)

        self.images = make_batch(self.num_gpus, self.data_dir, self.data_ext, self.batch_size,
                                 self.batches_per_epoch, self.data_coords, self.input_shape)['features']
        self.labels = make_batch(self.num_gpus, self.data_dir, self.data_ext, self.batch_size,
                                 self.batches_per_epoch, self.data_coords, self.input_shape)['label']
        print("Data Manager Is Ready")
