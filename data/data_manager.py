import ast
import configparser
import os
from glob import glob
from random import shuffle

import numpy as np
import tensorflow as tf

from . import readers
from ..utils.multi_gpu_utils import count_gpus

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import random_seed
from tensorflow.python.data.util import sparse


class DataManager(object):
    def __init__(self, data_list, data_ext):
        if not isinstance(data_list, tuple):
            data_list = tuple(data_list)
        self.data_list = data_list
        self.data_ext = data_ext

        if data_ext == 'tfrecords' or data_ext == 'tfrecord':
            self.dataset = tf.data.TFRecordDataset(self.data_list)
        elif data_ext in ['jpg', 'jpeg', 'png']:
            self.dataset = tf.data.Dataset.from_tensor_slices(self.data_list)
        else:
            raise ValueError("Invalid data extension")

    def reader_fn(self, reader_fn, output_type):
        if not isinstance(reader_fn, list):
            reader_fn = list(reader_fn)
        if not isinstance(output_type, list):
            output_type = list(output_type)
        return ImageReader(self.dataset, reader_fn, output_type, self.data_list, self.data_ext)

    def generator(self, batch_size, shuffle=True):
        self.dataset = self.dataset.repeat()
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=(int(len(self.data_list) * 0.4) + 3 * batch_size))
        self.dataset = self.dataset.batch(batch_size)

        # iterator
        return self.dataset.make_initializable_iterator()


class ImageReader(DataManager):
    def __init__(self, dataset, reader_fn, output_type, data_list, data_ext):
        super(ImageReader, self).__init__(data_list, data_ext)
        self._dataset = dataset
        self.reader_fn = reader_fn
        self.output_type = output_type
        self.dataset = dataset.map(lambda image_list, label_list: tuple(tf.py_func(
            self._data_reader, [image_list, label_list], self.output_type)))

    def _data_reader(self, image_path, label_path):
        image = self.reader_fn[0](image_path, self.data_ext).astype(np.float32)
        label = self.reader_fn[1](label_path, self.data_ext).astype(np.uint8)

        rand_x = np.random.randint(0, image.shape[0] - 512)
        rand_y = np.random.randint(0, image.shape[1] - 512)
        image = image[rand_x: rand_x + 512, rand_y: rand_y + 512]
        label = label[rand_x: rand_x + 512, rand_y: rand_y + 512].reshape(512, 512, 1)
        image = image.reshape(512, 512, 3)
        label = label.reshape(512, 512, 1)
        return image, label

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types


#################################################################################################


def check_call_config(work_dir, project_title):
    config_path = os.path.join(work_dir, project_title, 'configuration.txt')

    config = configparser.ConfigParser()
    config.read_file(open(config_path))
    return config


def get_coords(config, input_shape, inbound_shape, grid_n):
    data_shape = tuple(ast.literal_eval(config.get('data attributes', "image_shape")))
    if isinstance(input_shape, int):
        input_shape = [input_shape] * len(data_shape)
    if isinstance(inbound_shape, int):
        inbound_shape = [inbound_shape] * len(data_shape)

    data_coords = []
    for i in range(len(input_shape) - 1):
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


def get_data_reader(data_ext, filenames, input_shape, label_shape, compression):
    data_reader = {'hdf5': readers.get_hdf5,
                   'tfrecords': readers.get_tfrecords}
    return data_reader[data_ext](filenames, input_shape, label_shape, compression)


def produce_dataset(num_device, data_ext, batch_size, batches_per_epoch, data_coords,
                    filenames, input_shape, compression):
    dataset = get_data_reader(data_ext, filenames, input_shape, compression)
    coords = tf.data.Dataset.from_tensor_slices(data_coords)
    dataset = tf.data.Dataset.zip((dataset, coords))
    # dataset = dataset.map(deformation_tf.deformation(aug_list))
    dataset = dataset.shuffle(buffer_size=(int(0.4 * batches_per_epoch) + batch_size * 3))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    image_stacked, label_stacked = iterator.get_next()

    with tf.device('/cpu:0'):
        image_batch = tf.unstack(image_stacked, num=batch_size, axis=0)
        label_batch = tf.unstack(label_stacked, num=batch_size, axis=0)
        # map_batch = tf.unstack(map_stacked, num=self.batch_size, axis=0)

        feature_shards = [[] for _ in range(num_device)]
        label_shards = [[] for _ in range(num_device)]
        # map_shards = [[] for _ in range(num_gpus)]

        for i in range(batch_size):
            idx = i % num_device
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
            # map_shards[idx].append(map_batch[i])

        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        # map_shards = [tf.parallel_stack(x) for x in map_shards]

        # features_shards = []
        # for i in range(len(feature_shards)):
        #     feature_dict = {}
        #     feature_dict['image'] = feature_shards[i]
        #     feature_dict['map'] = map_shards[i]
        #     features_shards.append(feature_dict)

        return feature_shards, label_shards, iterator


def get_datalist(data_dir, target_labels, valid_rate=0.1):
    data_list = glob(os.path.join(data_dir, '*.tfrecord*'))

    # target_list = [os.path.join(data_dir, target + '.tfrecords.gz') for target in target_labels]
    # target_list = []
    # for target in target_list:
    #     data_list.remove(target)
    # seed(0)
    shuffle(data_list)
    data_list = data_list[:len(data_list)]
    train_list = data_list[:-int(len(data_list) * valid_rate)]
    valid_list = data_list[-int(len(data_list) * valid_rate):]
    return train_list, valid_list


class DataManager2(object):
    def __init__(self, project_title, batch_size=4, input_shape=None, inbound_shape=None, grid_n=1, compression='GZIP',
                 valid_rate=0.1, num_gpus=None, data_ext='tfrecords', work_dir='c://workspace/', **model_params):
        self.project_title = project_title
        self.data_ext = data_ext

        self.num_gpus = count_gpus(num_gpus)
        self.num_device = 1 if self.num_gpus == 0 else self.num_gpus
        self.train_device = 'cpu' if self.num_gpus == 0 else 'gpu'

        self.config = check_call_config(work_dir, self.project_title)
        self.data_dir = self.config.get('data directory', 'data_dir_%s' % self.data_ext)
        self.id_list = ast.literal_eval(self.config.get('data attributes', 'data_list'))

        # target_labels = model_params['target_labels']
        self.train_list, self.valid_list = get_datalist(self.data_dir, valid_rate)
        # print("Target List :", self.target_list)

        self.batch_size = batch_size
        self.batches_per_epoch = len(self.train_list) // self.batch_size

        self.num_class = self.config.getint('data attributes', 'num_class')
        self.num_channel = self.config.getint('data attributes', "num_channel")
        self.data_shape = tuple(
            ast.literal_eval(self.config.get('data attributes', "image_shape")) + [self.num_channel])
        self.label_shape = tuple(ast.literal_eval(self.config.get('data attributes', "label_shape")))
        self.input_shape = self.data_shape if input_shape is None else input_shape
        self.inbound_shape = self.input_shape if inbound_shape is None else inbound_shape
        self.data_dim = len(self.input_shape)
        self.grid_n = grid_n
        self.data_coords = get_coords(self.config, self.input_shape, self.inbound_shape, self.grid_n)

        self.compression = compression
        print("Data Manager Is Ready")

    def get_data(self, handle):
        train_dataset = get_data_reader(self.data_ext, self.train_list, self.input_shape, self.label_shape,
                                        self.compression)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(buffer_size=(int(len(self.train_list) * 0.4) + 3 * self.batch_size))
        train_dataset = train_dataset.batch(self.batch_size)

        valid_dataset = get_data_reader(self.data_ext, self.valid_list, self.input_shape, self.label_shape,
                                        self.compression)
        valid_dataset = valid_dataset.repeat()
        valid_dataset = valid_dataset.batch(self.batch_size)

        # data_coords = tf.data.Dataset.from_tensor_slices(self.data_coords)
        # dataset = tf.data.Dataset.zip((dataset, data_coords))
        # dataset = dataset.map(deformation_tf.deformation(aug_list))

        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        image_stacked, label_stacked, datainfo_stacked = iterator.get_next()

        self.train_iterator = train_dataset.make_one_shot_iterator()
        self.valid_iterator = valid_dataset.make_initializable_iterator()

        with tf.device('/cpu:0'):
            def stack_multi_data(data_stacked, batch_size, num_device):
                data_batch = tf.unstack(data_stacked, num=batch_size, axis=0)
                data_shards = [[] for _ in range(num_device)]
                for i in range(batch_size):
                    idx = i % self.num_device
                    data_shards[idx].append(data_batch[i])
                data_shards = [tf.parallel_stack(x, name='dataset') for x in data_shards]
                return data_shards

            image_shards = stack_multi_data(image_stacked, batch_size=self.batch_size, num_device=self.num_device)
            label_shards = stack_multi_data(label_stacked, batch_size=self.batch_size, num_device=self.num_device)

            info_shards = {}
            for key, value in datainfo_stacked.items():
                info_shards[key] = stack_multi_data(value, batch_size=self.batch_size, num_device=self.num_device)
            return image_shards, label_shards, info_shards
