import os
from glob import glob

import tensorflow as tf
import h5py as h5
import numpy as np
import cv2
import nibabel as nib


def nii_reader(data_path):
    data = np.array(nib.load(data_path).get_data())
    return data


def png_reader(data_path):
    data = cv2.imread(data_path, 0)
    return data


def data_reader(data_path):
    data_ext = data_path.split('.')[-1]
    if data_ext == 'nii':
        return nii_reader(data_path)
    elif data_ext == 'png' or data_ext == 'jpg' or data_ext == 'jpeg':
        return png_reader(data_path)


def numpy_reader(filenames):
    image = np.load(filenames)
    return image


def hdf5_reader(hdf5_data, filenames):
    hdf5_data = hdf5_data[filenames]
    image = hdf5_data['image']
    label = hdf5_data['label']
    return image, label


def get_hdf5(data_dir):
    filename = glob(os.path.join(data_dir, '*.hdf5'))
    if len(filename) > 1:
        raise ValueError("There are more than one hdf5 files. Should be single data")
    else:
        hdf5_data = h5.File(filename[0])
        data_ids = list(hdf5_data.keys())
        dataset = tf.data.Dataset.from_tensor_slices(data_ids)
        dataset = dataset.map(
            lambda hdf5_data, filenames: tf.py_func(hdf5_reader, [hdf5_data, filenames], [tf.float32, tf.uint8]))
        print("Number of Datasets :", len(data_ids))
        return dataset


def get_tfrecords(filenames, data_shape, label_shape, compression='GZIP'):
    def record_parser(tfrecords):
        keys_to_features = {}
        keys_to_features['image'] = tf.FixedLenFeature([], tf.string, default_value="")
        keys_to_features['label'] = tf.FixedLenFeature([], tf.string, default_value="")
        keys_to_features['filename'] = tf.FixedLenFeature([], tf.string, default_value="")

        # length_list = ['width', 'height', 'depth']
        # for length, _ in zip(length_list, data_shape[:-1]):
        #     keys_to_features['image_' + length] = tf.FixedLenFeature([], tf.int64)
        # for length, _ in zip(length_list, label_shape[:-1]):
        #     keys_to_features['label_' + length] = tf.FixedLenFeature([], tf.int64)

        features = tf.parse_single_example(tfrecords, keys_to_features)

        # image_shape = [features['image_' + length] for length, _ in zip(length_list, data_shape[:-1])]
        # image_shape = tf.stack([tf.cast(length, tf.int64) for length in image_shape] + [data_shape[-1]])
        # output_shape = [features['label_' + length] for length, _ in zip(length_list, label_shape[:-1])]
        # output_shape = tf.stack([tf.cast(length, tf.int64) for length in output_shape] + [label_shape[-1]])

        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, data_shape)

        label = tf.decode_raw(features['label'], tf.uint8)
        label = tf.reshape(label, list(label_shape))

        data_info = {}
        data_info['filename'] = features['filename']
        return image, label, data_info

    dataset = tf.data.TFRecordDataset(filenames, compression_type=compression)
    dataset = dataset.map(record_parser)
    return dataset
