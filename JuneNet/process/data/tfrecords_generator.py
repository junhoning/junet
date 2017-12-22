import os
from glob import glob

import sys
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2

from JuneNet.process.data.readers import data_reader
from JuneNet.process.utils import progress_bar
from JuneNet.process.data import data_manager


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""
    if isinstance(text, str):
        return text
    elif isinstance(text, 'unicode'):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)


def data_generator(save_dir, id_list, image_list, label_list, name_classes, weight_list=None, **data_preprocess):
    print("Converting Start TFRecords")
    print("Number of Data :", len(image_list))
    if not len(image_list) == len(label_list) and len(image_dataset) == len(id_list):
        raise ValueError("Different Length of Image and Label")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_classes = len(name_classes)
    label_cnt = {'label_%d' % class_n: 0 for class_n in range(num_classes)}

    for data_n, filename in enumerate(id_list):
        image = data_reader(image_list[data_n])

        label = label_list[data_n]
        if isinstance(label, str):
            if label not in name_classes:
                label = data_reader(label)
            else:
                label = name_classes.index(label)

        if 'data_preprocess' in data_preprocess:
            image, label = data_preprocess['data_preprocess'](image, label)

        # filenames = filename
        # for i, side in enumerate(['_left', '_right']):
        # image = image_pair[i]
        # label = label_pair[i]
        # filename = filenames + side

        label_onehot = data_manager.onehot_encoding(label, num_classes)

        label_cnt["label_%d" % np.argmax(label_onehot)] += 1

        tfrecords_filename = os.path.join(save_dir, filename + '.tfrecords.gz')
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(path=tfrecords_filename, options=options)

        data_features = {
            'image': _bytes_feature(image.astype(np.uint8).tobytes()),
            # 'image_width': _int64_feature(image.shape[0]),
            # 'image_height': _int64_feature(image.shape[1]),
            'label': _bytes_feature(label_onehot.astype(np.uint8).tobytes()),
            # 'label_width': _int64_feature(label.shape[0]),
            'filename': _bytes_feature(str.encode(filename))
        }

        # data_features['image_shape'] = _int64_feature(list(image.shape))
        # data_features['label_shape'] = _int64_feature(list(label.shape))

        data_dimension = len(image.shape) - 1

        if weight_list is not None:
            binary_weight = (weight_list[data_n] * 255).astype(np.uint8).tobytes()
            data_features['weight_map'] = _bytes_feature(binary_weight)

        string_set = tf.train.Example(features=tf.train.Features(feature=data_features))

        writer.write(string_set.SerializeToString())
        writer.close()

        progress_bar(len(id_list), data_n, "Generating : %s, Label Count: %s" % (filename, str(label_cnt)))

    configuration = {}
    configuration['image_shape'] = list(image.shape)
    configuration['label_shape'] = list(label_onehot.shape) if len(label_onehot.shape) > 2 else [label_onehot.shape[-1]]
    configuration['num_class'] = num_classes
    configuration['num_channel'] = 3 if image.shape[-1] > 3 else 1
    configuration['data_dimension'] = data_dimension
    configuration['data_list'] = id_list
    configuration['name_classes'] = name_classes
    configuration['label_count'] = label_cnt

    print("\nConverting Finished.")
    sys.stdout.flush()

    return configuration


def write_configuration(job_dir, data_dir, configuration):
    save_path = os.path.join(job_dir, 'configuration.txt')
    save_dirs = {'log_path': 'log/',
                 'ckpt_dir': 'save_point/',
                 'img_result_dir': 'image/',
                 'tboard_dir': 'tboard/',
                 'result_dir': 'result/'}

    data_list = configuration.pop('data_list')

    with open(save_path, 'w') as save_file:
        # Writing Train Setting
        save_file.write("[data directory]")
        save_file.write("\n%s = %s" % ('data_dir_tfrecords', data_dir))

        save_file.write("\n\n[data attributes]")
        for key, value in configuration.items():
            save_file.write("\n%s = %s" % (key, value))
        save_file.write("\ndata_list = %s" % str(data_list))


def get_tfrecords(project_title, id_list, image_list, label_list, name_classes, weight_list=None,
                  project_dir='c:/workspace/', save_dir=r'c:/workspace/data', **data_preprocess):
    project_dir = project_dir + project_title
    num_classes = len(name_classes)
    print('project title', project_title)
    print('project dir', project_dir)
    config_path = os.path.join(project_dir, 'configuration.txt')
    if not os.path.exists(config_path):
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        print("Start Generating DataSet")
        save_dir = os.path.join(save_dir, 'tfrecords/', project_title)
        print("Save at :", save_dir)
        configuration = data_generator(save_dir, id_list, image_list, label_list, name_classes, weight_list, **data_preprocess)
        write_configuration(project_dir, save_dir, configuration)
    else:
        print("Dataset seems already exists")
    return config_path


if __name__ == "__main__":
    project_title = 'temp'
    org_data_path = r'c:/workspace/data/SegEngine/'
    image_dataset = glob(org_data_path + "*_MRI.nii")[:10]
    label_dataset = glob(org_data_path + "*_Label.nii")[:10]
    get_tfrecords(project_title, image_dataset, label_dataset)
