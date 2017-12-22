import numpy as np
from JuneNet.process.data import hdf5_generator, tfrecords_generator


def generate_dataset(project_title, id_list, image_list, label_list, name_classes, weight_list=None, data_ext='tfrecords',
                     project_dir='c://workspace/', save_dir=r'c://workspace/data', **data_preprocess):
    data_generator = {'tfrecords': tfrecords_generator.get_tfrecords,
                       'hdf5': hdf5_generator}
    data_generator[data_ext](project_title, id_list, image_list, label_list, name_classes, weight_list, project_dir, save_dir, **data_preprocess)


