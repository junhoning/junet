import numpy as np
import h5py

import os
from glob import glob
import sys
from datetime import datetime

import nibabel as nib

ORG_SIZE = 256
NUM_CLASS = 9
TEST_RATE = 0.1
VALID_RATE = 0.1

dir_path = 'C://workspace/data/SegEngine/'
file_name = 'brain_seg_data_test.hdf5'
hdf5_path = os.path.join(dir_path, file_name)

data_path = 'C://workspace/data/SegEngine/dataset/'
image_list = glob('C://workspace/data/SegEngine/dataset/*_MRI.nii')[:60]
label_list = glob('C://workspace/data/SegEngine/dataset/*_Label.nii')[:60]

id_list = [file_id.split('\\')[-1].split('_')[0] for file_id in image_list]


# Run "onehot_label" to make label to onehot_encode
def _dense_onehot_(labels_dense, num_class):
    # Convert class labels from scalars to one-hot vectors.
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_class
    labels_one_hot = np.zeros((num_labels, num_class))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def onehot_label(label, num_class):
    # It is important to convert to type 'int'
    label = np.reshape(label, (label.shape[0], label.shape[1], label.shape[2])).astype(int)
    label_onehot = np.zeros((label.shape[0], label.shape[1], label.shape[2], num_class))
    for row in range(label.shape[1]):
        for col in range(label.shape[2]):
            single = label[:, row, col]
            one_hot = _dense_onehot_(single, num_class)
            label_onehot[:, row, col, :] = one_hot
    return label_onehot


def data_generator(hdf5_path):
    with h5py.File(hdf5_path, mode='w') as hdf5_file:
        print("Converting Start", datetime.now())
        # Converting Each Image
        for i, data_id in enumerate(id_list):
            if i % 10 == 0 and i > 1:
                print('Train data: {}/{}'.format(i, len(image_list)))
                sys.stdout.flush()

            image_path = data_path + data_id + '_MRI.nii'
            label_path = data_path + data_id + '_Label.nii'

            image = np.array(nib.load(image_path).get_data())
            label = np.array(nib.load(label_path).get_data())

            subgroup_dataid = hdf5_file.create_group(str(data_id))

            image_data = subgroup_dataid.create_dataset('image', image.shape, dtype=image.dtype, compression='gzip', compression_opts=4)
            label_data = subgroup_dataid.create_dataset('label', label.shape, dtype=label.dtype, compression='gzip', compression_opts=4)

            hdf5_file[image_data.name][...] = image
            hdf5_file[label_data.name][...] = label

        # Save the Mean and close the hdf5 file
        print("Finished Converting")


def main(hdf5_path):
    data_generator(hdf5_path)


if __name__ == '__main__':
    main(hdf5_path)