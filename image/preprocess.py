import SimpleITK as sitk
from skimage import measure
from scipy import ndimage
import h5py
import nibabel as nib
import numpy as np
import cv2


def get_array_nii(data_path):
    return np.array(nib.load(data_path).get_data())


def get_array_h5(h5_file):
    img_set = h5py.File(h5_file)
    label = img_set['labels']
    image = img_set['images']
    return image, label


def image_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Standardization Images
# https://www.thoughtco.com/z-score-formula-3126281
def z_score(x):
    # eps = 1e-10
    # x = np.clip(x, a_min=eps, a_max=None)
    return np.divide((x - np.mean(x)), np.std(x))


# Get the coordinates of the labels
def get_bbox(label):
    # To label each different area, you can use 'measure.label'
    '''
    :param label: array. Array of Label such as '.npy'
    :param label_n: int. Class Number of Label you want to get.
    :return: Each Coordinates of the label_n
    '''
    # label_classes = measure.label(label, neighbors=8, connectivity=2)
    region = measure.regionprops(label.astype(int)+1)  # add 1 to include background and get regions
    if len(region[-1].bbox) > 0:
        min_row, min_col, max_row, max_col = region[-1].bbox  # get bboxes
    else:
        min_row, min_col, max_row, max_col = 0, 0, label.shape[0], label.shape[1]

    return [min_row, min_col, max_row, max_col]


def bbox_center(label):
    min_row, min_col, max_row, max_col = get_bbox(label)
    cent_row = (max_row - min_row) // 2 + min_row
    cent_col = (max_col - min_col) // 2 + min_col
    return [cent_row, cent_col]


# Remove background
def get_background_coord(image, transpose=True):
    # Search Middle of MRI
    if transpose:
        image = image.transpose(2,0,1)
        cent_img_n = image.shape[2] // 2
    else:
        cent_img_n = image.shape[0] // 2
    image = image[cent_img_n]

    classes = measure.label(image)
    labels = np.where(classes == 0, 0, 1)
    new = measure.regionprops(labels)
    min_row, min_col, max_row, max_col = new[0].bbox
    return min_row, max_row, min_col, max_col

