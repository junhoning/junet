import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from random import randint


def _replace_nan(logit):
    # Replace 'NaN' to 0
    return tf.where(tf.is_nan(logit), tf.zeros_like(logit), logit)


def _cut_label_zero(label):
    label_shape = label.get_shape().as_list()
    dim = len(label_shape) - 1
    label_shape[-1] = label_shape[-1] - 1
    label_shape[0] = tf.shape(label)[0]
    result = tf.slice(label, [0]*dim + [1], tf.stack(label_shape))
    return result


def _pixel_wise_softmax(output_map, dim=3):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, dim, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1] * dim + [tf.shape(output_map)[dim]]))
    return tf.div(exponential_map, tensor_sum_exp)


def confusion_matrix(logit, y):
    logit = tf.one_hot(tf.argmax(logit, axis=3), depth=y.get_shape().as_list()[3])
    TP = tf.reduce_sum(tf.multiply(logit, y))
    FP = tf.reduce_sum(tf.multiply(logit, 1 - y))
    FN = tf.reduce_sum(tf.multiply(1 - logit, y))
    TN = tf.reduce_sum(tf.multiply(1 - logit, 1 - y))
    return TP, FP, FN, TN


def eval_dice_coef(logits, y, cut_zero=False):
    eps = 1e-5
    logits = _replace_nan(logits)
    if cut_zero:
        logits = _cut_label_zero(logits)
        y = _cut_label_zero(y)
    prediction = _pixel_wise_softmax(logits, dim=len(y.get_shape())-1)
    intersection = tf.reduce_sum(prediction * y) + eps
    union = tf.reduce_sum(prediction) + tf.reduce_sum(y) + eps
    dice = (2 * intersection) / union
    return dice


from numpy.core.umath_tests import inner1d

# Hausdorff Distance
def HausdorffDist(A, B):
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B)-2*(np.dot(A, B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))

    return dH


def accuracy_by_class(label):
    raise NotImplementedError


########## Display Result ##########
def visualize(config, img_info_list, image, pred_maxout, y_maxout, glob_num, index=0):

    save_path = config.get('data stats', 'save_path') + 'image/'
    num_class = config.getint('data attributes', 'num_class')

    img_info = img_info_list[index]
    file_name = str(img_info['coord'])

    # Consider 3D or 2D
    if len(image.shape) > 4:
        rand_idx = randint(0, image.shape[1]-1)
        image = image[:, rand_idx, :, :, :]
        pred_maxout = pred_maxout[:, rand_idx, :, :]
        y_maxout = y_maxout[:, rand_idx, :, :]
        file_name = file_name + '_' + str(rand_idx)

    # index = np.random.randint(image.shape[0])
    org_image = image[index, :, :, 0]
    ref_img = org_image[org_image.shape[0]//2 - y_maxout.shape[1]//2: org_image.shape[0]//2 + y_maxout.shape[1]//2,
             org_image.shape[1] // 2 - y_maxout.shape[2] // 2: org_image.shape[1] // 2 + y_maxout.shape[2] // 2]

    gt_img = y_maxout[index, :, :]
    err_img = gt_img - pred_maxout[index, :, :]

    # Plot for Image Data
    x_axis = np.linspace(0, ref_img.shape[0]-1, ref_img.shape[0])
    y_axis = np.linspace(ref_img.shape[1]-1, 0, ref_img.shape[1])

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(ref_img, 'gray')
    plt.title('Input Image : ' + file_name)
    plt.subplot(2, 2, 2)

    plt.pcolor(x_axis, y_axis, gt_img, vmin=0, vmax=num_class)
    plt.title('Ground truth')
    plt.subplot(2, 2, 3)
    plt.pcolor(x_axis, y_axis, pred_maxout[index, :, :], vmin=0, vmax=num_class)
    plt.title('Image Prediction')
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(err_img) > 0.5)
    plt.title('Error Image')
    plt.show()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(save_path + '/result_' + str(glob_num) + '.jpg')