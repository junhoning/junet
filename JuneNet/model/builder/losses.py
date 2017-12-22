import tensorflow as tf

from JuneNet.model.builder import layers


def l1_loss(input_, target_, lamb=1.0, name="l1_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        loss = tf.multiply(tf.reduce_mean(tf.abs(input_ - target_)), lamb, name="loss")
        return loss


def l2_loss(input_, target_, lamb=1.0, name="l2_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        loss = tf.multiply(tf.reduce_mean(tf.square(input_ - target_)), lamb, name="loss")
        return loss


def cross_entropy_loss(predictions, groundtruth, lamb=1.0, name="ce_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=groundtruth, logits=predictions)
        loss = tf.multiply(lamb, tf.reduce_mean(cross_entropy), name="loss")
        return loss


def _weight_decay(_weight_decay_rate=0.0001):
    costs = []
    for var in tf.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    return tf.multiply(_weight_decay_rate,tf.add_n(costs))


def pixel_wise_l1_loss(input_, target_, lamb=1.0, name="pixel_l1"):
    return l1_loss(input_, target_, lamb, name)


def pixel_wise_l2_loss(input_, target_, lamb=1.0, name="pixel_l2"):
    return l2_loss(input_, target_, lamb, name)


def pixel_wise_cross_entropy(input_, target_, lamb=1.0, name="pixel_ce"):
    flat = layers.flatten(input_)
    return cross_entropy_loss(flat, target_, lamb, name)


def pixel_wise_sparse_softmax_entropy(logits,y):
    _y = tf.argmax(y, axis=3)
    _xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y,logits=logits)
    _cost = tf.reduce_mean(_xent)+_weight_decay()
    return _cost


def _pixel_wise_softmax(output_map):
    dim = len(output_map.get_shape().as_list()) - 1
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, dim, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1] * dim + [tf.shape(output_map)[dim]]))
    return tf.div(exponential_map, tensor_sum_exp)


def weight_loss(logits, weight_list):
    '''
    :param logits: Tensor Type. Output from tensor
    :param weight_list: 'list' type.
    '''
    if logits.get_shape()[-1] != len(weight_list):
        raise ValueError("Weight Length has to be same as number of channel. Logit Shape :", logits.get_shape())
    else:
        return logits * weight_list


def weight_by_input(label, log=False, remove_background=True):
    if log:
        weights = tf.log(tf.reduce_sum(label) / tf.reduce_sum(label, axis=list(range(len(label.shape)-1))))
    else:
        weights = tf.reduce_sum(label) / tf.reduce_sum(label, axis=list(range(len(label.shape)-1)))
    if remove_background:
        return tf.concat([tf.constant([1e-5], dtype=weights.dtype), tf.split(weights, [1, -1])[1]], axis=-1)
    else:
        return weights


def weighted_cross_entropy(logits, y, weight_list):
    '''
    :param logits: Tensor Type. Output from tensor
    :param weight_list: 'list' type.
    '''
    if logits.get_shape()[-1] != len(weight_list):
        raise ValueError("Weight Length has to be same as number of channel. Logit Shape :", logits.get_shape())
    else:
        weighted_loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(y, logits, tf.constant(weight_list, dtype=logits.dtype)))
        return weighted_loss


def binary_dice_coef(predictions, groundtruth):
    eps = 1e-5
    intersection = tf.reduce_sum(predictions * groundtruth)
    union = eps + tf.reduce_sum(predictions) + tf.reduce_sum(groundtruth)
    dice = -(2 * intersection / union)


# Dice Series
def loss_dice_coef(predictions, groundtruth):
    if groundtruth.dtype != predictions.dtype:
        groundtruth = tf.cast(groundtruth, predictions.dtype)
    eps = 1e-5
    class_n = predictions.get_shape()[-1].value
    predictions = _pixel_wise_softmax(predictions)
    logit_splitted = tf.split(predictions, [1] * class_n, axis=len(predictions.get_shape())-1)
    label_splitted = tf.split(groundtruth, [1] * class_n, axis=len(groundtruth.get_shape())-1)

    dice_stack = []
    for logit_l, label_l in zip(logit_splitted, label_splitted):
        intersection = tf.reduce_sum(logit_l * label_l)
        union = eps + tf.reduce_sum(logit_l) + tf.reduce_sum(label_l)
        dice = -(2 * intersection / union)
        dice_stack.append(dice)

    return tf.reduce_mean(dice_stack)


# Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations
def generalised_loss_dice_coef(logits, y):
    eps = 1e-5
    prediction = _pixel_wise_softmax(logits)
    intersection = tf.reduce_sum(prediction * y) + eps
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y) + eps
    all_errors = tf.reduce_sum((1-prediction) * (1-y)) + eps
    dice = 1-(intersection / union) - (all_errors / (2 - union))
    return dice


def loss_jaccard_coef(logits, y):
    eps = 1e-5
    prediction = _pixel_wise_softmax(logits)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    jaccard = -(intersection / (union-intersection))
    return jaccard


# Made this func to solve NaN Problem.
# Add this code to input of the cost function and Try it.
# It turns to OneHot from All Zeros in Logit. [0,0,0] -> [1,0,0]
# Logit occurs to all zeros before turns to NaN
def change_all_zeros_to_one_hot(logit):
    logit_sum = tf.reduce_sum(logit, axis=-1)
    logit_shape = logit.get_shape().as_list()
    condition = tf.equal(logit_sum, tf.zeros(logit_shape[:-1]))
    condition = tf.reshape(condition, shape=logit_shape[:-1] + [1])

    zero_one = tf.concat([tf.ones(logit_shape[:-1] + [1]), tf.zeros(logit_shape[:-1] + [logit_shape[-1] - 1])], axis=-1)
    answer = tf.where(tf.concat([condition] * 3, axis=-1), x=zero_one, y=logit, name="NaN_convert")

    return answer


def weight_loss_dice_coef(logits, y, weight_list):
    eps = 1e-10
    weights = tf.constant(weight_list, dtype=logits.dtype)
    prediction = _pixel_wise_softmax(logits)
    prediction = tf.clip_by_value(prediction, eps, 0.9999999)
    intersection = tf.reduce_sum(prediction * y * weights)
    union = eps + tf.reduce_sum(prediction * weights) + tf.reduce_sum(y * weights)
    dice = -(2 * intersection / union)
    return dice


def loss_jaccard_coef(logits, y):
    eps = 1e-10
    prediction = _pixel_wise_softmax(logits)
    prediction = tf.clip_by_value(prediction, eps, 0.9999999)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y) - intersection
    jaccard = -(intersection / union)
    return jaccard


def ent_dice_coef(logits, y, weight_list):
    return tf.log(2+weight_loss_dice_coef(logits=logits, y=y, weight_list=weight_list))


# Paper Source: 2D-3D Fully Convolutional Neural Networks for Cardiac MR Segmentation
def weighted_ce_dice(logits, y, weight_list):
    weighted_ce = weighted_cross_entropy(logits, y, weight_list)
    weighted_dice = ent_dice_coef(logits, y, weight_list)
    return weighted_ce + weighted_dice


# Paper Source: Hybrid Loss Guided Convolutional Networks for Whole Heart Parsing
def hybrid_cost(logits, y, weight_list, alpha=1.5):
    weighted_ce = weighted_cross_entropy(logits, y, weight_list)
    cost_dice = loss_dice_coef(logits, y)
    return (weighted_ce * alpha) + (1 + cost_dice)


def _ASD(predicts, annots):
    """
    distance is calculated as euclidean distnace
    :param predicts: MxN matrix, M = #ofbatch, N:dimension
    :param annots: MxN matrix, M = #ofbatch, N:dimension
    :return:
    """
    r = tf.reduce_sum(predicts * annots, 1)
    r = tf.reshape(r, [-1, 1])
    sds = tf.reduce_sum(D == r - 2*tf.matmul(predicts, tf.transpose(predicts)) + tf.transpose(r))
    asd = tf.divide(sds,tf.reduce_sum(tf.sqrt(tf.square(predicts))))
    return asd


def ASSD_loss(predicts, annots):
    asd1 = _ASD(predicts, annots)
    asd2 = _ASD(annots, predicts)
    numerator = tf.add(asd1, asd2)
    cost = tf.divide(numerator, 2.0)
    return cost


def HDHD_loss(predicts, annots):
    """
    Hausdorf distance between two 3D volume
    :param predicts: prediction matrix - predictions [batch,width,height,depth]
    :param annots: ground-truth - annotation [batch, ~]
    :return:cost
    """
    A = tf.reshape(predicts, [-1])
    B = tf.reshape(annots, [-1])
    dotA = tf.matmul(A, A, transpose_a=False, transpose_b=True)
    dotB = tf.matmul(B, B, transpose_a=False, transpose_b=True)
    D_mat = tf.sqrt(dotA + dotB - 2 * tf.matmul(A, B, transpose_a=False, transpose_b=True))
    hd_cost = tf.reduce_max(tf.reduce_max(tf.reduce_min(D_mat, axis=0)), tf.reduct_max(tf.reduce_min(D_mat, axis=1)))
    return hd_cost
