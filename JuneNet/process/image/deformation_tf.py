import tensorflow as tf


'''I'm going to add data deformations in tensorflow in below codes'''
deformation_dict = {}


def base_crop(*dataset, input_size):
    results = []
    for data in dataset:
        data = tf.slice(data, input_size)
        results.append(data)
    return results


def deformation(data, aug_dict):
    for augment in list(aug_dict.keys):
        data = deformation_dict[augment](aug_dict[augment])
    return data


    image = tf.slice(image, [x, y, z], size)
    weight_map = tf.slice(weight_map, [x, y, z, 0], size+[9])
    label = tf.slice(label, [x, y, z], size)