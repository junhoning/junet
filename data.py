import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, input_shape, num_classes, batch_size):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _parse_image_function(self, example_proto):

        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            
            'min_value': tf.io.FixedLenFeature([], tf.int64),
            'max_value': tf.io.FixedLenFeature([], tf.int64)
        }

        return tf.io.parse_single_example(example_proto, image_feature_description)

    def data_reader(self, image_features):
        image_raw = image_features['image']
        label_raw = image_features['label']

        height = image_features['height']
        width = image_features['width']
        depth = image_features['depth']

        image = tf.io.decode_raw(image_raw, tf.uint8)
        # image = (image - image_features['min_value']) / (image_features['max_value'] - image_features['min_value'])
        image = image / 255
        image = tf.reshape(image, [height, width, depth])

        label = tf.io.decode_raw(label_raw, tf.uint8)
        label = tf.reshape(label, [height, width, depth])
        
        return image, label

    def normalization(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
        

    @tf.function
    def random_crop(self, image, label, crop_shape, seed=1):
        image = tf.image.random_crop(image, crop_shape, seed=seed)
        label = tf.image.random_crop(label, crop_shape, seed=seed)
        return image, label


    def preprocess_data(self, image, label):

        image, label = self.random_crop(image, label, self.input_shape[:-1])
        
        image = image[..., tf.newaxis]
        label = tf.one_hot(label, self.num_classes)
        
        return image, tf.cast(label, tf.float32)


    def test_preprocess(self, image, model):
        # normalization 
        image = self.normalization(image)
        
        # Cropping Image
        input_shape = np.array([128, 128, 128])
        image_shape = np.array(image.shape)

        pad_shape = np.stack([input_shape//2, (input_shape - image_shape % input_shape) + (input_shape//2)], -1)
        padded = np.pad(image, pad_shape, mode='constant')

        h, w, d = input_shape

        preds = np.zeros(list(padded.shape) + [3])
        for i in range(padded.shape[0] // h):
            for j in range(padded.shape[1] // w):
                for k in range(padded.shape[2] // d):
                    input_img = padded[(i*h):((i+1)*h), (j*w):((j+1)*w), (k*d):((k+1)*d)]
                    logit = model(tf.constant(input_img[tf.newaxis, ..., tf.newaxis], dtype=tf.float32), training=False)
                    preds[(i*h):((i+1)*h), (j*w):((j+1)*w), (k*d):((k+1)*d)] += np.squeeze(logit)
        pred = np.argmax(preds, -1)
        return pred[64:64+image.shape[0], 64:64+image.shape[1], 64:64+image.shape[2]]


    def get_dataset(self, filename, is_training=True):
        dataset = tf.data.TFRecordDataset(filename)
        if is_training:
            dataset = dataset.shuffle(100)
        dataset = dataset.map(self._parse_image_function)
        dataset = dataset.map(self.data_reader)

        dataset = dataset.map(self.preprocess_data)
        dataset = dataset.batch(self.batch_size)
        return dataset