import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, input_shape, num_classes, batch_size, augmentations, is_training):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.is_training = True
        self.num_dims = len(self.input_shape)-1

    def _parse_image_function_3d(self, example_proto):

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

    def data_reader_3d(self, image_features):
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

    def _parse_image_function(self, example_proto):

        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channel': tf.io.FixedLenFeature([], tf.int64),
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
        channel = image_features['channel']

        image = tf.io.decode_image(image_raw) # image = tf.io.decode_raw(image_raw, tf.uint8)
        # image = (image - image_features['min_value']) / (image_features['max_value'] - image_features['min_value'])
        image = image / 255
        # image = self.normalization(image)
        image = tf.reshape(image, [height, width, channel])

        label = tf.io.decode_image(label_raw) # label = tf.io.decode_raw(label_raw, tf.uint8)
        label = tf.reshape(label, [height, width])
        
        return image, label

    def normalization(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
        

    @tf.function
    def random_crop(self, image, label, seed=1):
        image = tf.image.random_crop(image, self.input_shape[:-1], seed=seed)
        label = tf.image.random_crop(label, self.input_shape[:-1], seed=seed)
        return image, label


    @tf.function
    def resize_data(self, image, label):
        image = tf.image.resize(image, self.input_shape[:-1])

        label = label[..., tf.newaxis]
        label = tf.image.resize(label, self.input_shape[:-1], method='nearest')
        label = tf.squeeze(label)
        return image, label


    def preprocess_data(self, image, label):

        preprocess_funcs = {'resize' : self.resize_data, 'random_crop' : self.random_crop}

        for aug_name in self.augmentations:
            aug_func = preprocess_funcs[aug_name]
            image, label = aug_func(image, label)
        
        if self.num_dims == 3:
            image = image[..., tf.newaxis]
        label = tf.one_hot(label, self.num_classes)
        
        return image, tf.cast(label, tf.float32)


    def test_preprocess(self, image, model):
        # normalization 
        image = self.normalization(image)
        
        # Cropping Image
        input_shape = np.array(self.input_shape[:-1])
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


    def get_dataset(self, filename, split_rate=0.7):
        if filename.split(".")[-1] == 'gzip':
            compression_type = 'GZIP'
        else:
            compression_type = None


        if split_rate > 0:
            num_ds = len([1 for _ in tf.data.TFRecordDataset(filename, compression_type='GZIP')])

            train_size = int(split_rate * num_ds)
            val_size = int((1-split_rate) * num_ds)
            # test_size = int(0.15 * num_ds)

            dataset = tf.data.TFRecordDataset(filename, compression_type)
            dataset = dataset.shuffle()

            train_dataset = dataset.take(train_size)
            test_dataset = dataset.skip(train_size)
            # val_dataset = test_dataset.skip(test_size)
            test_dataset = test_dataset.take(val_size)

            if self.num_dims == 2:
                train_dataset = train_dataset.map(self._parse_image_function)
                train_dataset = train_dataset.map(self.data_reader)

                test_dataset = test_dataset.map(self._parse_image_function)
                test_dataset = test_dataset.map(self.data_reader)

            elif self.num_dims == 3:
                train_dataset = train_dataset.map(self._parse_image_function_3d)
                train_dataset = train_dataset.map(self.data_reader_3d)

                test_dataset = test_dataset.map(self._parse_image_function_3d)
                test_dataset = test_dataset.map(self.data_reader_3d)
            
            train_dataset = train_dataset.map(self.preprocess_data)
            train_dataset = train_dataset.batch(self.batch_size)
            
            test_dataset = test_dataset.map(self.preprocess_data)
            test_dataset = test_dataset.batch(self.batch_size)

            return train_dataset, test_dataset

        else:
            dataset = tf.data.TFRecordDataset(filename, compression_type)
            dataset = dataset.shuffle()

            if self.is_training:
                dataset = dataset.shuffle(100)
            if self.num_dims == 2:
                dataset = dataset.map(self._parse_image_function)
                dataset = dataset.map(self.data_reader)
            elif self.num_dims == 3:
                dataset = dataset.map(self._parse_image_function_3d)
                dataset = dataset.map(self.data_reader_3d)
            
            dataset = dataset.map(self.preprocess_data)
            dataset = dataset.batch(self.batch_size)

            return dataset