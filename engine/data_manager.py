import numpy as np
import warnings


class DataManager(object):
    def __init__(self, image_train_list, label_train_list, image_valid_list=[], label_valid_list=[]):
        if len(image_train_list) != len(label_train_list):
            raise ValueError("Number of images and labels are not equal")

        self.image_train_list = image_train_list
        self.label_train_list = label_train_list
        self.num_data = len(self.image_train_list)
        self.image_valid_list = image_valid_list
        self.label_valid_list = label_valid_list
        self.data_list = [(image, label) for image, label in zip(self.image_train_list, self.label_train_list)]

        # init data manager
        self.is_patch = False

    def validate_data(self, valid_rate=0.1, shuffle=True):
        if len(self.image_valid_list) > 0:
            raise ValueError("Validation Set already exists")
        if shuffle:
            np.random.shuffle(self.data_list)
        valid_list = self.data_list[-int(len(self.data_list) * valid_rate):]
        for image, label in valid_list:
            self.image_valid_list.append(image)
            self.label_valid_list.append(label)

    def reader_fn(self, image_fn, label_fn):
        self.image_fn = image_fn
        self.label_fn = label_fn

    def data_parameter(self, org_shape, num_class):
        self.org_shape = org_shape
        self.input_shape = org_shape
        self.num_class = num_class
        self.ndim = len(self.input_shape)

    def preprocess_fn(self, process_fn):
        self.process_fn = process_fn

    def batch(self, batch_size):
        self.batch_size = batch_size
        self.batches_per_epoch = len(self.image_train_list) // batch_size

    def get_coords(self, num_patches=None, patch_shape=None):
        self.is_patch = True

        if patch_shape:
            num_patches = np.ceil(patch_shape / self.org_shape).astype(np.int8)
        elif not num_patches:
            raise ValueError("Num_patches or Patch_shape have to be given")
        self.input_shape = patch_shape

        def get_patches(ndim):
            steps, step_size = np.linspace(0, self.org_shape[ndim] - patch_shape[ndim],
                                           num_patches[ndim],
                                           retstep=True,
                                           dtype=np.int32)
            return tuple(steps)

        self.coords = [get_patches(n) for n in range(self.ndim)]
        self.data_list = [(path, patch_n) for path in self.data_list for patch_n in range(len(self.coords))]

    def patch(self, image, patch_n):
        coord = self.coords[patch_n]
        if self.ndim == 2:
            return image[coord[0]:coord[0]+self.input_shape[0],
                   coord[1]:coord[1] + self.input_shape[1]]
        elif self.ndim == 3:
            return image[coord[0]:coord[0]+self.input_shape[0],
                   coord[1]:coord[1] + self.input_shape[1],
                   coord[2]:coord[2] + self.input_shape[2]]

    def sample_generator(self, batch_n, shuffle=True):
        if batch_n == 0 and shuffle:
            np.random.shuffle(self.data_list)

        batch_list = self.data_list[batch_n * self.batch_size: (batch_n + 1) * self.batch_size]

        image_concat_list = []
        label_concat_list = []

        self.data_info = {}
        self.data_info['patch_n'] = []

        for (image_path, label_path), patch_n in batch_list:
            image = self.image_fn(image_path)
            image = self.patch(image, patch_n)
            label = self.label_fn(label_path)
            label = self.patch(label, patch_n)
            image, label = self.process_fn(image, label)
            image_concat_list.append(np.array([image]))
            label_concat_list.append(np.array([label]))
            self.data_info['patch_n'].append(self.coords[patch_n])

        self.batch_image = np.concatenate(image_concat_list, axis=0)
        self.batch_label = np.concatenate(label_concat_list, axis=0)

    def generator(self, batch_n, shuffle=True):
        if batch_n == 0 and shuffle:
            np.random.shuffle(self.data_list)

        batch_list = self.data_list[batch_n * self.batch_size: (batch_n + 1) * self.batch_size]

        image_concat_list = []
        label_concat_list = []
        for image_path, label_path in batch_list:
            image = self.image_fn(image_path)
            label = self.label_fn(label_path)
            image, label = self.process_fn(image, label)
            image_concat_list.append(np.array([image]))
            label_concat_list.append(np.array([label]))

        self.batch_image = np.concatenate(image_concat_list, axis=0)
        self.batch_label = np.concatenate(label_concat_list, axis=0)

    def generator2(self, batch_n, shuffle=True):
        if batch_n == 0 and shuffle:
            np.random.shuffle(self.data_list)

        batch_list = self.data_list[batch_n * self.batch_size: (batch_n + 1) * self.batch_size]

        image_concat_list = []
        label_concat_list = []

        patch_nums = []
        for (image_path, label_path), patch_n in batch_list:
            image = self.image_fn(image_path)
            image = self.patch(image, patch_n)
            label = self.label_fn(label_path)
            label = self.patch(label, patch_n)
            image, label = self.process_fn(image, label)
            image_concat_list.append(np.array([image]))
            label_concat_list.append(np.array([label]))
            patch_nums.append(self.coords[patch_n])

        batch_image = np.concatenate(image_concat_list, axis=0)
        batch_label = np.concatenate(label_concat_list, axis=0)

        class DataInput(self):
            pass

        DataInput.batch_image = batch_image
        DataInput.batch_image = batch_label

        return DataInput
