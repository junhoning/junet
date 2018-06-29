import numpy as np


def onehot_encoding(label, num_classes):
    if len(np.unique(label)) > num_classes:
        raise ValueError("Invalid number of classes")

    onehot_shape = tuple(list(np.array(label).shape) + [num_classes])
    label_onehot = np.zeros(onehot_shape, dtype=np.uint8)

    for label_n in range(num_classes):
        if len(label.shape) == 2:
            label_onehot[:, :, label_n] = np.where(label == label_n, 1, 0)
        elif len(label.shape) == 3:
            label_onehot[:, :, :, label_n] = np.where(label == label_n, 1, 0)
        else:
            label_onehot[label_n] = np.where(label == label_n, 1, 0)

    return label_onehot