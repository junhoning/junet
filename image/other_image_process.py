import numpy as np
import matplotlib.pyplot as plt
import scipy

from skimage.measure import label as seg_label
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing, binary_fill_holes


def search_largest_label(label, label_n, return_remain=False):
    labels = seg_label(np.where(label == label_n, 1, 0))

    size = np.bincount(labels.ravel())
    biggest_label = size[1:].argmax() + 1
    clump_mask = labels == biggest_label

    if return_remain:
        return clump_mask, np.where(labels == biggest_label, 0, 1)
    else:
        return clump_mask


def gaussian_kernel(sigma, truncate=4.0):
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = sigma ** 2

    k = 2 * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)
    k = k / np.sum(k)

    return k


rev_filters = -gaussian_kernel(1, 2.5)
filters = gaussian_kernel(1, 2.5)
result = scipy.ndimage.filters.convolve(target.astype(np.float32), rev_filters)
result = (result - np.min(result)) / (np.max(result) - np.min(result))
result = result * target

result_f = scipy.ndimage.filters.convolve(target.astype(np.float32), filters)
result_f = (result_f - np.min(result_f)) / (np.max(result_f) - np.min(result_f))
result_f = result_f * target

print('Result ', np.min(result), np.max(result))
print("Filter ", np.min(filters), np.max(filters))
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.title("Original Gaussian Filter")
plt.imshow(result_f)
plt.subplot(2, 2, 2)
plt.title("Inversed Gaussian Filter")
plt.imshow(result)
plt.subplot(2, 2, 3)
plt.title("Gaussian Filter")
plt.imshow(rev_filters)
plt.subplot(2, 2, 4)
plt.title("Original Label")
plt.imshow(target)
plt.show()