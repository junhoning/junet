import numpy as np


# Normalization
def z_score_norm(data):
    """
    z_score normalization
    :param data: n-dimensional array
    :return: normalized data
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data-mean)/std
    return normalized


def intensity_normalize(data):
    lmin = data.min()
    lmax = data.max()
    return np.floor((data-lmin)/(lmax-lmin)*255)


def histogram_equalization(data):
    histogram = np.histogram(data, bins=np.arange(len(data)+1))[0]
    histograms = np.cumsum(histogram) / float(np.sum(histogram))

    e = np.floor(histograms[data.flatten().astype('int')]*255)
    return e.reshape(data.shape)


def histogram_equal(data):
    hist, bins = np.histogram(data.flatten(), np.max(data), [np.min(data), np.max(data)+1])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return cdf_normalized


# Intensity Transformation
# NumPy / SciPy Recipes for Image Processing:“Simple” Intensity Transformations
# gamma correction of an intensity image
def correct_gamma(f, gamma=1.):
    '''
    gamma correction of an intensity image, where
    gamma = 1. : no effect
    gamma > 1. : image will darken
    gamma < 1. : image will brighten
    '''
    return 255. * (f/255.)**gamma


# Solarization
def solarize(f, nu=1.):
    '''
    nu=0.5 : inverse intensity
    nu=1.0 : v-shape
    nu=1.5 : N-shape
    nu=2.0 : W-shape
    '''
    return np.cos(f/255. * 2*np.pi * nu) * 127.5 + 127.5


# Soft Thresholding
def threshold(f, sigma=0.05):
    '''0.05 / 0.025'''
    return np.tanh((f-127.5) * sigma) * 127.5 + 127.5

