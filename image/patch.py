import numpy as np
import warnings
from ..engine.data_manager import DataManager


class DataInput(object):
    def __init__(self, path):
        self.path = path

    def get_reader(self, reader_fn):
        image = reader_fn(self.path)
        return image

    def preprocess(self, flip):
        pass

