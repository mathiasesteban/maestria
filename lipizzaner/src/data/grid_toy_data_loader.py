import random

import torch
from torch.utils.data import Dataset
import matplotlib

from helpers.pytorch_helpers import to_pytorch_variable

matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from math import sqrt

from helpers.configuration_container import ConfigurationContainer
from data.data_loader import DataLoader
from data.gaussian_2d_dataset import Gaussian2DDataSet

class GridToyDataLoader(DataLoader):

    """
    A dataloader that returns samples from a simple toyp roblem distribution multiple fixed points in a grid
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(GridToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 2

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)

    @property
    def points(self):
        pass


class GridToyDataSet(Gaussian2DDataSet):

    @staticmethod
    def points(number_of_modes):
        points_per_row = int(sqrt(number_of_modes))
        points_per_col = int(number_of_modes/ points_per_row)
        size = 5

        incr_x = (size * 2) / (points_per_row - 1)
        incr_y = (size * 2) / (points_per_col - 1)
        xs = []
        ys = []
        for i in range(points_per_row):
            for j in range(points_per_col):
                xs.append((i * incr_x) - size)
                ys.append((j * incr_y) - size)
        return xs, ys

    colors = None



