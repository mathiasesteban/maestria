import numpy as np

from data.data_loader import DataLoader
from data.gaussian_2d_dataset import Gaussian2DDataSet


class CircularToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a circle
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(CircularToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 2

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class CircularToyDataSet(Gaussian2DDataSet):

    @staticmethod
    def points(number_of_modes):
        thetas = np.linspace(0, 2 * np.pi, number_of_modes+1)[:-1]
        return np.sin(thetas) * 10, np.cos(thetas) * 10
