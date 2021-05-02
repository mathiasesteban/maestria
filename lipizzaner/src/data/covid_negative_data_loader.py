import logging

from helpers.configuration_container import ConfigurationContainer
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils.data import Dataset
from data.data_loader import DataLoader
from torchvision.utils import save_image

from PIL import Image
import torch
from torch.autograd import Variable

from imblearn.over_sampling import SMOTE

WIDTH = 128
HEIGHT = 128


def gaussian_augmentation(tensor_list, labels_list, augmentation_times, mean, std):

    augmented_tensor_list = []
    augmented_labels_list = []

    augmented_tensor_list.extend(tensor_list)
    augmented_labels_list.extend(labels_list)

    for i in range(augmentation_times-1):

        index = 0
        for img in tensor_list:
            print(len(tensor_list))
            input_perturbation = Variable(torch.empty(img.shape).normal_(mean=mean, std=std))
            new_tensor = img + input_perturbation
            augmented_tensor_list.append(new_tensor)
            augmented_labels_list.append(labels_list[index])
            index += 1

    return augmented_tensor_list, augmented_labels_list


def smote_augmentation(tensor_list, labels_list, augmentation_times):
    # input list_tensors
    # X -> Lista de tensores
    # Y -> Lista de enteros representando las clases

    smote = SMOTE()

    # Se agregan tensores de otra clase para controlar la proporcion generada
    # La implementacion por defecto de imbalanced-learn igual las proporciones entre clases
    # Si se quiere aumentar n veces la cantidad de tensores en tensor_list se deben agregar len(tensor_list)*n vectores
    # de relleno.

    for x in range(augmentation_times*len(tensor_list)):
        input_perturbation = Variable(torch.empty(tensor_list[0].shape).normal_(mean=0.5, std=0.001))
        tensor_list.append(input_perturbation)
        # Importante agregarlo con una etiqueta diferente a la original
        labels_list.append(-1)

    stack = torch.stack(tensor_list)
    n_samples = stack.shape[0]
    colour_dimension = stack.shape[1]
    heigth = stack.shape[2]
    width = stack.shape[3]
    to_smote = stack.reshape(n_samples, colour_dimension * heigth * width)   # (n_samples, COLOUR * HEIGTH * WIDTH)

    sm = SMOTE()
    smoted_stack, smoted_labels = sm.fit_sample(to_smote, labels_list)


    augmented_tensor_list = []
    augmented_labels_list = []

    index = 0
    for x in smoted_stack:
        if smoted_labels[index] is not (-1):
            augmented_tensor_list.append(torch.from_numpy(x.reshape(colour_dimension, heigth, width)))
            augmented_labels_list.append(smoted_labels[index])
        index += 1

    return augmented_tensor_list, augmented_labels_list



class CovidNegativeDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=1, n_batches=0, shuffle=False):
        super().__init__(COVIDNegativeDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return WIDTH*HEIGHT

    @staticmethod
    def save_images(images, shape, filename):

        # img_view = data.view(num_images, 1, WIDTH, HEIGHT)
        img_view = images.view(images.size(0), 1, WIDTH, HEIGHT)
        # img_view = images.view(images)
        save_image(img_view, filename)


class COVIDNegativeDataSet(Dataset):

    def __init__(self, **kwargs):

        self.cc = ConfigurationContainer.instance()
        settings = self.cc.settings['dataloader']
        self.smote_augmentation_times = settings.get('smote_augmentation_times', 0)
        self.gaussian_augmentation_times = settings.get('gaussian_augmentation_times', 0)
        self.gaussian_augmentation_mean = settings.get('gaussian_augmentation_mean', 0)
        self.gaussian_augmentation_std = settings.get('gaussian_augmentation_std', 0)

        self.covid_type = settings.get('covid_type', 'positive')

        self.use_batch = settings.get('use_batch', False)

        self.batch_size = settings.get('batch_size', None) if self.use_batch else None

        self._logger = logging.getLogger(__name__)

        # 1) CARGAR IMAGENES DESDE EL FILESYSTEM

        # Se cargan las imagenes en una lista de tuplas <tensor,int> donde:
        # tensor.shape = (1, HEIGHT, WIDTH)
        # int es el indice de la clase asociada a dicho tensor

        transforms = [Grayscale(num_output_channels=1), Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST),
                      ToTensor()]
        dataset = ImageFolder(root="data/datasets/covid-negative", transform=Compose(transforms))
        print(len(dataset))

        # Se separan las tuplas en lista de tensores y lista de labels
        tensor_list = []
        labels_list = []
        for img in dataset:
            tensor_list.append(img[0])
            labels_list.append(img[1])

        print("Original dataset size: " + str(len(tensor_list)))

        # 2) AUGMENTATION

        if self.gaussian_augmentation_times != 0:
            tensor_list, labels_list = gaussian_augmentation(tensor_list,
                                                             labels_list,
                                                             self.gaussian_augmentation_times,
                                                             self.gaussian_augmentation_mean,
                                                             self.gaussian_augmentation_std)

        self._logger.debug('Dataset size after Gaussian augmentation: {}'.format(len(tensor_list)))
        print("Dataset size after Gaussian augmentation: " + str(len(tensor_list)))

        if self.smote_augmentation_times is not None:
            tensor_list, labels_list = smote_augmentation(tensor_list, labels_list, self.smote_augmentation_times)

        self._logger.debug('Dataset size after SMOTE augmentation: {}'.format(len(tensor_list)))
        print("Dataset size after SMOTE augmentation: " + str(len(tensor_list)))

        # 3) REMOVER BATCH INCOMPLETO

        # Remuevo los ultimos elementos que no completan un batch
        if self.use_batch:
            reminder = len(tensor_list) % self.batch_size
            if reminder > 0:
                tensor_list = tensor_list[:-reminder]

        # 4) UNIFICAR LISTA EN TENSOR UNICO
        # Se conVierte la lista de tensores en un unico tensor de dimension (len(tensor_list), 1, HEIGHT, WIDTH)
        stacked_tensor = torch.stack(tensor_list)

        self._logger.debug('Final dataset shape: {}'.format(stacked_tensor.shape))
        print("Final dataset shape: " + str(stacked_tensor.shape))

        self.data = stacked_tensor
        self.labels = labels_list

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
