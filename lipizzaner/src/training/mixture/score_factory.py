import os
import pathlib

from torchvision import transforms

from helpers.configuration_container import ConfigurationContainer
from helpers.ignore_label_dataset import IgnoreLabelDataset
from training.mixture.fid_score import FIDCalculator
from training.mixture.inception_score import InceptionCalculator
from training.mixture.constant_score import ConstantCalculator
from training.mixture.gaussian_score import GaussianToyDistancesCalculator


class ScoreCalculatorFactory:

    @staticmethod
    def create():
        cc = ConfigurationContainer.instance()
        settings = cc.settings['trainer']['params']

        if 'score' not in settings:
            return None

        score_type = settings['score'].get('type', None)
        dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
        # Downloads dataset if its not yet available
        dataloader.load()

        if score_type == 'gaussian_toy_distances':
            number_of_modes = cc.settings['dataloader']['number_of_modes']
            return GaussianToyDistancesCalculator(dataloader.dataset.points(number_of_modes))
        elif score_type == 'fid':
            transforms_op = [transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
            if cc.settings['dataloader']['dataset_name'] != 'mnist':
                # Need to reshape for RGB dataset as required by pre-trained InceptionV3
                transforms_op = [transforms.Resize([64, 64])] + transforms_op

            dataset_path = str(pathlib.Path(__file__).parent.absolute()) + "/../../data/datasets/" + cc.settings['dataloader']['dataset_name']    

            dataset = dataloader.dataset(root=dataset_path, train=True,
                                         transform=transforms.Compose(transforms_op))

            return FIDCalculator(IgnoreLabelDataset(dataset), cuda=cc.settings['master'].get('cuda', False),
                                 n_samples=settings['score'].get('score_sample_size', 10000))
        elif score_type == 'inception_score':
            # CUDA may not work when multiple  nodes, as it uses high amounts of GPU memory (~3GB per instance)
            return InceptionCalculator(cuda=cc.settings['master'].get('cuda', False), resize=True)
        elif score_type == 'constant':
            return ConstantCalculator(cuda=cc.settings['master'].get('cuda', False), resize=True)
        else:
            raise Exception('Mixture score type {} is not supported. Use either "inception_score" or "fid".'
                            .format(score_type))
