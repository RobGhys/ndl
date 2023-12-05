import numpy as np

import torch
import torch.utils.data


def _count_labels(dataset):
    counts = {}
    targets = np.array([label for _, label in dataset])
    counts[0] = np.sum(targets == 0)
    counts[1] = np.sum(targets == 1)
    return counts


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
        super().__init__(data_source=dataset)

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        self.counts = _count_labels(dataset)
        self.weights = [1.0 / self.counts[i] for i in [0, 1]]

        weights = [1.0/(self.counts[dataset[idx][1]]) for idx in self.indices]
        self.weights = torch.tensor(weights, dtype=torch.double)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
