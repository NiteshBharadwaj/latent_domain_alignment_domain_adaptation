import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_ import Dataset, Dataset2

import numpy as np


class ClasswiseData(object):
    def __init__(self, data_loader_s, max_dataset_size):
        self.data_loader_s = data_loader_s
        self.num_datasets = len(data_loader_s)
        stop_source = []
        for i in range(self.num_datasets):
            stop_source.append(False)
        self.stop_source = stop_source
        self.max_dataset_size = max_dataset_size

    def __iter__(self):

        for i in range(self.num_datasets):
            self.stop_source[i] = False
        data_loader_s_iter = []
        for i in range(self.num_datasets):
            data_loader_s_iter.append(iter(self.data_loader_s[i]))
        self.data_loader_s_iter = data_loader_s_iter
        self.iter = 0
        return self

    def __next__(self):
        source = None
        source_paths = None
        i = np.random.randint(10)

        try:
            source, source_paths = next(self.data_loader_s_iter[i])
        except StopIteration:
            if source is None or source_paths is None:
                self.stop_source[i] = True
                self.data_loader_s_iter[i] = iter(self.data_loader_s[i])
                source, source_paths = next(self.data_loader_s_iter[i])

        self.iter += 1
        return {'S': source, 'S_label': source_paths,
                'T': source, 'T_label': source_paths}


class ClasswiseDataLoader():
    def initialize(self, source, batch_size, scale=32):
        transform = transforms.Compose([
            transforms.Scale(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset_source = []
        dataloader_source = []

        self.max_len = 0
        for i in range(10):
            img_list = []
            label_list = []
            for j in range(len(source)):
                imgs = source[j]['imgs']
                labels = source[j]['labels']
                mask = labels == i
                img_list.append(imgs[mask])
                label_list.append(labels[mask])
            dataset_source.append(Dataset2(img_list, label_list, transform=transform))
            self.max_len = max(self.max_len, len(dataset_source))
            dataloader_source.append(
                torch.utils.data.DataLoader(dataset_source[i], batch_size=batch_size, shuffle=True,
                                            num_workers=2))

        self.dataset_s = dataset_source
        self.paired_data = ClasswiseData(dataloader_source, float("inf"))

    def name(self):
        return 'ClasswiseDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return float("inf")
