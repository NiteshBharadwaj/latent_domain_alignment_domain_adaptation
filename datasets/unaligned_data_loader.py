import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from .datasets_ import Dataset


class PairedData(object):
    def __init__(self, data_loader_s, data_loader_t, max_dataset_size):
        self.data_loader_s = data_loader_s
        self.data_loader_t = data_loader_t
        self.num_datasets = len(data_loader_s)
        stop_source = []
        for i in range(self.num_datasets):
            stop_source.append(False)
        self.stop_source = stop_source

        self.stop_t = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):

        for i in range(self.num_datasets):
            self.stop_source[i] = False

        self.stop_t = False

        data_loader_s_iter = []
        for i in range(self.num_datasets):
            data_loader_s_iter.append(iter(self.data_loader_s[i]))
        self.data_loader_s_iter = data_loader_s_iter
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0
        return self

    def __next__(self):
        source = []
        source_paths = []

        for i in range(self.num_datasets):
            source.append(None)
            source_paths.append(None)

        t, t_paths = None, None

        for i in range(self.num_datasets):
            try:
                source[i], source_paths[i] = next(self.data_loader_s_iter[i])
            except StopIteration:
                if source[i] is None or source_paths[i] is None:
                    self.stop_source[i] = True
                    self.data_loader_s_iter[i] = iter(self.data_loader_s[i])
                    source[i], source_paths[i] = next(self.data_loader_s_iter[i])
        try:
            t, t_paths = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths = next(self.data_loader_t_iter)

        if (all(k for k in self.stop_source) and self.stop_t) or self.iter > self.max_dataset_size:
            for i in range(self.num_datasets):
                self.stop_source[i] = False
                self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': source, 'S_label': source_paths,
                    'T': t, 'T_label': t_paths}


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size_source, batch_size_target, scale=32):
        transform = transforms.Compose([
            transforms.Scale(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset_source = []
        dataloader_source = []

        self.max_len = 0
        for i in range(len(source)):
            dataset_source.append(Dataset(source[i]['imgs'], source[i]['labels'], transform=transform))
            self.max_len = max(self.max_len, len(dataset_source))
            dataloader_source.append(
                torch.utils.data.DataLoader(dataset_source[i], batch_size=batch_size_source, shuffle=True,
                                            num_workers=4))

        self.dataset_s = dataset_source

        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        self.max_len = max(self.max_len, len(dataset_target))
        dataloader_target = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size_target, shuffle=True,
                                                        num_workers=4)

        self.dataset_t = dataset_target
        self.paired_data = PairedData(dataloader_source, dataloader_target, float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(self.max_len, float("inf"))
