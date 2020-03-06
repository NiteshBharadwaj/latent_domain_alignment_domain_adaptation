import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_cars import Dataset
import numpy as np

class CombinedData(Dataset):
    def __init__(self, data_loader_S, data_loader_t, max_dataset_size):
        super(Dataset,self).__init__()
        self.data_loader_S = data_loader_S
        self.num_S = len(data_loader_S)
        self.stop_S = []
        self.stop_t = False
        for i in range(self.num_S):
            self.stop_S.append(False)
        self.max_dataset_size = max_dataset_size
        self.num_datasets = self.num_S
        self.data_loader_S_iter = []
        for i in range(self.num_S):
            self.data_loader_S_iter.append(iter(self.data_loader_S[i]))
        self.data_loader_t = data_loader_t
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0

    def __getitem__(self, index):
        S, S_paths,t,t_paths = None, None, None, None
        i = index%self.num_datasets
        try:
            S, S_paths = next(self.data_loader_S_iter[i])
        except StopIteration:
            if S is None or S_paths is None:
                self.stop_S[i] = True
                self.data_loader_S_iter[i] = iter(self.data_loader_S[i])
                S, S_paths = next(self.data_loader_S_iter[i])
        try:
            t, t_paths = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths = next(self.data_loader_t_iter)

        if (np.prod(self.stop_S)>0 and self.stop_t==True) or self.iter > self.max_dataset_size:
            for i in range(len(self.stop_S)):
                self.stop_S[i] = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': S, 'S_label': S_paths,
                    'T': t, 'T_label': t_paths}

    def __len__(self):
        return self.max_dataset_size*self.num_datasets


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):
        transform = transforms.Compose([
            transforms.Resize((scale,scale)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_sources = []
        data_loader_s = []
        max_size = 0
        for i in range(len(source)):
            data_sources.append(Dataset(source[i]['imgs'], source[i]['labels'], transform=transform))
            data_loader_s.append(torch.utils.data.DataLoader(data_sources[i], batch_size=batch_size1, shuffle=True, num_workers=4))
            max_size = max(max_size,len(data_sources[i]))
        self.dataset_s = data_loader_s

        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True, num_workers=4)

        self.dataset_t = dataset_target
        self.paired_data = CombinedData(data_loader_s, data_loader_t,
                                      float("inf"))

        self.num_datasets = len(source)
        self.num_samples = min(max(max_size,len(self.dataset_t)), float("inf"))*self.num_datasets


    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return self.num_samples
