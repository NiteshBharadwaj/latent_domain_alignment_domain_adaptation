import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_ import Dataset, Dataset2


class CombinedData(Dataset):
    def __init__(self, data_loader_s, data_loader_t, max_dataset_size):
        super(Dataset,self).__init__()
        self.data_loader_s = data_loader_s
        self.data_loader_t = data_loader_t
     
        self.stop_s = False
        self.stop_t = False
        self.max_dataset_size = max_dataset_size
        self.data_loader_s_iter = iter(self.data_loader_s)
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0

    def reset(self):
        self.stop_s = False
        self.stop_t = False
        self.data_loader_s_iter = iter(self.data_loader_s)
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0
        return self

    
    def __getitem__(self, index):
        S, S_paths,SD,t,t_paths = None, None, None, None, None
        try:
            S, S_paths, SD = next(self.data_loader_s_iter)
        except StopIteration:
            if S is None or S_paths is None:
                self.stop_s = True
                self.data_loader_s_iter = iter(self.data_loader_s)
                S, S_paths, SD = next(self.data_loader_s_iter)
        try:
            t, t_paths = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths = next(self.data_loader_t_iter)

        if (self.stop_s and self.stop_t) or self.iter > self.max_dataset_size:
            self.stop_s = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': S, 'S_label': S_paths, 'SD_label': SD,
                    'T': t, 'T_label': t_paths}

    def __len__(self):
        return self.max_dataset_size


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):
        transform = transforms.Compose([
            transforms.Scale(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        imgs = []
        labels = []
        domain_labels = []
        for i in range(len(source)):
            imgs.append(source[i]['imgs'])
            labels.append(source[i]['labels'])
            domain_labels.append([i]*len(source[i]['labels']))
        dataset_source = Dataset2(imgs, labels,domain_labels,transform=transform)
        data_loader_s = torch.utils.data.DataLoader(dataset_source, batch_size=batch_size1, shuffle=True, num_workers=4)

        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True, num_workers=4)
        

        self.dataset_t = dataset_target
        self.paired_data = CombinedData(data_loader_s, data_loader_t, float("inf"))
        self.num_samples = min(max(len(dataset_source),len(self.dataset_t)), float("inf"))*2


    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return self.num_samples
