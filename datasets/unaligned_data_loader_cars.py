import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_cars import Dataset
import numpy as np
from PIL import Image, ImageOps


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CombinedDataLOL(Dataset):
    def __init__(self, data_loader_S, data_loader_t, max_dataset_size):
        super(Dataset, self).__init__()
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
        S, S_paths, t, t_paths = None, None, None, None
        i = index % self.num_datasets
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

        if (np.prod(self.stop_S) > 0 and self.stop_t == True) or self.iter > self.max_dataset_size:
            for i in range(len(self.stop_S)):
                self.stop_S[i] = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': S, 'S_label': S_paths,
                    'T': t, 'T_label': t_paths}

    def __len__(self):
        return self.max_dataset_size * self.num_datasets

class CombinedData(Dataset):
    def __init__(self, data_loader_s, data_loader_t, max_dataset_size):
        super(Dataset, self).__init__()
        self.data_loader_s = data_loader_s
        self.data_loader_t = data_loader_t

        self.stop_s = False
        self.stop_t = False

        self.max_dataset_size = max_dataset_size

        self.data_loader_s_iter = iter(self.data_loader_s)
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0

    def __getitem__(self, index):
        S, S_paths, t, t_paths = None, None, None, None
        try:
            S, S_paths = next(self.data_loader_s_iter)
        except StopIteration:
            if S is None or S_paths is None:
                self.stop_s = True
                self.data_loader_s_iter = iter(self.data_loader_s)
                S, S_paths = next(self.data_loader_s_iter)
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
            return {'S': S, 'S_label': S_paths,
                    'T': t, 'T_label': t_paths}

    def __len__(self):
        return self.max_dataset_size * self.num_datasets


class UnalignedDataLoader():
    # Returns paired dataloader for source and target domain images and labels. 
    def __init__(self):

        self.__imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }

    def initialize(self, source, target, batch_size1, batch_size2, scale=32, split='Train'):
        if split == 'Train':
            transform_source = transforms.Compose([
                # transforms.Resize(scale),
                transforms.RandomCrop(224),
                transforms.Resize((scale, scale)),
                # transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                Lighting(0.1, self.__imagenet_pca['eigval'], self.__imagenet_pca['eigvec']),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transform_target = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.Resize((scale, scale)),
                # transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                Lighting(0.1, self.__imagenet_pca['eigval'], self.__imagenet_pca['eigvec']),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif split == 'Test':
            transform_source = transforms.Compose([
                # transforms.Resize(scale),
                # transforms.RandomCrop(scale),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transform_target = transforms.Compose([
                # transforms.Resize((scale,scale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise Exception('Wrong split')
        data_sources = []
        data_loader_s = []
        max_size = 0
        for i in range(len(source)):
            data_sources.append(Dataset(source[i]['imgs'], source[i]['labels'], transform=transform_source))
            data_loader_s.append(
                torch.utils.data.DataLoader(data_sources[i], batch_size=batch_size1, shuffle=True, num_workers=4,
                                            pin_memory=True))
            max_size = max(max_size, len(data_sources[i]))
        self.dataset_s = data_loader_s

        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform_target)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True, num_workers=4,
                                                    pin_memory=True)

        self.dataset_t = dataset_target
        self.paired_data = CombinedData(data_loader_s, data_loader_t,
                                        float("inf"))

        self.num_datasets = len(source)
        self.num_samples = min(max(max_size, len(self.dataset_t)), float("inf")) * self.num_datasets

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return self.num_samples
