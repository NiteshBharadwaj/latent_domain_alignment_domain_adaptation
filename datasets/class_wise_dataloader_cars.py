import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
# from datasets_office import Dataset
from PIL import Image, ImageOps
from datasets_cars_classwise import Dataset
from unaligned_data_loader_cars import Lighting

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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

        try:
            source, source_paths = next(self.data_loader_s_iter[0])
        except StopIteration:
            if source is None or source_paths is None:
                self.stop_source[0] = True
                self.data_loader_s_iter[0] = iter(self.data_loader_s[0])
                source, source_paths = next(self.data_loader_s_iter[0])

        self.iter += 1
        return {'S': source, 'S_label': source_paths,
                'T': source, 'T_label': source_paths}


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class ClasswiseDataLoader():
    def __init__(self):

        self.__imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
    def initialize(self, source, batch_size, num_workers_, scale):
        transform = transforms.Compose([
            # transforms.Resize(scale),
            # transforms.RandomCrop(scale),
            # ResizeImage(256),
            transforms.RandomCrop(scale),
            #transforms.Resize((scale, scale)),
            #transforms.Scale(scale),
            #transforms.RandomResizedCrop(scale2),

            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4),
            transforms.ToTensor(),
            # Lighting(0.1, self.__imagenet_pca['eigval'],self.__imagenet_pca['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset_source = []
        dataloader_source = []
        overall_images = []
        overall_labels = []
        self.max_len = 0
        for i in range(75):
            allImages = []
            allLabels = []
            for j in range(len(source)):
                imgs = source[j]['imgs']
                labels = source[j]['labels']
                indices = [k for k, x in enumerate(labels) if x[0] == i]
                if len(indices)==0:
                    continue
                allImages += [imgs[index] for index in indices]
                allLabels += [labels[index] for index in indices]
            if len(allImages)==0:
                continue
            overall_images.append(allImages)
            overall_labels.append(allLabels)
        dataset_source.append(Dataset(overall_images, overall_labels, batch_size, transform=transform))
        self.max_len = max(self.max_len, len(dataset_source[0]))
        dataloader_source.append(
            torch.utils.data.DataLoader(dataset_source[0], batch_size=1, shuffle=True, num_workers=num_workers_,
                                        worker_init_fn=worker_init_fn, pin_memory=True))
        self.dataset_s = dataset_source
        self.paired_data = ClasswiseData(dataloader_source, float("inf"))

    def name(self):
        return 'ClasswiseDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return float("inf")
