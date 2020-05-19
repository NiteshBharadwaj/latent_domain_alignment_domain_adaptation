import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
# from datasets_office import Dataset
from PIL import Image, ImageOps
from datasets_cars import Dataset

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
    def initialize(self, source, batch_size, scale=32):
        transform = transforms.Compose([
            #transforms.Resize(scale),
            #transforms.RandomCrop(scale),
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(0.4,0.2,0.2),
            transforms.ToTensor(),
            #Lighting(0.1, self.__imagenet_pca['eigval'],self.__imagenet_pca['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        dataset_source = []
        dataloader_source = []

        self.max_len = 0
        for i in range(31):
            allImages = []
            allLabels = []
            for j in range(len(source)):
                
            
                imgs = source[j]['imgs']
                labels = source[j]['labels']
                indices = [i for i, x in enumerate(labels) if x == i]
                allImages += [imgs[index] for index in indices]
                allLabels += [labels[index] for index in indices]
            dataset_source.append(Dataset(allImages, allLabels, transform=transform))
            self.max_len = max(self.max_len, len(dataset_source))
            dataloader_source.append(
                torch.utils.data.DataLoader(dataset_source[i], batch_size=batch_size, shuffle=True,num_workers=2))

        self.dataset_s = dataset_source

        self.paired_data = ClasswiseData(dataloader_source, float("inf"))

    def name(self):
        return 'ClasswiseDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return float("inf")
