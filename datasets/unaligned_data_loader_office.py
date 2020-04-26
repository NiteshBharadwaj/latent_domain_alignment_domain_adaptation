import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_cars import Dataset
import numpy as np
from PIL import Image, ImageOps

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h_off = random.randint(0, img.shape[1]-self.size)
        w_off = random.randint(0, img.shape[2]-self.size)
        img = img[:, h_off:h_off+self.size, w_off:w_off+self.size]
        return img


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
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


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
        if (np.prod(self.stop_S)>0 and self.stop_t==True) or self.iter > self.max_dataset_size:
            for i in range(len(self.stop_S)):
                self.stop_S[i] = False
            self.stop_t = False
            self.iter=0
            raise StopIteration()
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
                

        self.iter += 1 
        return {'S': S, 'S_label': S_paths,
                'T': t, 'T_label': t_paths}

    def __len__(self):
        return self.max_dataset_size*self.num_datasets


class UnalignedDataLoader():
    def __init__(self):

        self.__imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])
        }
    def initialize(self, source, target, batch_size1, batch_size2, scale=32, split='Train'):
        start_first = 0
        start_center = (256 - 224 - 1) / 2
        start_last = 256 - 224 - 1
            
        if split=='Train':
            transform_source = transforms.Compose([
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
            transform_target = transforms.Compose([ 
                #transforms.Resize((scale,scale)),
                ResizeImage(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(0.4,0.2,0.2),
                transforms.ToTensor(),
                #Lighting(0.1,self.__imagenet_pca['eigval'],self.__imagenet_pca['eigvec']),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        elif split=='Test':
            transform_source = transforms.Compose([
                #transforms.Resize(scale),
                #transforms.RandomCrop(scale),
                ResizeImage(256),
                PlaceCrop(224, start_center, start_center),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
            transform_target = transforms.Compose([ 
                #transforms.Resize((scale,scale)),
                ResizeImage(256),
                PlaceCrop(224, start_center, start_center),
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
            data_loader_s.append(torch.utils.data.DataLoader(data_sources[i], batch_size=batch_size1, shuffle=(split=='Train'), num_workers=3))
            max_size = max(max_size,len(data_sources[i]))
        self.dataset_s = data_loader_s

        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform_target)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=(split=='Train'), num_workers=3)

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