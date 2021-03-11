import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets_pacs import Dataset
import numpy as np
from PIL import Image, ImageOps


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def balance_classes(labels, domain_labels, balance_domains=False):
    num_labels = 31
    num_domains = 3
    class_counts = [0]*num_labels
    class_domain_counts = []
    for i in range(num_labels):
        class_domain_counts.append([0]*num_domains)
    for idx,label in enumerate(labels):
        class_counts[label]+=1
        class_domain_counts[label][domain_labels[idx]]+=1
    print(class_counts, class_domain_counts)
    N = float(sum(class_counts))
    class_weights = [0]*len(labels)
    class_domain_weights = [0]*len(labels)
    for idx, label in enumerate(labels):
        class_weights[idx] = N/float(class_counts[label])
        class_domain_weights[idx] = N/float(class_domain_counts[label][domain_labels[idx]])

    if balance_domains:
        return class_domain_weights
    else:
        return class_weights

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
        h_off = random.randint(0, img.shape[1] - self.size)
        w_off = random.randint(0, img.shape[2] - self.size)
        img = img[:, h_off:h_off + self.size, w_off:w_off + self.size]
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
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


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
        S, S_paths, SD_label, t, t_paths, td_label = None, None, None, None, None, None
        S_idx, T_idx = None,None
        try:
            S, S_paths, SD_label, S_idx = next(self.data_loader_s_iter)
        except StopIteration:
            if S is None or S_paths is None:
                self.stop_s = True
                self.data_loader_s_iter = iter(self.data_loader_s)
                S, S_paths, SD_label, S_idx = next(self.data_loader_s_iter)
        try:
            t, t_paths, td_label, T_idx = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t_paths is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t_paths, td_label, T_idx = next(self.data_loader_t_iter)

        if (self.stop_s and self.stop_t) or self.iter > self.max_dataset_size:
            self.stop_s = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': S, 'S_label': S_paths, 'SD_label': SD_label, 'S_idx': S_idx,
                    'T': t, 'T_label': t_paths, 'td_label': td_label, 'T_idx': T_idx}

    def __len__(self):
        return self.max_dataset_size


class UnalignedDataLoader():
    def __init__(self):

        self.__imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }

    def initialize(self, source, target, batch_size1, batch_size2, num_workers_, scale=256, split='Train'):
        start_first = 0
        start_center = (256 - 224 - 1) / 2
        start_last = 256 - 224 - 1
        scale = 256
        scale2 = 227
        if split == 'Train':
            transform_source = transforms.Compose([
                transforms.Resize(scale),
                transforms.RandomResizedCrop(scale2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transform_target = transforms.Compose([
                transforms.Resize(scale),
                transforms.RandomResizedCrop(scale2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif split == 'Test':
            transform_source = transforms.Compose([
                transforms.Resize(scale),
                PlaceCrop(224, start_center, start_center),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transform_target = transforms.Compose([
                transforms.Resize(scale),
                PlaceCrop(224, start_center, start_center),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise Exception('Wrong split')
        target_imgs = []
        target_labels = []
        imgs = []
        labels = []
        domain_labels = []
        for j in range(1):
            for i in range(len(source)):
                imgs += source[i]['imgs']
                labels += source[i]['labels']
                domain_labels+=len(source[i]['labels'])*[i]
            target_imgs += target['imgs']
            target_labels += target['labels']

        dataset_source = Dataset(imgs, labels, domain_labels, transform=transform_source)
        sampling_weights = torch.DoubleTensor(balance_classes(labels, domain_labels, balance_domains=False))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sampling_weights, len(sampling_weights))
        data_loader_s = torch.utils.data.DataLoader(dataset_source, batch_size=batch_size1,sampler=sampler,
                                                    num_workers=num_workers_, worker_init_fn=worker_init_fn,
                                                    pin_memory=True)


        dataset_target = Dataset(target_imgs, target_labels, [len(source)]*len(target_labels), transform=transform_target)
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True,
                                                    num_workers=num_workers_, worker_init_fn=worker_init_fn,
                                                    pin_memory=True)

        self.dataset_t = dataset_target
        self.paired_data = CombinedData(data_loader_s, data_loader_t, float("inf"))

        self.num_samples = min(max(len(dataset_source), len(self.dataset_t)), float("inf")) * 2

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return self.num_samples
