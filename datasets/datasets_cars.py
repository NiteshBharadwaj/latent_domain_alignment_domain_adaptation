from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2

def resize2SquareKeepingAspectRation(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

def transform(img):
    return resize2SquareKeepingAspectRation(img, 256, cv2.INTER_LINEAR)


class Dataset(data.Dataset):
    # Dataloader for cars
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, data, label,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """

        img_path, target = self.data[index], self.labels[index]
        img = Image.open(img_path)
        img = np.array(img)
        img = img[...,:3]
        img = transform(img) 
        img = Image.fromarray(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
            # return img, target
        return img, target*1.0
    def __len__(self):
        return len(self.data)
