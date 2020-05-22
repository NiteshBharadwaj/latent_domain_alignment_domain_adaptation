from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2

class Dataset(data.Dataset):
    # Dataloader for office
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, data, label, batch_size,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.n_datas = len(self.data)
        self.lens = [len(x) for x in self.data]
        self.rndseqs = []
        self.batch_size = batch_size
        self.batch_it = [0 for x in self.data]
        for i in range(self.n_datas):
            self.rndseqs.append(np.random.permutation(self.lens[i]))
        self.hashmap = {}

    def _get_single_item(self, index):
        img_path, target = self.data[index], self.labels[index]
        if index in self.hashmap:
            img = self.hashmap[index]
        else:
            img = Image.open(img_path)
            img = np.array(img)
            img = img[..., :3]
            img = Image.fromarray(img)
            self.hashmap[index] = img

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
            # return img, target
        return img, target * 1.0

    def __getitem__(self, index):
        ## Maintains it's own shuffle, outside shuffle is not taken into account
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        class_id = np.random.randint(self.n_datas)
        n_samples = 0
        start_idx = self.batch_it[class_id]
        end_idx = max(start_idx + self.batch_size, self.lens[class_id])
        if end_idx in [start_idx+1, start_idx+2, start_idx+3]:
            # Need to reset rand seq. Don't tolerate too small batches
            self.rndseqs[class_id] = np.random.permutation(self.lens[class_id])
            start_idx = 0
            end_idx = max(start_idx + self.batch_size, self.lens[class_id])

        final_images = np.zeros((self.batch_size,3,256,256))
        final_labels = np.zeros((self.batch_size))
        for img_idx in range(start_idx,end_idx):
            rnd_idx =self.rndseqs[class_id][img_idx]
            final_images[img_idx], final_labels[img_idx] = self._get_single_item(rnd_idx)
            n_samples+=1
            self.batch_it[class_id] +=1
        final_images = final_images[:n_samples]
        final_labels = final_labels[:n_samples]

        if end_idx == self.lens[class_id]:
            self.rndseqs[class_id] = np.random.permutation(self.lens[class_id])
            self.batch_it[class_id] = 0
        return final_images, final_labels

    def __len__(self):
        return float('inf')
