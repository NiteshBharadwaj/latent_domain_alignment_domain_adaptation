from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
from datasets_cars import transform

#class Sampler(d)

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
    def __init__(self, data, label, batch_size,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.n_datas = len(self.data)
        self.batch_size = batch_size
        self.act_lens = [len(x) for x in self.data]
        self.blens = []
        self.blen_end = [0]
        for i in range(self.n_datas):
            self.blen_end.append(len(self.data[i]) // self.batch_size + self.blen_end[-1] + 1)
            self.blens.append(len(self.data[i]) // self.batch_size + 1)
            last_batch_size = len(self.data[i]) % self.batch_size
            if  last_batch_size<=2 and last_batch_size>0:
                self.blen_end[-1] -=1
                self.blens[-1] -=1
        self.blen_end = self.blen_end[1:]
        self.total_len = self.blen_end[-1]
        self.batch_it = [0 for x in self.data]
        self.hashmap = {}
        self.rndseqs = []
        for i in range(self.n_datas):
            self.rndseqs.append(np.random.permutation(self.act_lens[i]))
            self.hashmap[i] = {}

    def _get_single_item(self, index, class_index):
        img_path, target = self.data[class_index][index], self.labels[class_index][index]
        if index in self.hashmap[class_index]:
            img = self.hashmap[class_index][index]
        else:
            img = Image.open(img_path)
            img = np.array(img)
            img = img[..., :3]
            img = transform(img)
            img = Image.fromarray(img)
            #self.hashmap[class_index][index] = img
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)

        return img, 1.0

    def __getitem__(self, index):
        ## Maintains it's own shuffle, outside shuffle is not taken into account
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        class_idx = -1000
        for i in range(self.n_datas):
            if index < self.blen_end[i]:
                class_idx = i
                break
        assert class_idx != -1000
        index = index%self.blens[class_idx]
        start_idx = index*self.batch_size
        end_idx = min(start_idx + self.batch_size, self.act_lens[class_idx])
        if end_idx <= start_idx:
            self.rndseqs[class_idx] = np.random.permutation(self.act_lens[class_idx])
            self.batch_it[class_idx] = 0
            start_idx = 0
            end_idx = min(start_idx + self.batch_size, self.act_lens[class_idx])
        final_images = np.zeros((self.batch_size,3,256,256),dtype='float32')
        final_labels = np.zeros((self.batch_size), dtype='float32')
        n_samples = 0
        for img_idx in range(start_idx,end_idx):
            rnd_idx =self.rndseqs[class_idx][img_idx]
            final_images[img_idx - start_idx], final_labels[img_idx - start_idx] = self._get_single_item(rnd_idx, class_idx)
            n_samples+=1
            self.batch_it[class_idx] +=1
        final_images = final_images[:n_samples]
        final_labels = final_labels[:n_samples]

        self.batch_it[class_idx] += 1
        if self.batch_it[class_idx] >=self.blens[class_idx]:
            self.rndseqs[class_idx] = np.random.permutation(self.act_lens[class_idx])
            self.batch_it[class_idx] = 0
        return final_images, final_labels

    def __len__(self):
        return self.total_len
