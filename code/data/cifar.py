from PIL import Image
import os
import os.path
import numpy as np
import sys
from torchvision import transforms
import pickle

from .vision import VisionDataset

class CIFAR10(VisionDataset):
    root = "../dataset/cifar-10-batches-py"
    train_list = [ 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    test_list = ['test_batch']
    
    def __init__(self, train=True, transform=None, contrastive_learning=False):
        super(CIFAR10, self).__init__(transform=transform) #target_transform)
        self.train = train  # training set or test set
        self.learning_type = contrastive_learning
        self.data = []
        self.targets = []
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        for file_name in data_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        ori_img = img
        toTensor = transforms.ToTensor()
        ori_img = toTensor(ori_img)
        
        if self.learning_type=='contrastive':
            img_2 = img.copy()

        elif self.learning_type=='linear_eval':
            if self.train:
                img_2 = img.copy()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.learning_type=='contrastive':
            img_2   = Image.fromarray(img_2)
        elif self.learning_type=='linear_eval':
            if self.train:
                img_2 = img.copy()

        if self.transform is not None:
            img  = self.transform(img)
            if self.learning_type=='contrastive':
                img_2   = self.transform(img_2)
            elif self.learning_type=='linear_eval':
                if self.train:
                    img_2 = self.transform(img_2)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if self.learning_type=='contrastive':
            return ori_img, img, img_2, target
        elif self.learning_type=='linear_eval':
            if self.train:
                return ori_img, img, img_2, target
            else:
                return img, target
        else:
            return img, target

    def __len__(self):
        return len(self.data)
        
    
    