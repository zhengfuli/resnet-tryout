# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np
import time

class Hand(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = root

        self._read_txt_file()
    
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if not train: 
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                    ]) 
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                    ])
                
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = self.images_root + "./images/train.txt"
        else:
            txt_file = self.images_root + "./images/test.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_root+self.images_path[index]
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)
    
    def __len__(self):
        return len(self.images_path)

import DataGeneration

def generate_masks(num):
    masks = []
    for i in range(num):
        mask = sorted(np.random.choice(range(30), 15))
        if mask in masks:
            i -= 1
        else:
            masks.append(mask)


class TrainData(object):
    def __init__(self, masks, transforms=None):
        np.random.seed(int(time.time()))
        self.masks = masks

        self.formations = DataGeneration.generate_warships_formation()

        self._generate_data()

        if not transforms:
            self.transforms = T.Compose([
                T.Scale(256),
                T.RandomSizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def _generate_data(self):
        self.train_data = {}

        for i in range(len(self.formations)):
            positions = DataGeneration._transform_formation_to_position(self.formations[i])

            for mask in self.masks[i*5000:(i+1)*5000]:
                masked_positions = positions
                for index in mask:
                    masked_positions.remove(masked_positions[index])
                masked_formation = DataGeneration._transform_position_to_formation(masked_positions)
                assert(masked_formation not in self.train_data)
                self.train_data[masked_formation] = i

        self.data = np.random.shuffle(list(self.train_data.items()))

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        formation = 255 - self.data[index][0] * 254

        img = Image.fromarray(formation.astype('uint8')).convert('RGB')

        # img.save('test.jpg', 'jpeg')

        data = self.transforms(img)
        return data, self.data[index][1]

class TestData(object):
    def __init__(self, masks, transforms=None):
        np.random.seed(int(time.time()))
        self.masks = masks

        self.formations = DataGeneration.generate_warships_formation()

        self._generate_data()

        if not transforms:
            self.transforms = T.Compose([
                T.Scale(224),
                T.CenterCrop(224),
                T.ToTensor(),
                ])

    def _generate_data(self):
        self.test_data = {}

        for i in range(len(self.formations)):
            positions = DataGeneration._transform_formation_to_position(self.formations[i])

            for mask in self.masks[i*500:(i+1)*500]:
                masked_positions = positions
                for index in mask:
                    masked_positions.remove(masked_positions[index])
                masked_formation = DataGeneration._transform_position_to_formation(masked_positions)
                assert(masked_formation not in self.test_data)
                self.test_data[masked_formation] = i

        self.data = np.random.shuffle(list(self.test_data.items()))

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        formation = 255 - self.data[index][0] * 254

        img = Image.fromarray(formation.astype('uint8')).convert('RGB')

        # img.save('test.jpg', 'jpeg')

        data = self.transforms(img)
        return data, self.data[index][1]
