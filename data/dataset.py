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
from tqdm import tqdm, trange

def generate_masks(num):
    print("Generating Masks...", end="")
    masks = []
    process_bar = tqdm()

    while len(masks) < num:
        mask = sorted(np.random.choice(range(30), 20, False))
        if mask in masks:
            pass
        else:
            masks.append(mask)
            process_bar.update(1)

    # process_bar.close()
    print(len(masks), "Done.")
    return masks

class TrainData(object):
    def __init__(self, masks, transforms=None):
        np.random.seed(int(time.time()))
        self.masks = masks

        self.formations = DataGeneration.generate_warships_formation()

        self._generate_data()

        if not transforms:
            self.transforms = T.Compose([
                T.Pad(padding=107, fill=(255, 255, 255)),
                T.Scale(1400),
                T.CenterCrop(224),
                T.ToTensor(),
                ])

    def _generate_data(self):
        self.train_data = []
        self.mappings = []

        # print(len(self.formations))
        for i in range(len(self.formations)):
            positions = DataGeneration._transform_formation_to_position(self.formations[i])

            mask = self.masks[i*500:(i+1)*500]
            for j in range(len(mask)):
                # print(len(positions))
                masked_positions = [positions[index] for index in mask[j]]
                self.train_data.append(masked_positions)
                self.mappings.append([j+i*500, i])

        np.random.shuffle(self.mappings)

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        masked_formation = DataGeneration._transform_position_to_formation(self.train_data[self.mappings[index][0]])

        masked_formation = 255 - masked_formation * 255

        data = Image.fromarray(masked_formation.astype('uint8')).convert('RGB')

        # img.save('test.jpg', 'jpeg')

        data = self.transforms(data)

        return data, self.mappings[index][1]

    def __len__(self):
        return len(self.train_data)

class TestData(object):
    def __init__(self, masks, transforms=None):
        np.random.seed(int(time.time()))
        self.masks = masks

        self.formations = DataGeneration.generate_warships_formation()

        self._generate_data()

        if not transforms:
            self.transforms = T.Compose([
                T.Pad(padding=107, fill=(255, 255, 255)),
                T.Scale(1400),
                T.CenterCrop(224),
                T.ToTensor(),
                ])

    def _generate_data(self):
        self.test_data = []
        self.mappings = []

        # print(len(self.formations))
        for i in range(len(self.formations)):
            positions = DataGeneration._transform_formation_to_position(self.formations[i])

            mask = self.masks[i*50:(i+1)*50]
            for j in range(len(mask)):
                # print(len(positions))
                masked_positions = [positions[index] for index in mask[j]]
                self.test_data.append(masked_positions)
                self.mappings.append([j+i*50, i])

        np.random.shuffle(self.mappings)

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        masked_formation = DataGeneration._transform_position_to_formation(self.test_data[self.mappings[index][0]])

        masked_formation = 255 - masked_formation * 255

        data = Image.fromarray(masked_formation.astype('uint8')).convert('RGB')

        # img.save('test.jpg', 'jpeg')

        data = self.transforms(data)

        return data, self.mappings[index][1]

    def __len__(self):
        return len(self.test_data)

if __name__ == '__main__':
    formations = DataGeneration.generate_warships_formation()

    positions = DataGeneration._transform_formation_to_position(formations[9])

    np.random.seed(int(time.time()))

    mask = sorted(np.random.choice(range(30),20,replace=False))

    masked_positions = [positions[index] for index in mask]

    masked_formation = DataGeneration._transform_position_to_formation(masked_positions)

    masked_formation = 255 - masked_formation * 255
    # print(type(masked_formation))
    transforms = T.Compose([
        T.Pad(padding=107, fill=(255,255,255)),
        T.Scale(1400),
        T.CenterCrop(224)
        # T.RandomHorizontalFlip(),
        # T.ToTensor(),
        ])

    img = Image.fromarray(np.uint8(masked_formation)).convert('RGB')
    img = transforms(img)
    img.save('test.jpg', 'jpeg')