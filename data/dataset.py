# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np


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
class Formation(object):
    def __init__(self):
        self.formations = DataGeneration.generate_warships_formation()

        self.transform = T.Compose([T.ToTensor(), T.ToPILImage()])

        other_channels = [[255] * 60 for i in range(60)]


        temp = [list(self.formations[0])*255, other_channels, other_channels]
        # temp.reshape(10, 60, 3)
        print(temp)

        img = np.ndarray([self.formations[0], other_channels, other_channels])

        img_tensor = self.transform(img)

        img_tensor.save('test.jpg','jpeg')

if __name__ == '__main__':
    tb = Formation()