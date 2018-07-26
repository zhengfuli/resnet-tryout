# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np
import time
from data import generate_formations
from tqdm import tqdm, trange
from settings import *

def generate_masks(num):
    print("Generating Masks...", end="")
    masks = []
    process_bar = tqdm()

    while len(masks) < num:
        mask = sorted(np.random.choice(range(warships_num), np.random.randint(detected__warships_num[0], detected__warships_num[1]), False))
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

        self.formations = generate_formations.generate_warships_formation()

        self._generate_data()

        if (224 - ocean_grid[0]) % 2 != 0 or (224 - ocean_grid[1]) % 2 != 0:
            padding = (int((224 - ocean_grid[1]) / 2), 224 - int((224 - ocean_grid[1]) / 2) - ocean_grid[0],
                       int((224 - ocean_grid[0]) / 2), 224 - int((224 - ocean_grid[0]) / 2) - ocean_grid[0])
        else:
            padding = (int((224 - ocean_grid[1]) / 2), int((224 - ocean_grid[0]) / 2))

        if not transforms:
            self.transforms = T.Compose([
                T.Pad(padding, fill=255),
                # T.Scale(1400),
                # T.CenterCrop(224),
                T.ToTensor()
                ])

    def _generate_data(self):
        self.train_data = []
        self.mappings = []
        samples_num = int(train_num/formation_num)

        for i in range(len(self.formations)):
            positions = generate_formations._transform_formation_to_position(self.formations[i])
            mask = self.masks[i*samples_num:(i+1)*samples_num]

            for j in range(len(mask)):
                masked_positions = [positions[index] for index in mask[j]]
                self.train_data.append(masked_positions)
                self.mappings.append([j+i*samples_num, i])

        np.random.shuffle(self.mappings)

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        masked_formation = generate_formations._transform_position_to_formation(self.train_data[self.mappings[index][0]])

        masked_formation = 255 - masked_formation * 255

        data = Image.fromarray(masked_formation.astype('uint8')).convert('1')

        data = self.transforms(data)

        return data, self.mappings[index][1]

    def __len__(self):
        return len(self.train_data)

class TestData(object):
    def __init__(self, masks, transforms=None):
        np.random.seed(int(time.time()))

        self.masks = masks

        self.formations = generate_formations.generate_warships_formation()

        self._generate_data()

        if (224 - ocean_grid[0]) % 2 != 0 or (224 - ocean_grid[1]) % 2 != 0:
            padding = (int((224 - ocean_grid[1]) / 2), 224 - int((224 - ocean_grid[1]) / 2) - ocean_grid[0],
                       int((224 - ocean_grid[0]) / 2), 224 - int((224 - ocean_grid[0]) / 2) - ocean_grid[0])
        else:
            padding = (int((224 - ocean_grid[1]) / 2), int((224 - ocean_grid[0]) / 2))

        if not transforms:
            self.transforms = T.Compose([
                T.Pad(padding, fill=255),
                # T.Scale(1400),
                # T.CenterCrop(224),
                T.ToTensor()
                ])

    def _generate_data(self):
        self.test_data = []
        self.mappings = []
        samples_num = int(test_num / formation_num)

        for i in range(len(self.formations)):
            positions = generate_formations._transform_formation_to_position(self.formations[i])

            mask = self.masks[i*samples_num:(i+1)*samples_num]
            for j in range(len(mask)):
                # print(len(positions))
                masked_positions = [positions[index] for index in mask[j]]
                self.test_data.append(masked_positions)
                self.mappings.append([j+i*samples_num, i])

        np.random.shuffle(self.mappings)

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        masked_formation = generate_formations._transform_position_to_formation(self.test_data[self.mappings[index][0]])

        masked_formation = 255 - masked_formation * 255

        data = Image.fromarray(masked_formation.astype('uint8')).convert('1')

        data = self.transforms(data)

        return data, self.mappings[index][1]

    def __len__(self):
        return len(self.test_data)

# data objects that will be returned to torch.DataLoader
class Data(object):
    def __init__(self, dataset):
        np.random.seed(int(time.time()))
        self.dataset = dataset
        np.random.shuffle(self.dataset)

    def __getitem__(self, item):
        # 2018/07/26 at this time the training data for the red side is in the form of
        # array((10,4)) + array((10,60))
        # it is needed to convert above two, together with the location of cities, to one sample and then resize
        # if it's necessary, you may modify the code here if any modification of the training data for the blue side's network happens

        city_position = [[0] * city_grid[1] for i in range(city_grid[0])]
        for pos in CITY_POSITION:
            city_position[pos[1]][pos[0] - 1 - ocean_grid[1]] = 1

        data = np.hstack((self.dataset[item][0], self.dataset[item][1], np.array(city_position)))

        data = 255

# reading data for the network of the blue side adn return Data() objects
class DataReader(object):
    def __init__(self, type):
        self.file_incomplete_data = open("./data/sample_data0723.npy", "rb")
        self.total_data = []

        for i in trange(data_amount):
            data = np.load(self.file_incomplete_data)
            self.total_data.append(data)
        self.file_incomplete_data.close()

    def get_training_set(self):
        return Data(self.total_data[:train_amount])

    def get_testing_set(self):
        return Data(self.total_data[train_amount:])

if __name__ == '__main__':
    # test training data for the red side
    formations = generate_formations.generate_warships_formation()

    positions = generate_formations._transform_formation_to_position(formations[9])

    np.random.seed(int(time.time()))

    mask = sorted(np.random.choice(range(30),20,replace=False))

    masked_positions = [positions[index] for index in mask]

    masked_formation = generate_formations._transform_position_to_formation(masked_positions)

    masked_formation = 255 - masked_formation * 255

    if (224 - ocean_grid[0]) % 2 != 0 or (224 - ocean_grid[1]) % 2 != 0:
        padding = (int((224 - ocean_grid[1]) / 2), 224 - int((224 - ocean_grid[1]) / 2) - ocean_grid[0],
                   int((224 - ocean_grid[0]) / 2), 224 - int((224 - ocean_grid[0]) / 2) - ocean_grid[0])
    else:
        padding = (int((224 - ocean_grid[1]) / 2), int((224 - ocean_grid[0]) / 2))

    # print(type(masked_formation))
    transforms = T.Compose([
        T.Pad(padding, fill=255),
        # T.Scale(1400),
        # T.CenterCrop(224)
        # T.RandomHorizontalFlip(),
        T.ToTensor()
        ])

    # print(masked_formation)
    img = Image.fromarray(masked_formation.astype('uint8')).convert('1')
    # img.show()
    img = transforms(img)
    print(img.size())
    # img.show()
    # img.save('partial_formation.png', 'png')