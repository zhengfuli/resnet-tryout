from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T
from data import generate_formations
import numpy as np
import time

from settings import *

class TestParams(object):
    # params based on your local env
    gpus = []  # default to use CPU mode

    # loading existing checkpoint
    ckpt = ''     # path to the ckpt file

class RedTester(object):

    TestParams = TestParams

    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        ckpt = self.params.ckpt

        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()

    def test(self):

        formations = generate_formations.generate_warships_formation()

        positions = generate_formations._transform_formation_to_position(formations[9])

        np.random.seed(int(time.time()))

        mask = sorted(np.random.choice(range(warships_num), np.random.randint(detected__warships_num[0], detected__warships_num[1]), replace=False))

        masked_positions = [positions[index] for index in mask]

        masked_formation = generate_formations._transform_position_to_formation(masked_positions)

        masked_formation = 255 - masked_formation * 255
        # print(type(masked_formation))
        transforms = T.Compose([
            T.Pad(padding=(82, 107), fill=255),
            # T.Scale(1400),
            # T.CenterCrop(224),
            # T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        img = Image.fromarray(np.uint8(masked_formation)).convert('1')
        img = transforms(img)
        img_input = Variable(torch.unsqueeze(img, 0))

        if len(self.params.gpus) > 0:
            img_input = img_input.cuda()

        output = self.model(img_input)
        # print(output)
        score = F.softmax(output, dim=1)
        # print(score)
        _, prediction = torch.max(score.data, dim=1)

        print('Prediction number: ' + str(prediction[0]))

    def _load_ckpt(self, ckpt):
        model = torch.load(ckpt)
        # print(model)
        new_dict = self.model.state_dict().copy()

        for i in range(len(model)):
            new_dict[list(self.model.state_dict().keys())[i]] = model[list(model.keys())[i]]

        self.model.load_state_dict(new_dict)
        # print(len(model))
        # print(len(list(self.model.state_dict().keys())))