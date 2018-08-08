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
from tqdm import trange
from matplotlib import pyplot as plt
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
        self.formations = generate_formations.generate_warships_formation()

    def test(self, num, visualize=False):
        accuracy = 0
        np.random.seed(int(time.time()))
        images = []
        n = 10 if visualize else num

        for i in trange(n):
            c = np.random.randint(10) if not visualize else i
            positions = generate_formations._transform_formation_to_position(self.formations[c])

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
            # img.save("test_red.png", "png")
            img_input = transforms(img)

            img_input = Variable(torch.unsqueeze(img_input, 0))

            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            # print(output)
            score = F.softmax(output, dim=1)
            # print(score)
            _, prediction = torch.max(score.data, dim=1)

            if prediction[0] == c:
                accuracy += 1

            if visualize:
                transforms = T.Compose([T.Pad(padding=(0, 25), fill=255)])
                image = transforms(img)
                ground_truth = transforms(Image.fromarray(np.uint8(255 - 255 * self.formations[c])).convert('1'))
                images.append([ground_truth, "Ground Truth: " + str(c)])
                images.append([image, "Prediction: " + str(int(prediction[0]))])

        print(round(accuracy / n, 4))

        if visualize:
            plt.figure("Formation Prediction")
            for i in range(1, len(images) + 1):
                plt.subplot(2, 10, i)
                plt.title(images[i - 1][1])
                plt.imshow(images[i - 1][0])
                # plt.axis('off')
            plt.show()

    def _load_ckpt(self, ckpt):
        model = torch.load(ckpt)
        # print(model)
        new_dict = self.model.state_dict().copy()

        for i in range(len(model)):
            new_dict[list(self.model.state_dict().keys())[i]] = model[list(model.keys())[i]]

        self.model.load_state_dict(new_dict)
        # print(len(model))
        # print(len(list(self.model.state_dict().keys())))