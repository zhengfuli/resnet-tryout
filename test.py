from torch import nn
from utils import tester
from network.resnet import resnet18, resnet34, resnet101
from settings import *
import sys

test = "red" if len(sys.argv) < 2 else sys.argv[1]

if test == "red":
    # Set Test parameters
    params = tester.TestParams()
    params.gpus = []  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
    # this model corresponds to the model trained in the 60th epoch shown in the two training results under ./architecture:
    # red_*_train_1e5_test_2e4_10_kinds_3min_per_epoch_resnet18.png
    params.ckpt = './models/formation_prediction.pth'

    # models
    # model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
    # model.fc = nn.Linear(512, 6)
    model = resnet18(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
    model.fc = nn.Linear(512, formation_num)

    # Test
    tester = tester.RedTester(model, params)
    tester.test()

elif test == "blue":
    pass

else:
    print("Please indicate which network you want to test.\nE.g. try to type ""python test.py red""")