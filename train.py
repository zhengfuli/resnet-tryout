import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import *
from utils import trainer
from network.resnet import resnet34, resnet101, resnet18

from settings import *
import sys

train = "red" if len(sys.argv) < 2 else sys.argv[1]

# Set Training parameters
params = trainer.TrainParams()
params.max_epoch = max_epoch
params.gpus = gpus  # set 'params.gpus=[]' to use CPU mode
params.save_dir = model_path
params.ckpt = None
params.save_freq_epoch = save_freq_epoch


if train == "red":
    # setting loss function
    params.criterion = nn.CrossEntropyLoss()

    # load data
    print("Loading dataset...")
    masks = generate_masks(train_num + test_num)

    train_data = TrainData(masks=masks[:train_num])
    val_data = TestData(masks=masks[train_num:])

    batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('train dataset len: {}'.format(len(train_dataloader.dataset)))

    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('val dataset len: {}'.format(len(val_dataloader.dataset)))

    # models
    model = resnet18(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
    model.fc = nn.Linear(512, formation_num)
    # model = resnet101(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
    # model.fc = nn.Linear(512*4, 6)

    # optimizer
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    print("Training with sgd")
    params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov=nesterov)

    # Train
    params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=10, cooldown=10, verbose=True)
    trainer = trainer.RedTrainer(model, params, train_dataloader, val_dataloader)
    trainer.train()

elif train == "blue":
    # setting loss function
    params.criterion = nn.MSELoss(reduce=True, size_average=True)

    # load data
    print("Loading dataset...")
    dataset = DataReader()

    batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)
    train_dataloader = DataLoader(dataset.get_training_set(), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('train dataset len: {}'.format(len(train_dataloader.dataset)))
    val_dataloader = DataLoader(dataset.get_testing_set(), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('val dataset len: {}'.format(len(val_dataloader.dataset)))

    # models
    model = resnet18(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
    model.fc = nn.Linear(512, 1)

    # optimizer
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    print("Training with sgd")
    params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov=nesterov)

    # Train
    params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=10, cooldown=10, verbose=True)
    trainer = trainer.BlueTrainer(model, params, train_dataloader, val_dataloader)
    trainer.train()

else:
    print("Please indicate which network you want to train.\nE.g. try to type ""python train.py red""")
