# Cross-checked with the results from https://github.com/kuangliu/pytorch-cifar
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import ceddl
from ceddl import log
import ceddl.experiment_utils as utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    args = utils.parse_args()
    # log.set_allowed_ranks(list(range(args.world_size)))
    utils.init(args)

    model = Net().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model, optimizer = utils.wrap_model(model, optimizer, args)
    log.info('Model is on %s', next(model.parameters()).device)

    train_loader, val_loader = utils.load_cifar10(args.rank, args.world_size, args.batch_size, 
                                            shuffle=not args.deterministic, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    criterion = nn.CrossEntropyLoss()
    train_res, val_res = utils.train(model, criterion, optimizer, train_loader, args,
                                     exp_name='cifar10', val_loader=val_loader, classes=classes)

    log.info('Process %d exited', args.rank)
