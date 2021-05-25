import torch
import torch.distributed as dist
import torch.nn as nn

import ceddl
from ceddl import log
import ceddl.experiment_utils as utils

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


parser = utils.get_parser()
args = utils.parse_args(parser)
utils.init(args)
# log.set_allowed_ranks(list(range(args.world_size)))

train_loader, val_loader = utils.load_mnist(args.rank, args.world_size, args.batch_size)

model = Net().to(args.device)

args.use_ref_module = True
args.tracking_loader = train_loader
args.criterion = nn.CrossEntropyLoss()
model, optimizer = utils.wrap_model(model, args)
optimizer = ceddl.optim.NetworkSVRG(model, lr=args.lr)

# optimizer = torch.optim.SGD(model.parameters(),
                                # lr=args.lr,
                                # weight_decay=args.weight_decay,
                                # momentum=args.momentum)

log.info('Model is on %s', next(model.parameters()).device)


classes = [int(i) for i in range(10)]
criterion = nn.CrossEntropyLoss()

train_res, val_res = utils.train(model, criterion, optimizer, train_loader, args,
                                 exp_name='mnist', val_loader=val_loader, classes=classes)

log.info('Process %d exited', args.rank)
