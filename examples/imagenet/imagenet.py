# https://tiny-imagenet.herokuapp.com/
# https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/ResNet18_64.ipynb

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

import ceddl
from ceddl import log
import ceddl.experiment_utils as utils



class WarmStartStepLR(StepLR):
    def __init__(self, *args, warm_up_epochs=0, **kwargs):
        self.warm_up_epochs = warm_up_epochs
        super().__init__(*args, **kwargs)
        # del self._get_closed_form_lr

    def get_lr(self):
        if self.last_epoch <= self.warm_up_epochs:
            warm_up_gamma = self.gamma ** (self.warm_up_epochs // self.step_size)
            return [max(base_lr * warm_up_gamma * self.last_epoch / self.warm_up_epochs, 1e-6)
                    for base_lr in self.base_lrs]
        else:
            return super().get_lr()

parser = utils.get_parser()
parser.add_argument('--pretrained', default=False, action='store_true')
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--optimizer', default='sgd', type=str, help='Which optimizer to use', choices=['sgd', 'lamb', 'adam'])
parser.add_argument('--lamb_adam', default=False, action='store_true', help='Use Adam or not')

args = utils.parse_args(parser)
utils.init(args)
# log.set_allowed_ranks(list(range(args.world_size)))

train_loader, val_loader = utils.load_imagenet(args.rank, args.world_size, args.batch_size, num_workers=args.num_workers)

classes = range(1000)
criterion = nn.CrossEntropyLoss()

model = getattr(models, args.model)(pretrained=args.pretrained).to(args.device)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

elif args.optimizer == 'lamb':
    optimizer = ceddl.optim.Lamb(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 betas=(.9, .999),
                                 adam=args.lamb_adam)
elif args.optimizer == 'adam':
    pass
else:
    log.fatal('Optimizer %s is not supported for now', args.optimizer)

scheduler = WarmStartStepLR(optimizer, warm_up_epochs=5, step_size=30, gamma=0.1, verbose=True)
model = model.to(args.device)
model, optimizer = utils.wrap_model(model, optimizer, args)
log.info('Model is on %s', next(model.parameters()).device)

train_res, val_res = utils.train(model, criterion, optimizer, train_loader, args,
                                 exp_name='imagenet', val_loader=val_loader, classes=classes, scheduler=scheduler)

log.info('Process %d exited', args.rank)
