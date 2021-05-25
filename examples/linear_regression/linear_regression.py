import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from time import time

import ceddl
from ceddl import log
import ceddl.experiment_utils as utils


def generate_data(n_samples, dim, kappa, noise_variance):
    '''Helper function to generate data''' 

    powers = float(- np.log(kappa) / np.log(dim) / 2)

    S = np.power(np.arange(dim)+1, powers)
    X = np.random.randn(n_samples, dim) # Random standard Gaussian data
    X *= S                              # Conditioning

    # Normalize, while the condition number is not changed, it helps convergence
    norm = np.linalg.norm(X.T.dot(X), 2) / X.shape[0] 
    X /= norm

    # The "true" underlying variable
    w_0 = np.random.rand(dim, 1)

    # Generate Y and the optimal solution
    Y_0 = X.dot(w_0)
    Y = Y_0 + np.sqrt(noise_variance) * np.random.randn(n_samples, 1)

    # The best guess from data
    w_min = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

    # The best empirical loss
    loss = np.mean((X.dot(w_min) - Y)**2)

    # Because GPU usu use float32, we convert everything to float32
    return [torch.tensor(_.astype(np.float32)) for _ in [X, Y, w_0]] + [loss]


class Net(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x)


parser = utils.get_parser()
parser.add_argument('--dim', default=20, type=int, help='Dimension of the input data')
parser.add_argument('--n_samples', default=10000, type=int, help='Number of total samples')
parser.add_argument('--condition_number', default=10, type=float, help='The condition number of the problem')
parser.add_argument('--noise_variance', default=1, type=float, help='Variance of added noise')
args = utils.parse_args(parser)
args.distributed_val = False
if args.backend == 'nccl':
    log.warn('NCCL backend does not support scatter, using Gloo instead')
    args.backend = 'gloo'

utils.init(args)
# log.set_allowed_ranks(list(range(args.world_size)))

# Local data
local_n_samples = int(args.n_samples / args.world_size)
X = torch.zeros(local_n_samples, args.dim)
Y = torch.zeros(local_n_samples, 1)

if args.rank == 0:
    # Set the random seed so the data is the same in every run
    np.random.seed(0)

    # Generate random data at node 0,
    X_total, Y_total, w_0, loss_0 = generate_data(args.n_samples, args.dim, args.condition_number, args.noise_variance)
    log.info('Data generated, the best loss is %.7f' % loss_0)

    # then send to all other processes
    dist.scatter(X, [_ for _ in X_total.split(local_n_samples)])
    dist.scatter(Y, [_ for _ in Y_total.split(local_n_samples)])

else:
    dist.scatter(X)
    dist.scatter(Y)


dataset = torch.utils.data.TensorDataset(X, Y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=local_n_samples, shuffle=False)


model = Net(args.dim).to(args.device)
optimizer = ceddl.optim.NetworkSVRG(model, lr=args.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

args.apex = False
args.use_ref_module = True
model, optimizer = utils.wrap_model(model, args)

criterion = torch.nn.MSELoss()

if args.deterministic:
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

log.info('Model is on %s', next(model.parameters()).device)


train_res, val_res = utils.train(model, criterion, optimizer, train_loader, args,
                           val_loader=val_loader, exp_name='linear_regression')

log.info('Process %d exited', args.rank)
