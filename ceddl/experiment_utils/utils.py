import argparse
import os
from time import time
from pprint import pformat
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

import ceddl
from ceddl import log


def get_parser():
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument('--epochs',        default=1,     type=int,            help='Number of epochs'                       )
    parser.add_argument('--batch_size',    default=100,   type=int,            help='Batch size per worker'                  )
    parser.add_argument('--lr',            default=1e-3,  type=float,          help='Learning rate'                          )
    parser.add_argument('--momentum',      default=0,     type=float,          help='Momentum'                               )
    parser.add_argument('--weight_decay',  default=0,     type=float,          help='Weight decay'                           )
    parser.add_argument('--val_interval',  default=None,  type=int,            help='Number of iterations before validation' )
    parser.add_argument('--disp_interval', default=100,   type=int,            help='Number of iterations before display'    )
    parser.add_argument('--num_workers',   default=0,     type=int,            help='Number of workers for data loader'      )
    parser.add_argument('--cpu',           default=False, action='store_true', help='Use CPU'                                )
    parser.add_argument('--deterministic', default=False, action='store_true', help='Use fixed random seed'                  )

    # Apex args
    parser.add_argument('--apex',               default=False, action='store_true', help='Whether to use Apex'               )
    parser.add_argument('--keep_batchnorm_fp32',default=None,  action='store_true', help='Whether to use fp32 for batch norm')
    parser.add_argument('--loss_scale',         default=None,  type=float,          help='Loss scale factor'                 )
    parser.add_argument('--opt_level',          default='O0',  type=str,            help='Optimization level of Apex'        )

    # Distributed args
    parser.add_argument('--backend',             default='nccl',        type=str,             help='Distributed backend',           choices=['nccl', 'gloo', 'mpi'])
    parser.add_argument('--ddp',                 default='pytorch',     type=str,             help='Which DDP to use',      choices=['pytorch', 'DistributedDataParallel', 'SparseDistributedDataParallel', 'DistributedGradientParallel', 'NetworkDataParallel'])
    parser.add_argument('--sync_freq',           default=1,             type=int,             help='Number of iterations before synchroning'                       )
    parser.add_argument('--fp16_grads',          default=False,         action='store_true',  help='Whether to use fp16 gradients'                                 )
    parser.add_argument('--graph_type',          default='exponential', type=str,             help='Type of communication graph', choices=['complete', 'exponential', 'adaptive', 'er'] )
    parser.add_argument('--n_peers',             default=None,          type=int,             help='Number of iterations before synchroning'                       )
    parser.add_argument('--async_op',            default=False,         action='store_true',  help='Whether to use async communications for SGP'                   )
    parser.add_argument('--gradient_accumulation',default=False,        action='store_true',  help='Whether to use gradient accumulation'                          )

    # Misc args
    parser.add_argument('--data_path',      default=None,   type=str, help='Path to the data folder' )
    parser.add_argument('--output_dir',     default=None,   type=str, help='Path to the output folder' )
    parser.add_argument('-v','--verbosity', default='INFO', type=str, help='Verbosity of log', choices=['DEBUG', 'INFO', 'WARN'] )

    return parser

def parse_args(parser=None):
    if parser is None:
        parser = get_parser()

    args = parser.parse_args()
    
    # Load options from envs
    for name in ['MASTER_ADDR', 'MASTER_PORT']:
        setattr(args, name.lower(), os.environ[name])
    for name in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'WORLD_LOCAL_SIZE', 'WORLD_NODE_RANK']:
        setattr(args, name.lower(), int(os.environ[name]))
    args.node_rank = args.world_node_rank

    return args


def validate_args(args):
    # Check the backend and device compatibility
    if (not args.cpu) and (not torch.cuda.is_available()):
        log.warn('GPU is not availabel, using CPU instead')
        args.cpu = True

    args.device = torch.device('cpu') if args.cpu else torch.device('cuda:%d' % args.local_rank)

    if args.cpu and args.backend == 'nccl':
        log.warn('Setting backend to gloo when using CPU')
        args.backend = 'gloo'

    if args.gradient_accumulation and args.apex:
        log.warn('Using gradient accumulation with Apex may affect precision of accumulated gradient')

    return args


def init(args):
    log.set_rank(args.rank)
    if args.output_dir is not None:
        log.set_directory(args.output_dir)
    log.set_level(args.verbosity)

    args = validate_args(args)

    if args.apex:
        from apex import amp

    log.info('Configurations:\n' + pformat(args.__dict__))

    log.info('world_size = %d, batch_size = %d, device = %s, backend = %s',
              args.world_size, args.batch_size, args.device, args.backend)

    if not args.cpu:
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    if args.deterministic:
        torch.manual_seed(args.rank)
        np.random.seed(args.rank)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.rank)

    dist.init_process_group(backend=args.backend)


def wrap_model(model, args, optimizer=None):

    if args.apex:
        log.info('Apex wrapping')
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale)

    if args.ddp == 'DistributedDataParallel':
        model = ceddl.parallel.DistributedDataParallel(model, **args.__dict__)

    elif args.ddp == 'SparseDistributedDataParallel':
        model = ceddl.parallel.SparseDistributedDataParallel(model, **args.__dict__)

    elif args.ddp == 'NetworkDataParallel':
        model = ceddl.parallel.NetworkDataParallel(model, **args.__dict__)

    else:
        if args.cpu:
            device_ids = None
        else:
            device_ids = [args.rank]

        if args.ddp == 'pytorch':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        elif args.ddp == 'DistributedGradientParallel':
            model = ceddl.parallel.DistributedGradientParallel(model, fp16_grads=args.fp16_grads, sync_freq=args.sync_freq, device_ids=device_ids)


    return model, optimizer


def accuracy(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().float().mean()


def train(model, criterion, optimizer, train_loader, args, val_loader=None, exp_name=None, classes=None, scheduler=None):
    if args.apex:
        from apex import amp

    def _val():
        if args.val_interval is not None:
            val_start = time()
            model.eval()
            val_res.append([i, train_time, run_time, *validate(model, val_loader, criterion, classes=classes, device=args.device)])
            model.train()
            val_end = time()
            return val_end - val_start
        else:
            return 0

    def _save():
        if args.rank == 0:
            fname = get_fname(args, exp_name=None)
            save_data(train_res, val_res, fname, output_dir=args.output_dir)
            log.debug('Data saved to %s', fname)

    def _eta():
        _time = train_time / i * (total_batches - i)
        if args.val_interval is not None:
            _time += val_time / (i // args.val_interval + 1) * ((total_batches - i) // args.val_interval + 1)

        h = _time / 3600
        if h > 1:
            return "%.2fh" % h

        m = _time / 60
        if m > 1:
            return "%.2fm" % m

        return "%.2fs" % _time

    total_batches = len(train_loader) * args.epochs
    train_res = []
    val_res = []
    running_loss = []
    running_acc = []
    i = 0
    val_time = run_time = train_time = 0
    train_start = time()
    printed = False

    val_time += _val()

    log.info('Training started')
    model.train()
    optimizer.zero_grad()

    if args.gradient_accumulation and args.ddp == 'pytorch':
        model.require_backward_grad_sync = False


    for epoch in range(1, args.epochs + 1):

        for _, (data, target) in enumerate(train_loader):

            i += 1

            target = target.to(device=args.device, non_blocking=True)
            data = data.to(device=args.device, non_blocking=True)

            if args.ddp == 'pytorch':
                if args.gradient_accumulation and i % args.sync_freq != 0:
                    model.require_backward_grad_sync = False
                else:
                    model.require_backward_grad_sync = True

            # ==== Step begin ====
            output = model(data)
            loss = criterion(output, target)

            if args.gradient_accumulation:
                loss /= args.sync_freq

            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.ddp == 'DistributedGradientParallel' and printed == False:
                for n, p in model.named_parameters():
                    log.warn('%s.grad.dtype = %s, max difference between original grad and half precision grad is %f', n, p.grad.dtype, (p.grad - p.grad.clone().half()).abs().max())
                printed = True

            if not args.gradient_accumulation or i % args.sync_freq == 0 :
                log.debug('[%d/%d, %5d/%d] optimizer step', epoch, args.epochs, i, total_batches)
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.item()
            running_loss.append(loss)
            if classes is not None:
                acc = accuracy(output, target).item()
                running_acc.append(acc)

            # ==== Step done ====

            current_time = time()
            run_time = current_time - train_start
            train_time = run_time - val_time

            if args.gradient_accumulation:
                tmp_res = [i, train_time, run_time, loss * args.sync_freq]
            else:
                tmp_res = [i, train_time, run_time, loss]
            if classes is not None:
                tmp_res += [acc]

            train_res.append(tmp_res)

            if i % args.disp_interval == 0:
                log.info('[%d/%d, %5d/%d] local running loss %.5f, local running acc %.5f%%, average train time %.4f seconds per batch, eta %s',
                        epoch, args.epochs, i, total_batches, np.mean(running_loss), np.mean(running_acc) * 100, train_time / i, _eta())
                running_loss = []
                running_acc = []

            if args.val_interval is not None and i % args.val_interval == 0:
                val_time += _val()
                # Update saved data after every validation
                _save()

            # end for

        current_time = time()
        run_time = current_time - train_start
        train_time = run_time - val_time

        log.info('Training epoch %d ends, total run time %.4f seconds, average train time %.4f seconds per batch', epoch, run_time, train_time / i)

        if scheduler is not None:
            log.debug('schedule.step() called')
            scheduler.step()
     

    if args.val_interval is not None and i % args.val_interval != 0:
        val_time += _val()

    current_time = time()
    run_time = current_time - train_start
    train_time = run_time - val_time

    _save()

    if classes is not None:
        best_acc = max([x[-1] for x in val_res])
        log.info('Training finished, %d epochs, final val loss %.5f, final val acc %.5f%%, best val acc %.5f%%',
                 epoch, val_res[-1][-2], val_res[-1][-1] * 100, best_acc * 100)
    else:
        log.info('Training finished, %d epochs, final val loss %.5f', epoch, val_res[-1][-1])

    return train_res, val_res


@torch.no_grad()
def validate(model, val_loader, criterion, classes=None, device=None):
    log.info('Validating model')

    losses = []

    if classes is not None:
        confusion_matrix = np.zeros((len(classes), len(classes)))

    for data, target in val_loader:
        target = target.to(device=device, non_blocking=True)
        data = data.to(device=device, non_blocking=True)
        output = model(data)

        loss = criterion(output, target)
        losses.append(loss.cpu().item())

        if classes is not None:
            _, predicted = torch.max(output, 1)
            for i in range(len(target)):
                l = target[i]
                p = predicted[i]
                confusion_matrix[l][p] += 1

    loss = np.mean(losses) / dist.get_world_size()
    loss = torch.Tensor([loss]).to(device)

    dist.all_reduce(loss)

    loss = loss.cpu().item()

    if classes is not None:
        confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        dist.all_reduce(confusion_matrix)
        confusion_matrix = confusion_matrix.cpu().numpy()

    log.debug('Synchronized from other wokers')

    if classes is not None:

        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        confusion_matrix /= confusion_matrix.sum(axis=1)
        # log.debug(confusion_matrix)

        max_len = str(max([len(str(c)) for c in classes]))
        if len(classes) > 10:
            log.info('Accuracy of first 5 classes')
            for i in range(5):
                log.info('%-' + max_len + 's: %8.5f%%', classes[i], 100 * confusion_matrix[i, i])

            log.info('Accuracy of last 5 classes')
            for i in range(len(classes) - 5, len(classes)):
                log.info('%-' + max_len + 's: %8.5f%%', classes[i], 100 * confusion_matrix[i, i])
        else:
            log.info('Accuracy of each class')
            for i in range(len(classes)):
                log.info('%-' + max_len + 's: %8.5f%%', classes[i], 100 * confusion_matrix[i, i])

        log.info('Validation loss %.5f, accuracy %.5f%%', loss, acc * 100)

        return loss, acc

    else:
        log.info('Validation loss %.5f', loss)
        return [loss]


def get_fname(args, exp_name=None):

    fname = exp_name + '_' if exp_name is not None else ''

    if args.ddp == 'pytorch':
        fname += '%d_gpus_%s_%s' % (args.world_size, args.ddp.lower(), args.device.type)
    else:
        fname += '%d_gps_sync_%d_%s_%s' % (args.world_size, args.sync_freq, args.ddp.lower(), args.device.type)

    if args.fp16_grads:
        fname += '_fp16_grads'

    if args.apex:
        fname += '_apex_' + args.opt_level

        if args.keep_batchnorm_fp32:
            fname += '_keep_batchnorm_fp32'

    return fname


def save_data(train_res, val_res, fname, output_dir='data'):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]

    header = 'iterations train_time run_time loss'
    if len(train_res[0]) == 5:
        header += ' accuracy'

    def _save(res, name):
        res = np.array(res)
        np.savetxt(output_dir + '/full_' + fname + name, res, header=header, comments='')

        # Downsample if needed
        if len(res) > 500:
            idx = np.r_[:50, 50:500:5, 500:len(res):int((len(res)) / 100)]
            res = res[idx]

        np.savetxt(output_dir + '/' + fname + name, res, header=header, comments='')

    _save(train_res, '_train.txt')
    _save(val_res, '_val.txt')

    log.info('Data saved to %s/[full_]%s_[train/val].txt', output_dir, fname)

def plot(dirs, exp_name='figure', save=True, show=True, plot_train=False):

    import pandas as pd
    import matplotlib.pyplot as plt
    import tikzplotlib

    def _patch_tikzplotlib():

        def _new_draw_line2d(data, obj):
            content = ["\\addplot +[mark=none] "] + tikzplotlib._line2d._table(obj, data)[0]
            legend_text = tikzplotlib._util.get_legend_text(obj)
            if legend_text is not None:
                content.append(f"\\addlegendentry{{{legend_text}}}\n")
            return data, content

        def _new_init(self, data, obj):
            _tmp(self, data, obj)
            self.axis_options = [x for x in self.axis_options if 'mod' in x or 'label' in x or 'title' in x]

        _tmp = tikzplotlib._axes.Axes.__init__
        tikzplotlib._axes.Axes.__init__ = _new_init
        tikzplotlib._line2d.draw_line2d = _new_draw_line2d

    def _plot(ax, x, y, xlabel):
        for exp in exps.values():
            if plot_train:
                ax.plot(exp['train'][x], exp['train'][y], ':')
            mask = exp['val'][y] < 4
            ax.plot(exp['val'][x][mask], exp['val'][y][mask])
        ax.set(xlabel=xlabel, ylabel=y)

    exps = {}
    legends = []
    for dir_name in dirs:
        fnames = [_name[:-10] for _name in os.listdir(dir_name) if 'full' not in _name and 'train' in _name]
        for fname in fnames:
            train = pd.read_csv(dir_name + '/' + fname + '_train.txt', sep=' ')
            val = pd.read_csv(dir_name + '/' + fname + '_val.txt', sep=' ')
            exps[fname] = {'train': train, 'val': val}
            fname = fname.replace('_', '-')
            if plot_train:
                legends += [fname + '-train']
            legends += [fname + '-val']

    fig, axs = plt.subplots(1, 4)
    _plot(axs[0], 'iterations', 'loss', '\#iters')
    _plot(axs[1], 'train_time', 'loss', 'Training time / $s$')
    # _plot(axs[1], 'run_time', 'loss', 'Run time / $s$')
    _plot(axs[2], 'iterations', 'accuracy', '\#iters')
    _plot(axs[3], 'train_time', 'accuracy', 'Training time / $s$')
    # _plot(axs[3], 'run_time', 'accuracy', 'Run time / $s$')
    plt.legend(legends)

    if save:
        _patch_tikzplotlib()
        figure_content = tikzplotlib.get_tikz_code(filepath=exp_name, figure=fig, externalize_tables=True, override_externals=False, strict=False)

        with open(exp_name + '-data.tex', 'w') as f:
            data_fils = [x for x in os.listdir() if x.endswith('tsv')]
            for f_path in data_fils:
                with open(f_path, 'r') as f_data:
                    f.write('\\begin{filecontents}{%s}\n%s\\end{filecontents}\n\n\n' % (f_path, f_data.read()))
                os.remove(f_path)

        with open('%s.tex' % exp_name, 'w') as f:
            f.write('''\
\\documentclass{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage{pgfplots}
\\usepgfplotslibrary{groupplots,dateplot}
\\usetikzlibrary{patterns,shapes.arrows}
\\pgfplotsset{compat=newest}
\\begin{document}
\\input{%s-data.tex}
%s
\\end{document}''' % (exp_name, figure_content))

    if show:
        plt.show()
    '''
    \addlegendimage{black, dashed}
    \addlegendimage{black}
    \addlegendentry{Train}
    \addlegendentry{Val}
    \foreach \c in \LegendList {\addlegendentryexpanded{\c}}
}
    \node (title) at ($(group c1r1.center)!0.5!(group c2r1.center)+(0, 90pt)$) {#1};
    \def\LegendList{
    100\% all-reduce,
    10\% all-reduce,
    1\% all-reduce,
    100\% SGP,
    10\% SGP,
    1\% SGP
}
'''

