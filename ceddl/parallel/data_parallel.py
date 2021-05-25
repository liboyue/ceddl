import torch
import torch.distributed as dist
from torch.nn.modules import Module

from ceddl import log

class DistributedDataParallel(Module):
    r"""The base distributed data parallel module.

    To reduce memory copy, flatten tensors into buckets, then assign unflattened
    new tensor to parameters.

    .. note::
        The actual communication happens at the beginning of each forward call.
        When training, the model should be validated before optimizer.step() to
        produce correct results.
    """

    def __init__(self, module, world_local_size=None, node_rank=None,
                 local_rank=None, sync_freq=1, num_streams=1, premultiplier=None,
                 **kwargs):
        r"""Init function.

        Args:
            module:
                The module to be wrapped.

            sync_freq:
                Number of steps between communications.

            num_streams:
                Number of CUDA streams to use for communication.

            premultiplier:
                The multiplier to be applied before communication. If not none,
                parameters will be multiplied by pre-multiplier before
                communication, then divided by the pre-multiplier after
                communication.
        """

        super().__init__()

        log.info('Using %s', self.__class__.__name__)

        self.module = module
        self.device = next(self.module.parameters()).device

        # Assume torch.dist is initialized
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank if local_rank is not None else self.rank
        self.node_rank = node_rank if node_rank is not None else 0
        self.world_local_size = world_local_size if world_local_size is not None else 1

        # When the counter equals to sync_freq, perform communication and reset
        self.premultiplier = premultiplier
        self.sync_freq = sync_freq
        self._iter_counter = 0

        self.param_info = [{'numel': param.numel(), 'shape': param.shape} for param in self.parameters()]
        self.flat_parameters, self.flat_indexes = self.flatten_tensors(list(self.parameters()))
        self.assign_unflattened_tensors(self.parameters(), self.flat_parameters)

        log.debug('Broadcasting init params')
        for param in self.flat_parameters:
            dist.broadcast(param, 0)
        log.debug('Broadcasting init params done')

        self.num_streams = num_streams
        if self.device.type == 'cuda':
            self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]


    def eval(self):
        self._communicate()
        return super().eval()


    def forward(self, *args, **kwargs):
        """Forward function.

        First, update the internal iteration counter. If communication is needed, call self._communicate(). Finally, call self.module instance.
        The update procedures are explicitly provided here to make the logic easier to understand.
        """
        if self.training:
            # Update iteration counter
            self._iter_counter %= self.sync_freq
            self._iter_counter += 1

            log.debug('forward called on %s, rank %d, _iter_counter %d', self.device, self.rank, self._iter_counter)

            if self._iter_counter == 1:
                self._communicate()

        return self.module(*args, **kwargs)


    @torch.no_grad()
    def _communicate(self):
        """Perform all-reduce on flattened parameters.

        To be rewritten by all subclasses.
        """

        log.debug('Communicate')

        if self.premultiplier is None:
            for flat_param in self.flat_parameters:
                flat_param.mul_(1 / self.world_size)
        else:
            for flat_param in self.flat_parameters:
                flat_param.mul_(self.premultiplier / self.world_size)

        reqs = self.all_reduce_tensors(self.flat_parameters)

        for req in reqs:
            req.wait()

        if self.premultiplier is not None:
            for flat_param in self.flat_parameters:
                flat_param.mul_(1 / self.premultiplier)

        self.assign_unflattened_tensors(self.parameters(), self.flat_parameters)


    def all_reduce_tensors(self, tensors):
        r"""Perform all-reduce on a list of tensors.

        Args:
            tensors (list):
                The list of tensors to reduce.

        Returns:
            list: Returns a list of request handlers.
        """
        return [dist.all_reduce(tensor, async_op=True) for tensor in tensors]


    def reduce_tensors(self, tensors, dst, group, bufs=None):
        r"""Perform reduce on a list of tensors.

        Args:
            tensors:
                The list of tensors to reduce.

            dst:
                The destination rank.

            group:
                The desired communication group.

            bufs (optional):
                The buffers to store reduced parameters. If not provided,
                in-place operations will be performed on tensors.

        Returns:
            list: Returns a list of request handlers.
        """
        reqs = []

        if bufs is None:
            if self.device.type == 'cpu':
                for tensor in tensors:
                    # Hack for Gloo on CPU. It may change the sender's tensor.
                    if dist.get_backend() == 'gloo':
                        tensor = tensor.clone().detach()
                    reqs.append(dist.reduce(tensor, dst, group=group, async_op=True))
            else:
                for i, tensor in enumerate(tensors):
                    with torch.cuda.stream(self.streams[i % self.num_streams]):
                        reqs.append(dist.reduce(tensor, dst, group=group, async_op=True))
            # fi
        else:
            if self.device.type == 'cpu':
                for tensor, buf in zip(tensors, bufs):
                    buf[:] = tensor[:]
                    reqs.append(dist.reduce(buf, dst, group=group, async_op=True))
            else:
                for i, tensor in enumerate(tensors):
                    with torch.cuda.stream(self.streams[i % self.num_streams]):
                        buf = bufs[i]
                        buf[:] = tensor[:]
                        reqs.append(dist.reduce(buf, dst, group=group, async_op=True))
            # fi
        # fi
        return reqs


    def flatten_tensors(self, tensors, buf_size=40000000):
        """Flatten tensors to several large chunks.

        Args:
            tensors:
                The list of tensors to flatten.

            buf_size (optional):
                The maximum flattened chunk size.

        Returns:
            (tuple): list, list
                (list): List of flattened tensors.

                (list): List of tensors ids.
        """
        dtypes = {param.dtype for param in tensors}
        params_by_dtype = {dtype: [(index, param) for index, param in enumerate(tensors) if param.dtype == dtype] for dtype in dtypes}

        flat_tensors = []
        indexes = []

        for dtype, index_params in params_by_dtype.items():
            size = 0
            tmp_params = []
            tmp_indexes = []

            for index, param in index_params:
                if size > buf_size: # Flatten if size exceeds the buffer size
                    flat_tensors.append(torch.cat([t.contiguous().view(-1) for t in tmp_params], dim=0).detach())
                    indexes.append(tmp_indexes)
                    tmp_params = []
                    tmp_indexes = []
                    size = 0

                size += param.numel() * param.element_size()
                tmp_indexes.append(index)
                tmp_params.append(param)

            # Deal with the last parameters
            flat_tensors.append(torch.cat([t.contiguous().view(-1) for t in tmp_params], dim=0).detach())
            indexes.append(tmp_indexes)

        return flat_tensors, indexes


    def unflatten_tensors(self, flat_tensors, indexes):
        """Unflatten tensors to original order and shapes.

        Args:
            flat_tensors:
                List of tensors to flatten.

            indexes:
                List of tensor indexes corresponding to each flat tensor.

        Returns:
            (list): List of unflattened tensors in original order and shapes.
        """
        outputs = []

        for index_list, flat_tensor in zip(indexes, flat_tensors):
            offset = 0

            for index in index_list:
                tensor = flat_tensor \
                        .narrow(0, offset, self.param_info[index]['numel']) \
                        .reshape(self.param_info[index]['shape'])
                outputs.append((index, tensor))
                offset += self.param_info[index]['numel']

        outputs = sorted(outputs, key=lambda x: x[0])

        return [output[1] for output in outputs]


    def assign_unflattened_tensors(self, tensors, flat_tensors):
        """Assign views of unflattened tensors to original tensors.

        Args:
            tensors:
                List of original tensors.

            flat_tensors:
                List of views of unflattened tensors.
        """
        new_tensors = self.unflatten_tensors(flat_tensors, self.flat_indexes)
        for old, new in zip(tensors, new_tensors):
            tmp = old.data
            old.data = new.data
            del tmp
