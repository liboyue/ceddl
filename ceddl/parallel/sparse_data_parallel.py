from copy import deepcopy
import torch
import torch.distributed as dist

from ceddl import log
from . import DistributedDataParallel
from . import communication_graphs as graphs

def _check_reqs(reqs):
    r"""Check if a list of requests are completed or not.

    Args:
        reqs:
            List of request handlers.

    Returns:
        (bool): True if all requests are complete or reqs is empty, else False.
    """
    return all([req.is_completed() for req in reqs])


class SparseDistributedDataParallel(DistributedDataParallel):
    r"""Stochastic Gradient Push.

    A slightly simplified implementation of `Stochastic Gradient Push for
    Distributed Deep Learning` https://arxiv.org/abs/1811.10792.
    """

    def __init__(self, module, n_peers=None, graph_type='exponential', async_op=False, **kwargs):
        r"""Init function.

        Args:
            module:
                The module to be wrapped.

            n_peers (optional):
                Number of peers for exponential graph. Ignored for other graphs. Defaults to None.

            graph_type (optional):
                Type of communication graph.

            async_op (optional):
                Use asynchronous communication if True. Defaults to False.

            kwargs:
                Args for base class.
        """

        super().__init__(module, **kwargs)

        self.async_op = async_op

        # Generate communication graph
        if graph_type == 'exponential':
            self.n_peers = n_peers
            self.G = graphs.ExponentialGraph(self.world_size, n_peers=self.n_peers)

        elif graph_type == 'complete':
            self.n_peers = self.world_size
            self.G = graphs.CompleteGraph(self.world_size)

        else:
            log.fatal('Graph type %s not supported', graph_type)

        self.flat_bufs = [param.clone().detach() for param in self.flat_parameters]

        self._recv_reqs = []
        self._send_reqs = []

        # The copy of model on node 0 for validation
        self._val_model = None


    def eval(self):
        # Create validation model
        with torch.no_grad():
            if self._val_model is None:
                if self.rank == 0:
                    self._val_model = self.module
                else:
                    self._val_model = deepcopy(self.module)
                    self._val_model.eval()
                    for p in self._val_model.parameters():
                        p.detach_()

                log.debug('Created _val_model')
            
            # Receive weigths from node 0
            for p in self._val_model.parameters():
                dist.broadcast(p, 0)

            log.debug('Updated _val_model')

        # Skip DistributedDataParallel's eval() function, because we don't need to communcate here.
        return super(DistributedDataParallel, self).eval()


    def forward(self, *args, **kwargs):
        r"""Forward function.

        If there are finished requests, mix received model parameters before
        calling super().forward().
        """
        if not self.training:
            return self._val_model(*args, **kwargs)

        if self.training and len(self._recv_reqs) > 0:
            self._process_async_recv()

        return super().forward(*args, **kwargs)


    def _process_async_recv(self):
        r"""Mix received model parameters with current model.

        The model is updated according to: new_model = current * ratio + received * (1 - ratio),
        where ratio is defined by ratio = 0.0001 / (#steps between update and receive + 2).
        No justification for this.
        """
        if len(self._recv_reqs) > 0 and _check_reqs(self._recv_reqs):
            log.debug('Update local parameters')

            for tensor, buf in zip(self.flat_parameters, self.flat_bufs):
                ratio = 0.0001 / (self._iter_counter + 2)
                tensor.mul_(ratio)
                tensor.add_(buf, alpha=1 - ratio)

            self._recv_reqs = []


    @torch.no_grad()
    def _communicate(self):

        log.debug('communicate')

        send = _check_reqs(self._send_reqs)

        if send:
            self._send_reqs = []
            flat_params_to_send = []

            for flat_param in self.flat_parameters:
                if self.async_op:
                    flat_param = flat_param.clone().detach()

                flat_param.div_(self.n_peers + 1)
                flat_params_to_send.append(flat_param)
        else:
            log.debug('Fail to send')
            # If we can't send now, try to send again at the next step
            self._iter_counter -= 1
            return

        for dst in range(self.world_size):
            if send and dst != self.rank:
                log.debug('Send')

                if self.G.has_predecessor(dst, self.rank):
                    self._send_reqs += self.reduce_tensors(flat_params_to_send, dst, self.G.process_group[dst])
            if dst == self.rank:
                # Recv
                if _check_reqs(self._recv_reqs):
                    log.debug('Recv')
                    self._recv_reqs = self.reduce_tensors(flat_params_to_send, self.rank, self.G.process_group[self.rank], self.flat_bufs)

        if not self.async_op:
            for req in self._recv_reqs + self._send_reqs:
                req.wait()

            self._recv_reqs = []
            self._send_reqs = []

            # Switch flat_bufs and flat_parameters to keep names consistent
            self.flat_bufs, self.flat_parameters = self.flat_parameters, self.flat_bufs

            # Re-assign parameters
            self.assign_unflattened_tensors(self.parameters(), self.flat_parameters)

        log.debug('communicate done')
