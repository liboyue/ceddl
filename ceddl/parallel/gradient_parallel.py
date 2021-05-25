import torch
import torch.distributed as dist

from ceddl import log

def fp16_compress_hook_nccl(state: object, bucket: dist._GradBucket):
    group_to_use = dist.group.WORLD
    world_size = dist.get_world_size()
    compressed_tensor = bucket.get_tensors()[0].div_(world_size).half()
    future = dist.all_reduce(compressed_tensor, async_op=True).get_future()
    return future


def fp16_compress_hook_gloo(state: object, bucket: dist._GradBucket):
    group_to_use = dist.group.WORLD
    world_size = dist.get_world_size()
    compressed_tensor = bucket.get_tensors()[0].div_(world_size).half()
    dist.all_reduce(compressed_tensor, async_op=False)
    future = torch.futures.Future()
    future.set_result([compressed_tensor])
    return future

class DistributedGradientParallel(torch.nn.parallel.DistributedDataParallel):
    r"""The distributed gradient parallel module. It is extended from PyTorch's DistributedDataParallel module, with synchronization scheduling and gradient compression.
    """

    def __init__(self, *args, sync_freq=1, fp16_grads=False, **kwargs):
        r"""Init function.

        Args:
            module:
                The module to be wrapped.

            sync_freq:
                Number of steps between communications.

            fp16_grads:
                Whether to use fp16 gradients.

            kwargs:
                Other args torch.nn.parallel.DistributedDataParallel requires.
        """
        log.info('Using %s', self.__class__.__name__)

        # Test PyTorch version
        if torch.__version__ < '1.7.0':
            log.FATAL("Please install PyTorch v1.7.0-rc1 to use DistributedGradientParallel!")

        if dist.get_backend() != 'nccl':
            log.warn('DistributedGradientParallel performs better with NCCL')

        super().__init__(*args, **kwargs)

        self.sync_freq = sync_freq
        self.fp16_grads = fp16_grads
        self._iter_counter = 0

        if self.fp16_grads:
            log.info('Using fp16 gradients')
            if dist.get_backend() != 'nccl':
                self._register_comm_hook(state=None, hook=fp16_compress_hook_gloo)
            else:
                self._register_comm_hook(state=None, hook=fp16_compress_hook_nccl)

        def _forward_pre_hook(*args, **kwargs):
            if self.training:
                # Update iteration counter
                self._iter_counter += 1
                self._iter_counter %= self.sync_freq

                log.debug('_forward_pre_hook called on %s, _iter_counter %d', self.device, self._iter_counter)

                if self._iter_counter == 0:
                    self.require_backward_grad_sync = True
                else:
                    self.require_backward_grad_sync = False

        self.register_forward_pre_hook(_forward_pre_hook)


    # We never use this function, so overwrite it with an empty function
    def _sync_params(self):
        pass
