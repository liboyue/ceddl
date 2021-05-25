import numpy as np

from .communication_graph import CommunicationGraph
from ceddl import log


class CompleteGraph(CommunicationGraph):

    def __init__(self, world_size, **kwargs):
        super().__init__(world_size, cycle=1, **kwargs)

        log.info('Complete graph initialized')

    def update_graph(self):
        # Don't update
        pass

    def generate_adjacency_matrix(self, t):
        return np.ones((self.world_size, self.world_size))


if __name__ == '__main__':
    a = CompleteGraph(5, n_peers=1)
    log.info(str(a.adjacency_matrix))
    log.info(a.cycle)
