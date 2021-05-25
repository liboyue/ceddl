from scipy.linalg import circulant
import numpy as np

from .communication_graph import CommunicationGraph
from ceddl import log


class ExponentialGraph(CommunicationGraph):

    def __init__(self, world_size, **kwargs):
        cycle = int(np.log(world_size - 1) / np.log(2))
        super().__init__(world_size, cycle=cycle, **kwargs)

        log.info('Exponential graph initialized with cycle %d', self.cycle)


    def generate_adjacency_matrix(self, t):

        row = [0] * self.world_size
        for i in range(self.n_peers):
            row[2**(i + t) % self.world_size] = 1

        adj_matrix = circulant(row).T
        adj_matrix += np.diag(1 - adj_matrix.diagonal()) # Make sure the diagonal is 1

        return adj_matrix
