import numpy as np
import networkx as nx

from .communication_graph import CommunicationGraph
from ceddl import log


class ERGraph(CommunicationGraph):

    def __init__(self, world_size, p, **kwargs):
        connected = False

        while connected is False:
            G = nx.erdos_renyi_graph(world_size, p, seed=0)
            connected = nx.is_connected(G)

        self._adjacency_matrix = nx.adjacency_matrix(G).toarray() + np.eye(world_size)
        print(self._adjacency_matrix)

        super().__init__(world_size, **kwargs)

    def generate_adjacency_matrix(self, t):
        return self._adjacency_matrix
