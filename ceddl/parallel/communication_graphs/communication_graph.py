import networkx as nx
import torch.distributed as dist

from ceddl import log


class CommunicationGraph:
    """
    Base communication graph class.
    """

    def __init__(self, world_size, n_peers=None, t=0, cycle=None, rank=None):
        log.info('Using %s', self.__class__.__name__)

        self.t = t
        self.rank = rank
        self.world_size = world_size
        self.n_peers = n_peers if n_peers is not None else world_size
        self.cycle = cycle if cycle is not None else 1

        self._process_groups = []
        """
        The eletment i of _process_group[t] is a group of rank i and all its predecessors.
        """

        self._mixing_matrices = []
        self._graphs = []

        def generate_mixing_matrix(adj_matrix):
            """
            Generate a symmetric matrix with the same column sums from the adjaciancy matrix.
            """
            mixing_matrix = adj_matrix.astype(float)
            mixing_matrix /= mixing_matrix.sum(axis=1)[0]
            return mixing_matrix

        for t in range(self.cycle):

            adj_matrix = self.generate_adjacency_matrix(t)
            graph = nx.DiGraph(adj_matrix)

            self._graphs.append(graph)
            self._mixing_matrices.append(generate_mixing_matrix(adj_matrix))

            if dist.is_initialized():
                self._process_groups.append(self.create_process_group(graph))


    @property
    def adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph).todense()


    @property
    def graph(self):
        return self._graphs[self.t]


    @property
    def mixing_matrix(self):
        return self._mixing_matrices[self.t]


    @property
    def process_group(self):
        # The eletment i of process_group is a group of rank i and all its predecessors
        return self._process_groups[self.t]


    def predecessors(self, u):
        return self.graph.predecessors(u)


    def has_predecessor(self, u, v):
        return self.graph.has_predecessor(u, v)


    def update(self):
        self.t += 1
        self.t %= self.cycle


    def create_process_group(self, graph):
        group = []
        for rank in range(self.world_size):
            predecessors = list(graph.predecessors(rank))
            log.debug('creating %d\'s predecessoe group from %s',
                      rank, predecessors)
            group.append(dist.new_group(ranks=predecessors))
            log.debug('%d\'s predecessor group created from %s',
                      rank, predecessors)
        log.info('predecessor groups %d created', self.t)
        return group




    def draw(self):
        nx.draw_circular(self)
