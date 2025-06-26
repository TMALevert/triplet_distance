from networkx.algorithms.tree import SpanningTreeIterator
from networkx.classes import DiGraph

from .. import GeneralTree
from ..__abstract import AbstractGraph
from .__network_triplet import NetworkTriplet


class LevelOneNetwork(AbstractGraph):
    """
    A class representing a level one network, which is a directed graph that can be used to find triplets among nodes in
    the network and compare with other networks.
    """

    def __init__(self, tree: dict, labels: list[str] = None):
        """
        Initializes a LevelOneNetwork instance.
        :param tree: A nested dictionary representing the network structure, where keys are node identifiers and values are lists of child nodes.
        :param labels: A list of labels for the nodes in the network. If None, all nodes are considered labeled.
        """
        super().__init__(tree, labels)
        self.__spanning_trees = None

    @property
    def spanning_trees(self) -> list[GeneralTree]:
        """
        Returns a list of spanning trees derived from the network obtained by removing one of the incoming edges of
        each contamination node.
        :return: A list of GeneralTree instances representing the spanning trees of the network.
        """
        if self.__spanning_trees is None:
            self.__spanning_trees = self.__get_spanning_trees()
        return self.__spanning_trees

    def __get_spanning_trees(self) -> list[GeneralTree]:
        """
        Generates all spanning trees of the network by removing one of the incoming edges of each contamination node.
        :return: A list of GeneralTree instances representing the spanning trees of the network.
        """
        spanning_trees = []
        network_edges = self._tree.edges()
        root = [v for v in self._tree.nodes if self._tree.in_degree(v) == 0][0]

        def __create_tree_dictionary(adjacency, root):
            tree_dict = {root: dict(adjacency[root])}
            for v in tree_dict[root]:
                tree_dict[root][v] = __create_tree_dictionary(adjacency, v)[v]
            return tree_dict

        for t in SpanningTreeIterator(self._tree.to_undirected()):
            undirected_edges = set(t.edges())
            actual_edges = [e for e in network_edges if e in undirected_edges or tuple(reversed(e)) in undirected_edges]
            tree = DiGraph(actual_edges)
            if len(tree.nodes) == len(self._tree.nodes) and len([v for v in tree.nodes if tree.in_degree(v) == 0]) == 1:
                tree_dict = __create_tree_dictionary(dict(tree.adj), root)
                spanning_trees.append(GeneralTree(tree_dict, self.labels))
        return spanning_trees

    def _find_triplets(self) -> list[NetworkTriplet]:
        """
        Finds all triplets in the level one network by examining the triplets in each spanning tree.
        :return: A list of NetworkTriplet instances representing the triplets found in the network.
        """
        triplets = set()
        for spanning_tree in self.spanning_trees:
            for triplet in spanning_tree.triplets:
                triplets.add(triplet)
        return list(triplets)
