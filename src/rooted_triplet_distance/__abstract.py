from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from networkx import DiGraph, multipartite_layout, draw_networkx, is_isomorphic


class AbstractTriplet(ABC):
    def __init__(self, triplet: str):
        self._labels = None
        self._parts = None
        self._string = str(triplet)

    def __str__(self):
        return self._string

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(r"{self._string}")'

    def __contains__(self, item):
        return item in self._labels

    def __iter__(self):
        return iter(self.labels)

    @property
    def parts(self):
        return self._parts

    @property
    def labels(self):
        return self._labels

    @abstractmethod
    def __eq__(self, other: str):
        ...


class AbstractTree(ABC):
    def __init__(self, tree: dict, labels: list[str] = None):
        self.layers = {}
        self._tree_dict = tree
        self._labels = labels
        self._triplets = []
        self._tree = self._construct_tree(tree)

    @property
    def triplets(self):
        if self._triplets == []:
            self._triplets = self._find_triplets()
        return self._triplets

    @property
    def labels(self):
        if self._labels is None:
            self._labels = list(self._tree.nodes)
        return self._labels

    @abstractmethod
    def _find_triplets(self) -> list[AbstractTriplet]:
        ...

    def _construct_tree(self, tree_dict, layer=0):
        tree = DiGraph()
        for node, children in tree_dict.items():
            current_layer = self.layers.get(str(layer), [])
            self.layers[str(layer)] = current_layer + [node] if node not in current_layer else current_layer
            for child in children:
                tree.add_edge(node, child)
                next_layer = self.layers.get(str(layer + 1), [])
                self.layers[str(layer + 1)] = next_layer + [child] if child not in next_layer else next_layer
                if tree_dict[node][child] != {}:
                    sub_tree = self._construct_tree({child: tree_dict[node][child]}, layer=layer + 1)
                    tree.add_edges_from(sub_tree.edges)
        return tree

    def visualize(self, show=True, save=False, save_name=None):
        pos = multipartite_layout(self._tree, self.layers)
        plt.figure()
        draw_networkx(
            self._tree,
            pos=pos,
            with_labels=True,
            arrows=True,
            nodelist=self.labels,
            labels={label: label for label in self.labels},
        )
        if show:
            plt.show()
        if save:
            plt.savefig(f"{save_name}.png")

    def __eq__(self, other):
        return is_isomorphic(self._tree, other._tree)


class AbstractTreeReconstruction(ABC):
    def __init__(self, labels: list[str]):
        self._labels = list(set(labels))

    @abstractmethod
    def reconstruct(self):
        ...
