from abc import ABC, abstractmethod
from dataclasses import dataclass
import re

from matplotlib import pyplot as plt
from networkx import DiGraph, multipartite_layout, draw_networkx, is_isomorphic
from networkx.drawing import spring_layout
from networkx.exception import NetworkXError
from networkx.relabel import convert_node_labels_to_integers


_re_patern_to_triplet_types = {
    re.compile(r"(.*)\|(.*),(.*)"): r"1|2,3",
    re.compile(r"(.*)\|(.*)\|(.*)"): r"1|2|3",
    re.compile(r"(.*)/(.*)\|(.*)"): r"1/2|3",
    re.compile(r"(.*)/(.*)/(.*)"): r"1/2/3",
    re.compile(r"(.*)/(.*)\\(.*)"): r"1/2\3",
    re.compile(r"(.*)\|(.*)\\(.*)"): r"1|2\3",
    re.compile(r"(.*),(.*)\|(.*)"): r"1,2|3",
    re.compile(r"(.*)\\(.*)\\(.*)"): r"1\2\3",
}

_triplet_types_to_re_pattern = {
    triplet_type: re_pattern for re_pattern, triplet_type in _re_patern_to_triplet_types.items()
}

_triplet_to_tuples = {
    r"1|2,3": lambda x, y, z: (None, (x, (None, tuple(sorted((y, z)))))),
    r"1|2|3": lambda x, y, z: (None, tuple(sorted((x, y, z)))),
    r"1/2|3": lambda x, y, z: (None, (z, (y, tuple({x})))),
    r"1/2/3": lambda x, y, z: (z, (y, tuple({x}))),
    r"1/2\3": lambda x, y, z: (y, tuple(sorted((x, z)))),
    r"1|2\3": lambda x, y, z: (None, (x, (y, tuple({z})))),
    r"1,2|3": lambda x, y, z: (None, (z, (None, tuple(sorted((x, y)))))),
    r"1\2\3": lambda x, y, z: (x, (y, tuple({z}))),
}


@dataclass(slots=True)
class AbstractTriplet(ABC):
    def __init__(self, triplet: str):
        self.labels = None
        self.parts = None
        self._string = str(triplet)
        self._tree_relation = self.__define_relations()

    def __define_relations(self) -> dict:
        for template in _re_patern_to_triplet_types.keys():
            if template.fullmatch(self._string):
                self.type = _re_patern_to_triplet_types[template]
                nodes = template.fullmatch(self._string).groups()
                relation_function = _triplet_to_tuples[self.type]
                return relation_function(*nodes)
        else:
            raise ValueError(f"Invalid triplet: {self._string}")

    def __str__(self):
        return self._string

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(r"{self._string}")'

    def __contains__(self, item):
        return item in self.labels

    def __iter__(self):
        return iter(self.labels)

    def __hash__(self):
        return hash(self._tree_relation)

    @abstractmethod
    def __eq__(self, other: str):
        ...


class AbstractGraph(ABC):
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

    def visualize(self, show=True, save=False, save_name=None, title: str = None):
        try:
            pos = multipartite_layout(self._tree, self.layers, align="horizontal", scale=-1)
        except NetworkXError:
            # pos = multipartite_layout(self._tree)
            # pos = spring_layout(self._tree)
            from networkx import bfs_layout, forceatlas2_layout

            pos = bfs_layout(self._tree, list(self._tree_dict.keys())[0], align="horizontal", scale=-1)
            # pos = forceatlas2_layout(self._tree)
        plt.figure()
        draw_networkx(
            self._tree,
            pos=pos,
            with_labels=True,
            arrows=True,
            nodelist=self.labels,
            labels={label: label for label in self.labels},
        )
        if title is not None:
            plt.title(title)
        if show:
            plt.show()
        if save:
            plt.savefig(f"{save_name}.png")

    def __eq__(self, other):
        if self.labels != other.labels:
            return False
        return is_isomorphic(
            convert_node_labels_to_integers(self._tree, label_attribute="label"),
            convert_node_labels_to_integers(other._tree, label_attribute="label"),
            node_match=lambda x, y: x == y if (x["label"] in self.labels or y["label"] in self.labels) else True,
        )


class AbstractGraphReconstruction(ABC):
    def __init__(self, labels: list[str]):
        self._labels = list(set(labels))

    @abstractmethod
    def reconstruct(self):
        ...
