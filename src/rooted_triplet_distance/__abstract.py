from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from itertools import permutations


import networkx as nx
from matplotlib import pyplot as plt
from networkx import DiGraph, draw_networkx, is_isomorphic
from networkx.algorithms.components import biconnected_components
from networkx.algorithms.dag import descendants
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.drawing import bfs_layout
from networkx.relabel import convert_node_labels_to_integers
from multiset import Multiset
from numpy import sign

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


def _get_tree_dict(tree: DiGraph) -> dict:
    """
    Convert a directed graph into a tree dictionary representation.
    :param tree: The directed graph to convert.
    :return: A dictionary representing the tree structure.
    """
    tree_dict = {}

    def __add_children(node, sub_dict):
        children = list(tree.successors(node))
        if children:
            tree_dict[node] = {child: {} for child in children}
            for child in children:
                __add_children(child, tree_dict[node])

    root = [node for node in tree.nodes if tree.in_degree(node) == 0][0]
    __add_children(root, tree_dict)
    return tree_dict


@dataclass
class AbstractTriplet(ABC):
    def __init__(self, triplet: str):
        self.__slots__ = ("_string", "_tree_relation", "type", "labels", "parts")
        self.labels = None
        self.parts = None
        self._string = str(triplet)
        self._tree_relation = self.__define_relations()

    def __define_relations(self) -> dict:
        for template in _re_patern_to_triplet_types.keys():
            match = template.fullmatch(self._string)
            if match:
                self.type = _re_patern_to_triplet_types[template]
                nodes = match.groups()
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

    def __eq__(self, other: str | "AbstractTriplet") -> bool:
        """
        Checks if two AbstractTriplet instances are equal.
        :param other: An instance of AbstractTriplet or a string representation of a triplet.
        :return: True if the triplets are equal, False otherwise.
        """
        if not isinstance(other, (AbstractTriplet, str)):
            raise TypeError(
                f"unsupported operand type(s) for ==: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
            )
        other = AbstractTriplet(other)
        return self._tree_relation == other._tree_relation


class AbstractGraph(ABC):
    def __init__(self, tree: dict, labels: list[str] = None):
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
            for child in children:
                tree.add_edge(node, child)
                if tree_dict[node][child] != {}:
                    sub_tree = self._construct_tree({child: tree_dict[node][child]}, layer=layer + 1)
                    tree.add_edges_from(sub_tree.edges)
        return tree

    def perform_spr_move(
        self,
        node: str,
        new_parent_node: str | None = None,
        insert_edge: tuple[str, str] | None = None,
        allow_breaking_cycles: bool = False,
    ) -> tuple[dict, int]:
        """
        Perform an SPR move.
        :param node: The node for which to find SPR moves.
        :param new_parent_node: The new parent node for the SPR move (optional).
        :param insert_edge: A tuple representing the edge in between which the subtree should be added (optional).
        :return: The network after performing the SPR move and the distance between its original and new parent node.
        """
        if new_parent_node is None and insert_edge is None:
            raise ValueError("Either new_parent_node or insert_edge must be provided.")
        if new_parent_node is not None and insert_edge is not None:
            raise ValueError("Only one of new_parent_node or insert_edge can be provided.")
        if node not in self._tree:
            raise ValueError(f"Node {node} not found in the tree.")
        if any(
            node in cycle
            and not cycle.issubset(descendants(self._tree, node).union({node}))
            and not self._tree.in_degree(node) == 2
            for cycle in biconnected_components(self._tree.to_undirected())
            if len(cycle) > 2
        ):
            raise ValueError(f"Node {node} is part of a cycle, cannot perform SPR move.")
        if self._tree.in_degree(node) >= 2 and not allow_breaking_cycles:
            raise ValueError(
                f"Node {node} has more than one parent, cannot perform SPR move. Set allow_breaking_cycles=True to allow breaking cycles."
            )
        parent_edges = list(self._tree.in_edges(node))
        if new_parent_node is not None:
            if new_parent_node not in self._tree:
                raise ValueError(f"New parent node {new_parent_node} not found in the tree.")
            if new_parent_node == node:
                raise ValueError("New parent node cannot be the same as the node.")
            if new_parent_node in [edge[0] for edge in parent_edges]:
                raise ValueError(f"New parent node {new_parent_node} is already a parent of the node {node}.")
            if new_parent_node in descendants(self._tree, node):
                raise ValueError(f"New parent node {new_parent_node} is a descendant of the node {node}.")
            distance = min(
                len(shortest_path(self._tree.to_undirected(), parent_node, new_parent_node))
                for parent_node, _ in parent_edges
            )
        if insert_edge is not None:
            if insert_edge not in self._tree.edges:
                raise ValueError(f"Insert edge {insert_edge} not found in the tree.")
            if any(parent_edge == insert_edge for parent_edge in parent_edges):
                raise ValueError(f"Insert edge {insert_edge} is the same as the parent edge.")
            if insert_edge[0] in descendants(self._tree, node) or insert_edge[1] in descendants(self._tree, node):
                raise ValueError(f"Insert edge {insert_edge} is a descendant of the node {node}.")
            distance = min(
                len(shortest_path(self._tree.to_undirected(), parent_node, insert_edge[1]))
                for parent_node, _ in parent_edges
            )
        tree = self._tree.copy()
        for parent_edge in parent_edges:
            tree.remove_edge(*parent_edge)
        if insert_edge is not None:
            tree.remove_edge(*insert_edge)
            tree.add_edge(insert_edge[0], "SPR_TEMP")
            tree.add_edge("SPR_TEMP", insert_edge[1])
            tree.add_edge("SPR_TEMP", node)
        elif new_parent_node is not None:
            tree.add_edge(new_parent_node, node)
        return _get_tree_dict(tree), distance

    def visualize(self, show=True, save=False, save_name=None, title: str = None):
        pos = bfs_layout(self._tree, list(self._tree_dict.keys())[0], align="horizontal", scale=-1)
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

    def __eq__(self, other: "AbstractGraph") -> bool:
        if self.labels != other.labels:
            return False
        return is_isomorphic(
            convert_node_labels_to_integers(self._tree, label_attribute="label"),
            convert_node_labels_to_integers(other._tree, label_attribute="label"),
            node_match=lambda x, y: x == y if (x["label"] in self.labels or y["label"] in self.labels) else True,
        )

    def __sub__(self, other: "AbstractGraph") -> float:
        if not isinstance(other, AbstractGraph):
            raise TypeError(f"Cannot subtract {type(other)} from {type(self)}")
        triplets1 = set(self.triplets)
        triplets2 = set(other.triplets)
        sym_diff = triplets1.symmetric_difference(triplets2)
        return len(sym_diff) / len(triplets1.union(triplets2))

    def robinson_foulds_distance(self, other: "AbstractGraph") -> float:
        """
        Calculate the normalized Robinson-Foulds distance between two trees.
        :param other: The other tree to compare with.
        :return: The Robinson-Foulds distance.
        """
        if not isinstance(other, AbstractGraph):
            raise TypeError(f"Cannot calculate Robinson-Foulds distance with {type(other)}")
        cluster_set = {
            frozenset(descendants(self._tree, node).union({node}).intersection(self.labels))
            for node in self._tree.nodes
        }
        other_cluster_set = {
            frozenset(descendants(other._tree, node).union({node}).intersection(other.labels))
            for node in other._tree.nodes
        }
        sym_diff = cluster_set.symmetric_difference(other_cluster_set)
        return len(sym_diff) / len(cluster_set.union(other_cluster_set))

    def tripartition_distance(self, other: "AbstractGraph") -> float:
        """
        Calculate the tripartition distance between two trees.
        :param other: The other tree to compare with.
        :return: The tripartition distance.
        """
        if not isinstance(other, AbstractGraph):
            raise TypeError(f"Cannot calculate tripartition distance with {type(other)}")

        def _get_tripartition_for_node(graph: "AbstractGraph", node, root_node) -> tuple[frozenset, frozenset]:
            strict_descendants = set()
            node_descendants = descendants(graph._tree, node).union({node}).intersection(graph.labels)
            for descendant in node_descendants:
                if all(node in path for path in nx.all_simple_paths(graph._tree, root_node, descendant)):
                    strict_descendants.add(descendant)
            return (frozenset(strict_descendants), frozenset(node_descendants - strict_descendants))

        root = [n for n in self._tree.nodes if self._tree.in_degree(n) == 0][0]
        other_root = [n for n in other._tree.nodes if other._tree.in_degree(n) == 0][0]
        tripartitions = {_get_tripartition_for_node(self, node, root) for node in self._tree.nodes}
        other_tripartitions = {_get_tripartition_for_node(other, node, other_root) for node in other._tree.nodes}
        sym_diff = tripartitions.symmetric_difference(other_tripartitions)
        return len(sym_diff) / len(tripartitions.union(other_tripartitions))

    def mu_distance(self, other: "AbstractGraph") -> float:
        """
        Calculate the Mu distance between two trees.
        :param other: The other tree to compare with.
        :return: The Mu distance.
        """
        if not isinstance(other, AbstractGraph):
            raise TypeError(f"Cannot calculate Mu distance with {type(other)}")
        if self.labels != other.labels:
            raise ValueError("Labels of the trees must be the same for Mu distance calculation.")
        mu_set = Multiset(
            [
                tuple(len(list(nx.all_simple_paths(self._tree, node, label))) for label in sorted(self.labels))
                for node in self._tree.nodes
            ]
        )
        other_mu_set = Multiset(
            [
                tuple(len(list(nx.all_simple_paths(other._tree, node, label))) for label in sorted(other.labels))
                for node in other._tree.nodes
            ]
        )
        sym_diff = mu_set.symmetric_difference(other_mu_set)
        return len(sym_diff) / len(mu_set.union(other_mu_set))

    def average_sign_distance(self, other: "AbstractGraph") -> float:
        """
        Calculate the average signed distance between two trees.
        :param other: The other tree to compare with.
        :return: The average signed distance.
        """
        if not isinstance(other, AbstractGraph):
            raise TypeError(f"Cannot calculate average signed distance with {type(other)}")
        if self.labels != other.labels:
            raise ValueError("Labels of the trees must be the same for average signed distance calculation.")
        undirected_self = self._tree.to_undirected()
        undirected_other = other._tree.to_undirected()
        total_distance = 0
        count = 0
        for l1, l2, l3 in permutations(self.labels, 3):
            d_l1_l2 = len(shortest_path(undirected_self, l1, l2)) - 1
            d_l1_l3 = len(shortest_path(undirected_self, l1, l3)) - 1
            d_l1_l2_other = len(shortest_path(undirected_other, l1, l2)) - 1
            d_l1_l3_other = len(shortest_path(undirected_other, l1, l3)) - 1
            total_distance += 0.5 * abs(sign(d_l1_l2 - d_l1_l3) - sign(d_l1_l2_other - d_l1_l3_other))
            count += 1
        return total_distance / count if count > 0 else 0.0


class AbstractGraphReconstruction(ABC):
    def __init__(self, labels: list[str]):
        self._labels = list(set(labels))

    @abstractmethod
    def reconstruct(self):
        ...
