from __future__ import annotations
from dataclasses import dataclass
import re


from ..__abstract import AbstractTriplet, _triplet_types_to_re_pattern


class GeneralTriplet(AbstractTriplet):
    """
    A class representing a general triplet, which can be used to represent the relationships between three nodes in a
    tree structure. It can also represent more complex triplets than just the fanned and resolved triplets.
    """

    def __init__(self, triplet: str | "GeneralTriplet"):
        """
        Initializes a GeneralTriplet instance.
        :param triplet: A string representation of the triplet or an instance of AbstractTriplet.
        """
        if isinstance(triplet, GeneralTriplet):
            super().__init__(triplet)
            self.parts = triplet.parts
            self.labels = triplet.labels
            self._branches = triplet._branches
            self._possible_root = triplet._possible_root
            self._descendants = triplet._descendants
            self._separations = triplet._separations
            self.type = triplet.type
            self.__nodes = triplet.__nodes
        else:
            super().__init__(triplet)
            self.parts = {tuple(part.split(",")) if "," in part else part for part in re.split(r"[/\\|]", self._string)}
            self.labels = set()
            for label in self.parts:
                self.labels = self.labels.union({label} if isinstance(label, str) else set(label))
            self._branches = self.__get_branches()
            self._possible_root = self.__get_possible_root()
            self.__nodes = _triplet_types_to_re_pattern[self.type].fullmatch(self._string).groups()
            self._descendants = self.__get_descendants()
            self._separations = self.__get_separations()

    @property
    def root(self) -> set[str]:
        """
        Returns the possible root of the triplet.
        :return: A set containing the possible root node(s) of the triplet.
        """
        return self._possible_root

    @property
    def branches(self) -> list[set]:
        """
        Returns the branches of the triplet.
        :return: A list of sets, where each set contains the labels of the nodes in that branch.
        """
        return [branch.copy() for branch in self._branches]

    def __get_descendants(self) -> dict[str, set]:
        """
        Returns a dictionary where keys are nodes and values are sets of their descendants in that triplet.
        :return: A dictionary mapping each node to its descendants in that triplet.
        """
        nodes = self.__nodes
        if self.type == "1|2,3":
            return {}
        elif self.type == "1|2|3":
            return {}
        elif self.type == "1/2|3":
            return {nodes[1]: {nodes[0]}}
        elif self.type == "1/2/3":
            return {nodes[2]: {nodes[0], nodes[1]}, nodes[1]: {nodes[0]}}
        elif self.type == r"1/2\3":
            return {nodes[1]: {nodes[0], nodes[2]}}
        elif self.type == r"1|2\3":
            return {nodes[1]: {nodes[2]}}
        elif self.type == r"1,2|3":
            return {}
        elif self.type == r"1\2\3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[2]}}

    def __get_separations(self) -> dict[str, set]:
        """
        Returns a dictionary where keys are nodes and values are sets of nodes that are separated from that node in the triplet.
        :return: A dictionary mapping each node to the set of nodes that are separated from it in the triplet.
        """
        nodes = self.__nodes
        if self.type == "1|2,3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0], nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.type == "1|2|3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0], nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.type == "1/2|3":
            return {nodes[0]: {nodes[2]}, nodes[1]: {nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.type == "1/2/3":
            return {}
        elif self.type == r"1/2\3":
            return {nodes[0]: {nodes[2]}, nodes[2]: {nodes[0]}}
        elif self.type == r"1|2\3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0]}, nodes[2]: {nodes[0]}}
        elif self.type == r"1,2|3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0], nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.type == r"1\2\3":
            return {}

    def __get_possible_root(self) -> set:
        """
        Returns the possible root of the triplet based on its type.
        :return:  A set containing the possible root node(s) of the triplet.
        """
        if self.type == "1|2,3":
            return set()
        elif self.type == "1|2|3":
            return set()
        elif self.type == "1/2|3":
            return set()
        elif self.type == "1/2/3":
            return {self._tree_relation[0]}
        elif self.type == r"1/2\3":
            return {self._tree_relation[0]}
        elif self.type == r"1|2\3":
            return set()
        elif self.type == r"1,2|3":
            return set()
        elif self.type == r"1\2\3":
            return {self._tree_relation[0]}

    def __get_branches(self) -> list[set]:
        """
        Returns the branches of the triplet based on its type.
        :return: A list of sets, where each set contains the labels of the nodes in that branch.
        """
        if self.type == "1|2,3":
            return [set(self._tree_relation[1][1][1]), {self._tree_relation[1][0]}]
        elif self.type == "1|2|3":
            return [{label} for label in self.labels]
        elif self.type == "1/2|3":
            return [{self._tree_relation[1][1][0], self._tree_relation[1][1][1][0]}, {self._tree_relation[1][0]}]
        elif self.type == "1/2/3":
            return [self.labels]
        elif self.type == r"1/2\3":
            return [{self._tree_relation[1][0]}, {self._tree_relation[1][1]}]
        elif self.type == r"1|2\3":
            return [{self._tree_relation[1][1][0], self._tree_relation[1][1][1][0]}, {self._tree_relation[1][0]}]
        elif self.type == r"1,2|3":
            return [set(self._tree_relation[1][1][1]), {self._tree_relation[1][0]}]
        elif self.type == r"1\2\3":
            return [self.labels]
