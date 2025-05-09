from itertools import combinations

from networkx import ancestors, descendants

from ..__abstract import AbstractGraph
from .__general_triplet import GeneralTriplet


class GeneralTree(AbstractGraph):
    def __init__(self, tree: dict, labels: list[str] = None):
        super().__init__(tree, labels)

    def _find_triplets(self) -> list[GeneralTriplet]:
        triplets = []
        for node1, node2, node3 in combinations(self._tree.nodes, 3):
            if node1 not in self.labels or node2 not in self.labels or node3 not in self.labels:
                continue
            descendants1 = descendants(self._tree, node1)
            descendants2 = descendants(self._tree, node2)
            descendants3 = descendants(self._tree, node3)
            ancestors1: set = ancestors(self._tree, node1)
            ancestors2 = ancestors(self._tree, node2)
            ancestors3 = ancestors(self._tree, node3)
            common_ancestors = ancestors1.intersection(ancestors2).intersection(ancestors3)
            if node2 in descendants1 and node3 in descendants1:
                parent = node1
                if node2 in descendants3:
                    triplets.append(GeneralTriplet(rf"{parent}\{node3}\{node2}"))
                elif node3 in descendants2:
                    triplets.append(GeneralTriplet(rf"{parent}\{node2}\{node3}"))
                elif ancestors2.intersection(ancestors3) - common_ancestors == {parent}:
                    triplets.append(GeneralTriplet(rf"{node2}/{parent}\{node3}"))
            elif node1 in descendants2 and node3 in descendants2:
                parent = node2
                if node1 in descendants3:
                    triplets.append(GeneralTriplet(rf"{parent}\{node3}\{node1}"))
                elif node3 in descendants1:
                    triplets.append(GeneralTriplet(rf"{parent}\{node1}\{node3}"))
                elif ancestors1.intersection(ancestors3) - common_ancestors == {parent}:
                    triplets.append(GeneralTriplet(rf"{node1}/{parent}\{node3}"))
            elif node1 in descendants3 and node2 in descendants3:
                parent = node3
                if node1 in descendants2:
                    triplets.append(GeneralTriplet(rf"{parent}\{node2}\{node1}"))
                elif node2 in descendants1:
                    triplets.append(GeneralTriplet(rf"{parent}\{node1}\{node2}"))
                elif ancestors1.intersection(ancestors2) - common_ancestors == {parent}:
                    triplets.append(GeneralTriplet(rf"{node1}/{parent}\{node2}"))
            elif node1 in descendants2:
                triplets.append(GeneralTriplet(rf"{node1}/{node2}|{node3}"))
            elif node2 in descendants1:
                triplets.append(GeneralTriplet(rf"{node2}/{node1}|{node3}"))
            elif node1 in descendants3:
                triplets.append(GeneralTriplet(rf"{node1}/{node3}|{node2}"))
            elif node3 in descendants1:
                triplets.append(GeneralTriplet(rf"{node3}/{node1}|{node2}"))
            elif node2 in descendants3:
                triplets.append(GeneralTriplet(rf"{node2}/{node3}|{node1}"))
            elif node3 in descendants2:
                triplets.append(GeneralTriplet(rf"{node3}/{node2}|{node1}"))
            else:
                if len((ancestors1.intersection(ancestors2) - common_ancestors)) > 0:
                    triplets.append(GeneralTriplet(f"{node1},{node2}|{node3}"))
                elif len((ancestors2.intersection(ancestors3) - common_ancestors)) > 0:
                    triplets.append(GeneralTriplet(f"{node2},{node3}|{node1}"))
                elif len((ancestors1.intersection(ancestors3) - common_ancestors)) > 0:
                    triplets.append(GeneralTriplet(f"{node1},{node3}|{node2}"))
                else:
                    triplets.append(GeneralTriplet(f"{node1}|{node2}|{node3}"))
        return triplets
