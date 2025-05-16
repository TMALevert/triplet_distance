from networkx import ancestors
from itertools import combinations

from .__multifurcating_triplet import MultifurcatingTriplet
from ..__abstract import AbstractGraph


class MultifurcatingTree(AbstractGraph):
    def __init__(self, tree: dict, labels: list[str] = None):
        super().__init__(tree, labels)

    def _find_triplets(self) -> list[MultifurcatingTriplet]:
        triplets = []
        for node1, node2, node3 in combinations(self._tree.nodes, 3):
            if node1 not in self.labels or node2 not in self.labels or node3 not in self.labels:
                continue
            ancestors1: set = ancestors(self._tree, node1)
            ancestors2 = ancestors(self._tree, node2)
            ancestors3 = ancestors(self._tree, node3)
            if (
                node1 in ancestors2.union(ancestors3)
                or node2 in ancestors1.union(ancestors3)
                or node3 in ancestors1.union(ancestors2)
            ):
                continue
            else:
                common_ancestors = ancestors1.intersection(ancestors2).intersection(ancestors3)
                if len((ancestors1.intersection(ancestors2) - common_ancestors)) > 0:
                    triplets.append(MultifurcatingTriplet(f"{node1},{node2}|{node3}"))
                    continue
                elif len((ancestors2.intersection(ancestors3) - common_ancestors)) > 0:
                    triplets.append(MultifurcatingTriplet(f"{node2},{node3}|{node1}"))
                    continue
                elif len((ancestors1.intersection(ancestors3) - common_ancestors)) > 0:
                    triplets.append(MultifurcatingTriplet(f"{node1},{node3}|{node2}"))
                    continue
                else:
                    triplets.append(MultifurcatingTriplet(f"{node1}|{node2}|{node3}"))
                    continue
        return triplets
