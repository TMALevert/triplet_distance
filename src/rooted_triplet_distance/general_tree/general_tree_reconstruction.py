from random import choice

from ..__abstract import AbstractGraphReconstruction
from .__general_triplet import GeneralTriplet


class GeneralTreeReconstruction(AbstractGraphReconstruction):
    def __init__(
        self,
        labels: list[str],
        triplets: list[str | GeneralTriplet],
        numb_unlabelled_nodes: int = 0,
        descendants: dict[str, set] = None,
        separations: dict[str, set] = None,
    ):
        super().__init__(labels)
        self.__triplets = [GeneralTriplet(triplet) for triplet in triplets]
        self.__numb_unlabelled_nodes = numb_unlabelled_nodes
        if descendants is not None and separations is not None:
            self.__descendants = {label: descendants[label] for label in self._labels}
            self.__separation = {label: separations[label] for label in self._labels}
        else:
            self.__descendants, self.__separation = self.__get_descendants_and_separations()

    def __compute_transitive_descendants(self, descendants: dict[str, set]) -> dict[str, set]:
        # Create a dictionary to store the transitive closure
        transitive_descendants = {key: set() for key in descendants}

        def dfs(node, current_descendants):
            if node not in descendants:
                return
            for child in descendants[node]:
                if child not in current_descendants:
                    current_descendants.add(child)
                    dfs(child, current_descendants)

        # Compute transitive closure for each key
        for key in descendants:
            dfs(key, transitive_descendants[key])

        return transitive_descendants

    def __get_descendants_and_separations(self) -> tuple[dict[str, set], dict[str, set]]:
        descendants = {label: set() for label in self._labels}
        separations = {label: set() for label in self._labels}
        for triplet in self.__triplets:
            for label in triplet.labels:
                descendants[label] = descendants[label].union(triplet._descendants.get(label, set()))
                separations[label] = separations[label].union(triplet._separations.get(label, set()))
        descendants = self.__compute_transitive_descendants(descendants)
        return descendants, separations

    def __find_possible_roots(self) -> list[str]:
        possible_roots: set = {
            label
            for label in self._labels
            if self.__separation[label].intersection(self._labels) == set()
            and all(label not in self.__descendants[other_label] for other_label in self._labels)
        }
        triplet_index = 0
        while len(possible_roots) >= 1 and triplet_index < len(self.__triplets):
            triplet = self.__triplets[triplet_index]
            # possible_roots.difference_update(triplet.labels - triplet.root)
            if triplet.type == r"1/2\3" and triplet.root in possible_roots:
                triplet_branches = triplet.branches
                if any(
                    set.union(*triplet_branches).issubset(self.__descendants[other_label])
                    for other_label in set(self._labels) - triplet.root
                ):
                    possible_roots.difference_update(triplet.root)
                    continue
                node_1, node_2 = triplet.labels - triplet.root
                descendants_1 = self.__descendants[node_1].union({node_1})
                descendants_2 = self.__descendants[node_2].union({node_2})
                if any(
                    {desc_1, desc_2} in other_triplet.branches
                    for other_triplet in self.__triplets
                    if triplet.type in ["1|2,3", "1,2|3"]
                    for desc_1 in descendants_1
                    for desc_2 in descendants_2
                ):
                    possible_roots.difference_update(triplet.root)
                    continue
            triplet_index += 1
        return list(possible_roots)

    def __divide_in_branches(self, root: str) -> list[set[str]]:
        branches: list[set[str]] = []
        fanned_triplets = []
        placed_nodes = set()

        for label in self._labels:
            if label == root or label in placed_nodes:
                continue
            descedants_and_label = self.__descendants[label].union({label})
            if descedants_and_label.intersection(placed_nodes) != set():
                branches_containing_descendants = [
                    branch for branch in branches if descedants_and_label.intersection(branch) != set()
                ]
                for branch_to_remove in branches_containing_descendants:
                    branches.remove(branch_to_remove)
                branches.append(set.union(*branches_containing_descendants, descedants_and_label))
            else:
                branches.append(descedants_and_label)
            placed_nodes.update(descedants_and_label)

        def resolve_fanned_triplet(triplet: GeneralTriplet) -> None | list:
            branches_containing_triplet = [
                any(branch.intersection(triplet_branch) == triplet_branch for triplet_branch in triplet.branches)
                for branch in branches
            ]
            if sum(branches_containing_triplet) == 2:
                idx1 = branches_containing_triplet.index(True)
                idx2 = branches_containing_triplet.index(True, idx1 + 1)
                branch_1, branch_2 = branches[idx1], branches[idx2]
                branches.remove(branch_1)
                branches.remove(branch_2)
                branches.append(branch_1.union(branch_2))
                other_triplets = [
                    other_triplet
                    for other_triplet in self.__triplets
                    if other_triplet.type == r"1|2|3"
                    and other_triplet.labels.intersection(triplet.labels) != set()
                    and other_triplet != triplet
                ]
                return other_triplets

        # Handle specific triplets
        for triplet in self.__triplets:
            if triplet.type == r"1|2|3":
                fanned_triplets.append(triplet)
            elif triplet.type in (r"1,2|3", r"1|2,3"):
                for triplet_branch in triplet.branches:
                    if len(triplet_branch) == 2:
                        branches_containing_branch = [
                            branch for branch in branches if triplet_branch.intersection(branch) != set()
                        ]
                        if len(branches_containing_branch) == 2:
                            branch_1, branch_2 = branches_containing_branch
                            branches.remove(branch_1)
                            branches.remove(branch_2)
                            branches.append(branch_1.union(branch_2))
        for fanned_triplet in fanned_triplets:
            extra_triplets_to_resolve = resolve_fanned_triplet(fanned_triplet)
            fanned_triplets.extend(extra_triplets_to_resolve or [])
            nodes_with_two_descendants_of_triplet = [
                node for node in self._labels if len(self.__descendants[node].intersection(fanned_triplet.labels)) == 2
            ]
            for node in nodes_with_two_descendants_of_triplet:
                self.__descendants[node].update(fanned_triplet.labels)
        self.__descendants = self.__compute_transitive_descendants(self.__descendants)
        return branches

    def reconstruct(self) -> dict[str, dict]:
        tree = {}
        roots = self.__find_possible_roots()
        while len(roots) >= 1:
            root = choice(roots)
            branches = self.__divide_in_branches(root)
            if any(
                sum([branch.intersection(triplet.labels) != set() for branch in branches]) != 2
                for triplet in self.__triplets
                if ({root} == triplet.root and triplet.type == r"1/2\3")
            ):
                roots.remove(root)
            else:
                break
        else:
            root = f"*_{self.__numb_unlabelled_nodes}"
            self.__numb_unlabelled_nodes += 1
            branches = self.__divide_in_branches(root)
            if len(branches) == 1:
                raise ValueError("The triplets are contradictory")
        tree[root] = {}
        for branch in branches:
            if len(branch) == 1:
                sub_dict = {branch.pop(): {}}
            elif len(branch) == 2:
                label1, label2 = branch
                if label1 in self.__descendants[label2]:
                    sub_dict = {label2: {label1: {}}}
                elif label2 in self.__descendants[label1]:
                    sub_dict = {label1: {label2: {}}}
                else:
                    sub_dict = {f"*_{self.__numb_unlabelled_nodes}": {node: {} for node in branch}}
                    self.__numb_unlabelled_nodes += 1
            else:
                subtree = GeneralTreeReconstruction(
                    list(branch),
                    [triplet for triplet in self.__triplets if all(node in branch for node in triplet)],
                    self.__numb_unlabelled_nodes,
                    self.__descendants,
                    self.__separation,
                )
                sub_dict = subtree.reconstruct()
                self.__numb_unlabelled_nodes = subtree.__numb_unlabelled_nodes
            tree[root].update(sub_dict)
        return tree
