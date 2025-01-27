from random import choice

from rooted_triplet_distance.__abstract import AbstractTreeReconstruction
from rooted_triplet_distance.general_tree.__general_triplet import GeneralTriplet, _triplet_types_to_re_pattern as _triplet_types


class GeneralTreeReconstruction(AbstractTreeReconstruction):
    def __init__(
        self, labels: list[str], triplets: list[str | GeneralTriplet], numb_unlabelled_nodes: int = 0, descendants=None, separations=None
    ):
        super().__init__(labels)
        self.__triplets = [GeneralTriplet(triplet) for triplet in triplets]
        self.__numb_unlabelled_nodes = numb_unlabelled_nodes
        if descendants is None:
            descendants = {}
        if separations is None:
            separations = {}
        self.__descendants = {label: descendants.get(label, set()) for label in self._labels}
        self.__separation = {label: separations.get(label, set()) for label in self._labels}

    def __possible_root(self) -> list[str]:
        roots = []
        for label in self._labels:
            if self.__separation[label].intersection(self._labels) != set():
                continue
            if all([triplet.root == {label} for triplet in self.__triplets if label in triplet]) and all(
                label not in self.__descendants[other_label] for other_label in self._labels
            ):
                for triplet in self.__triplets:
                    if triplet.type == r"1/2\3" and triplet.root == {label}:
                        triplet_branches = triplet.branches
                        if any(set.union(*triplet_branches).issubset(self.__descendants[other_label]) for other_label in set(self._labels) - {label}):
                            break
                        node_1, node_2 = triplet.labels - {label}
                        descendants_1 = self.__descendants[node_1].union({node_1})
                        descendants_2 = self.__descendants[node_2].union({node_2})
                        if any({desc_1, desc_2} in other_triplet.branches for other_triplet in self.__triplets for desc_1 in descendants_1 for desc_2 in descendants_2):
                            print("Caught edge case")
                            break
                else:
                    roots.append(label)
        print(roots)
        return roots

    def __update_descendants(self, node:str, descendants: set) -> None:
        for label in self._labels:
            if label == node:
                self.__descendants[label].update(descendants)
                for descendant in descendants:
                    self.__descendants[label].update(self.__descendants[descendant])
            elif node in self.__descendants[label]:
                self.__descendants[label].update(descendants)

    def __update_separations(self, node:str, separated_from: set) -> None:
        self.__separation[node].update(separated_from)

    def __divide_in_branches(self, root: str) -> list[set[str]]:
        branches: list[set[str]] = []
        placed_nodes = set()
        rooted = root in self._labels
        fanned_triplets = []

        # Process triplets
        for triplet in self.__triplets:
            # Handle fanned triplets separately
            if triplet.type == "1|2|3":
                fanned_triplets.append(triplet)

            # Adjust branches for "1/2\3" triplets if root is not involved
            triplet_branches = triplet.branches
            if triplet.type == r"1/2\3" and root not in triplet:
                triplet_branches = [triplet.labels]

            for triplet_branch in triplet_branches:
                if rooted and root in triplet_branch:
                    triplet_branch.remove(root)

                # Merge branches if necessary
                if any(node in placed_nodes for node in triplet_branch):
                    branches_to_merge = [
                        branch for branch in branches if any(node in branch for node in triplet_branch)
                    ]
                    for branch_to_remove in branches_to_merge:
                        branches.remove(branch_to_remove)
                    branches.append(set.union(*branches_to_merge, triplet_branch))
                else:
                    branches.append(triplet_branch)

                placed_nodes.update(triplet_branch)

            # Ensure triplet branches are represented correctly
            branches_containing_triplet = [
                any(branch.intersection(triplet_branch) == triplet_branch for branch in branches)
                for triplet_branch in triplet_branches
            ]
            if sum(branches_containing_triplet) not in {1, len(triplet_branches)}:
                branches_to_remove = [
                    branch for branch, contains in zip(branches, branches_containing_triplet) if contains
                ]
                for branch_to_remove in branches_to_remove:
                    branches.remove(branch_to_remove)
                branches.append(set.union(*branches_to_remove))

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
                return [other_triplet for other_triplet in self.__triplets if other_triplet.type == r"1|2|3" and other_triplet.labels.intersection(triplet.labels) != set()]

        # Handle fanned triplets
        for triplet in fanned_triplets:
            extra_triplets_to_resolve = resolve_fanned_triplet(triplet)
            fanned_triplets.extend(extra_triplets_to_resolve or [])
            nodes_with_two_descendants_of_triplet = [node for node in self._labels if len(self.__descendants[node].intersection(triplet.labels)) == 2]
            for node in nodes_with_two_descendants_of_triplet:
                self.__update_descendants(node, triplet.labels)

        # Add remaining labels
        for label in set(self._labels) - placed_nodes - {root}:
            branches.append({label})
            placed_nodes.add(label)

        # Ensure descendants are valid
        for label in set(self._labels) - {root}:
            branch_containing_label = None
            for branch in branches:
                if label in branch:
                    branch_containing_label = branch
                    break
            if not self.__descendants[label].issubset(branch_containing_label):
                branches_to_add = [
                    branch
                    for branch in branches
                    if branch != branch_containing_label and self.__descendants[label].intersection(branch) != set()
                ]
                for branch in branches_to_add:
                    branches.remove(branch)
                branch_containing_label.update(*branches_to_add)
                fanned_triplets = [other_triplet for other_triplet in self.__triplets if other_triplet.type == r"1|2|3" and other_triplet.labels.intersection(branch_containing_label) != set()]
                for triplet in fanned_triplets:
                    extra_fanned_triplets = resolve_fanned_triplet(triplet)
                    fanned_triplets.extend(extra_fanned_triplets or [])
        return branches

    def reconstruct(self) -> dict[str, dict]:
        if len(self._labels) == 1:
            return {self._labels[0]: {}}
        tree = {}
        for triplet in self.__triplets:
            # Update descendants for each label in the triplet
            for label in triplet.labels:
                self.__update_descendants(label, triplet._descendants.get(label, set()))
                self.__update_separations(label, triplet._separations.get(label, set()))
        roots = self.__possible_root()
        if len(roots) >= 1:
            while len(roots) >= 1:
                root = choice(roots)
                branches = self.__divide_in_branches(root)
                if any(sum([branch.intersection(triplet.labels) != set() for branch in branches]) != 2 for triplet in self.__triplets if ({root} == triplet.root and triplet.type == r"1/2\3")):
                    print(branches)
                    roots.remove(root)
                    print(f"removed root: {root}")
                else:
                    break
            else:
                root = f"*_{self.__numb_unlabelled_nodes}"
                self.__numb_unlabelled_nodes += 1
                branches = self.__divide_in_branches(root)
        else:
            root = f"*_{self.__numb_unlabelled_nodes}"
            self.__numb_unlabelled_nodes += 1
            branches = self.__divide_in_branches(root)
        tree[root] = {}
        if root not in self._labels and len(branches) == 1:
            print(root)
            for triplet in self.__triplets:
                print(triplet)
            print(self.__descendants)
            print(self.__separation)
            self.__possible_root()
            raise ValueError("The triplets are contradictory")
        for branch in branches:
            found_bottom_case = False
            if len(branch) == 2:
                found_bottom_case = True
                for triplet in self.__triplets:
                    if triplet.type in (r"1|2,3", r"1,2|3") and branch in triplet.branches:
                        sub_dict = {f"*_{self.__numb_unlabelled_nodes}": {node: {} for node in branch}}
                        self.__numb_unlabelled_nodes += 1
                        break
                    elif (
                        triplet.type in (r"1|2\3", r"1/2|3", r"1/2/3", r"1\2\3")
                        and branch.issubset(triplet.branches[0])
                        and not any(node in triplet.root for node in branch)
                    ):
                        nodes = _triplet_types[triplet.type].fullmatch(triplet._string).groups()
                        parent = nodes[1]
                        child = branch.difference({parent}).pop()
                        sub_dict = {parent: {child: {}}}
                        break
                else:
                    for label in branch:
                        if all(
                            label not in self.__descendants[other_label]
                            for other_label in branch
                            if other_label != root
                        ):
                            parent = label
                            sub_dict = {parent: {node: {} for node in branch if node != parent}}
                            break
                    else:
                        sub_dict = {f"*_{self.__numb_unlabelled_nodes}": {node: {} for node in branch}}
                        self.__numb_unlabelled_nodes += 1

            if not found_bottom_case:
                subtree = GeneralTreeReconstruction(
                    list(branch),
                    [triplet for triplet in self.__triplets if all(node in branch for node in triplet)],
                    self.__numb_unlabelled_nodes,
                    self.__descendants,
                    self.__separation
                )
                sub_dict = subtree.reconstruct()
                self.__numb_unlabelled_nodes = subtree.__numb_unlabelled_nodes
            tree[root].update(sub_dict)
        return tree
