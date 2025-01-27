from copy import copy, deepcopy

from rooted_triplet_distance.multifurcating_tree.__multifurcating_triplet import MultifurcatingTriplet
from rooted_triplet_distance.__abstract import AbstractTreeReconstruction


class MultifurcatingTreeReconstruction(AbstractTreeReconstruction):
    def __init__(
        self,
        labels: list[str],
        triplets: list[str | MultifurcatingTriplet],
        numb_unlabelled_nodes: int = 0,
        root_name="root",
    ):
        super().__init__(labels)
        self.__triplets = [MultifurcatingTriplet(triplet) for triplet in triplets]
        self.__D_sets = {label: self.__create_D_set(label) for label in self._labels}
        self.__numb_unlabelled_nodes = numb_unlabelled_nodes
        self.__root_name = root_name

    def __create_D_set(self, label: str):
        D = copy(self._labels)
        D.remove(label)
        for triplet in self.__triplets:
            if label in triplet:
                for l in triplet:
                    if l != label and l in D:
                        D.remove(l)
        return D

    def __find_children_of_root(self):
        children = []
        for label in self._labels:
            if all(triplet.apart(label) for triplet in self.__triplets if all(l in self._labels for l in triplet)):
                children.append(label)
        return children

    def __divide_in_branches(
        self,
        remaining_nodes: list = None,
        triplets: list[str | MultifurcatingTriplet] = None,
        children: list[str] = None,
    ):
        if remaining_nodes is None:
            remaining_nodes = copy(self._labels)
        if triplets is None:
            triplets = self.__triplets
        fanned_triplets = []
        branches: list[set] = []
        for triplet in triplets:
            if len(triplet.parts) == 2:
                for part in triplet.parts:
                    if isinstance(part, tuple):
                        node_1, node_2 = part
                        if node_1 in remaining_nodes and node_2 in remaining_nodes:
                            branches.append({node_1, node_2})
                            remaining_nodes.remove(node_1)
                            remaining_nodes.remove(node_2)
                        elif node_1 not in remaining_nodes and node_2 in remaining_nodes:
                            for branch in branches:
                                if node_1 in branch:
                                    branch.update({node_2})
                                    remaining_nodes.remove(node_2)
                                    break
                        elif node_2 not in remaining_nodes and node_1 in remaining_nodes:
                            for branch in branches:
                                if node_2 in branch:
                                    branch.update({node_1})
                                    remaining_nodes.remove(node_1)
                                    break
                        else:
                            branches_to_combine = []
                            for id, branch in enumerate(branches):
                                if node_1 in branch and node_2 not in branch:
                                    branches_to_combine.append(branch)
                                elif node_2 in branch and node_1 not in branch:
                                    branches_to_combine.append(branch)
                            if len(branches_to_combine) > 1:
                                branches.append(set.union(*branches_to_combine))
                                for branch in branches_to_combine:
                                    branches.remove(branch)
            else:
                fanned_triplets.append(triplet)

        def check_fanned_triplets(fanned_triplets):
            for triplet in fanned_triplets:
                if all(part not in remaining_nodes for part in triplet.parts):
                    branches_containing_parts = [any(part in branch for part in triplet.parts) for branch in branches]
                    number_of_branches = sum(branches_containing_parts)
                    if number_of_branches == 2:
                        branch_1 = branches[branches_containing_parts.index(True)]
                        branch_2 = branches[
                            branches_containing_parts.index(True, branches_containing_parts.index(True) + 1)
                        ]
                        branches.remove(branch_1)
                        branches.remove(branch_2)
                        branches.append(branch_1.union(branch_2))
                    continue
                for part in triplet.parts:
                    if part in remaining_nodes:
                        other_parts = [p for p in triplet.parts if p != part]
                        for branch in branches:
                            if all(p in branch for p in other_parts):
                                branch.add(part)
                                remaining_nodes.remove(part)
                                break
                        else:
                            if all(p not in remaining_nodes for p in other_parts):
                                part_placed = False
                                if part in children:
                                    for branch in branches:
                                        if len(branch) >= 2 and branch == set(self.__D_sets[part]):
                                            branch.add(part)
                                            remaining_nodes.remove(part)
                                            part_placed = True
                                            break
                                if not part_placed:
                                    branches.append({part})
                                    remaining_nodes.remove(part)
                            break

        old_branches = deepcopy(branches)
        check_fanned_triplets(fanned_triplets)
        while old_branches != branches:
            old_branches = copy(branches)
            check_fanned_triplets(fanned_triplets)

        while remaining_nodes != []:
            for remaining_node in remaining_nodes:
                check_fanned_triplets([triplet for triplet in fanned_triplets if remaining_node in triplet])
                if remaining_node not in remaining_nodes:
                    continue
                if self.__D_sets[remaining_node] == []:
                    branches.append({remaining_node})
                    remaining_nodes.remove(remaining_node)
                    continue
                for branch in branches:
                    if len(set(self.__D_sets[remaining_node]) - branch) == 0 and remaining_node not in children:
                        branch.add(remaining_node)
                        break
                    elif (
                        remaining_node in children
                        and len(branch) >= 2
                        and len(branch - set(self.__D_sets[remaining_node])) == 0
                    ):
                        branch.add(remaining_node)
                        break
                else:
                    branches.append({remaining_node})
                remaining_nodes.remove(remaining_node)
                check_fanned_triplets([triplet for triplet in fanned_triplets if remaining_node in triplet])
        return branches

    def reconstruct(self):
        if len(self._labels) == 0:
            return {}
        tree = {}
        if len(self._labels) == 2:
            tree[self.__root_name] = {node: {} for node in self._labels}
            return tree
        children = self.__find_children_of_root()
        children_set = set(children)
        branches = self.__divide_in_branches(
            copy(self._labels),
            [triplet for triplet in self.__triplets if all(label in self._labels for label in triplet.labels)],
            children=children,
        )
        if len(branches) == 1:
            raise ValueError("The triplets are contradictory")
        for branch in branches:
            has_labelled_root = False
            if len(branch.intersection(children_set)) > 0 and len(branch) != 2:
                possible_roots = branch.intersection(children_set)
                for root in possible_roots:
                    if len(branch - set(self.__D_sets[root]) - {root}) == 0:
                        root_node = root
                        subtree = MultifurcatingTreeReconstruction(
                            list(branch.difference({root_node})),
                            self.__triplets,
                            self.__numb_unlabelled_nodes,
                            root_name=root_node,
                        )
                        has_labelled_root = True
                        break
            if not has_labelled_root:
                root_node = f"*_{self.__numb_unlabelled_nodes + 1}"
                subtree = MultifurcatingTreeReconstruction(
                    list(branch),
                    self.__triplets,
                    self.__numb_unlabelled_nodes + 1,
                    root_name=root_node,
                )
            tree[root_node] = subtree.reconstruct().get(root_node, {})
            self.__numb_unlabelled_nodes = subtree.__numb_unlabelled_nodes
        return {self.__root_name: tree}
