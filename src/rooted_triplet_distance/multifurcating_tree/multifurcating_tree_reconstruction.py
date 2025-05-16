from .__multifurcating_triplet import MultifurcatingTriplet
from ..__abstract import AbstractGraphReconstruction

class MultifurcatingTreeReconstruction(AbstractGraphReconstruction):
    def __init__(
            self,
            labels: list[str],
            triplets: list[str | MultifurcatingTriplet],
            D_set: dict[str, set[str]] = None,
            numb_unlabelled_nodes: int = 0,
            root_name="root",):
        super().__init__(labels)
        self.__triplets = [MultifurcatingTriplet(triplet) for triplet in triplets]
        if D_set is None:
            self.__D_sets = {label: self.__create_D_set(label) for label in self._labels}
        else:
            self.__D_sets = D_set
        self.__numb_unlabelled_nodes = numb_unlabelled_nodes
        self.__root_name = root_name

    def __create_D_set(self, label: str) -> set[str]:
        D = set(self._labels)
        D.remove(label)
        for triplet in self.__triplets:
            if label in triplet:
                for l in triplet:
                    if l != label and l in D:
                        D.remove(l)
        return D

    def __children_of_root(self):
        children = set(self._labels)
        for triplet in self.__triplets:
            if triplet._type == r"1|2,3":
                for part in triplet.parts:
                    if isinstance(part, tuple):
                        children.difference_update(part)
        return children

    def __divide_in_branches(self, children: set[str]):
        branches = []
        placed_nodes = set()
        fanned_triplets = []
        for triplet in self.__triplets:
            if triplet._type == r"1|2,3":
                for part in triplet.parts:
                    if isinstance(part, tuple):
                        numb_of_placed_nodes = len(placed_nodes.intersection(part))
                        if numb_of_placed_nodes == 0:
                            branches.append(set(part))
                        elif numb_of_placed_nodes == 2:
                            branches_containing_nodes = [branch for branch in branches if branch.intersection(part) != set()]
                            for branch_to_remove in branches_containing_nodes:
                                branches.remove(branch_to_remove)
                            branches.append(set.union(*branches_containing_nodes, set(part)))
                        else:
                            branch_contaning_nodes = [branch for branch in branches if part[0] in branch or part[1] in branch][0]
                            branch_contaning_nodes.update(part)
                        placed_nodes.update(part)
            else:
                fanned_triplets.append(triplet)

        def resolve_fanned_triplet(fanned_triplet: MultifurcatingTriplet):
            numb_placed_nodes = len(fanned_triplet.labels.intersection(placed_nodes))
            branches_containing_nodes = [branch for branch in branches if fanned_triplet.labels.intersection(branch) != set()]
            if len(branches_containing_nodes) == 3 or (numb_placed_nodes == 3 and len(branches_containing_nodes) == 1):
                return
            elif numb_placed_nodes > len(branches_containing_nodes):
                for branch_to_remove in branches_containing_nodes:
                    branches.remove(branch_to_remove)
                branches.append(set.union(*branches_containing_nodes, fanned_triplet.labels))
                placed_nodes.update(fanned_triplet.labels)
                additional_fanned_triplets = [triplet for triplet in fanned_triplets if triplet.labels.intersection(branches[-1]) != set()]
                for additional_fanned_triplet in additional_fanned_triplets:
                    resolve_fanned_triplet(additional_fanned_triplet)
            elif numb_placed_nodes == 2:
                for unplaced_node in fanned_triplet.labels - placed_nodes:
                    placed_node = False
                    if unplaced_node in children:
                        for branch in branches:
                            if len(branch) >= 2 and self.__D_sets[unplaced_node].intersection(branch) == branch:
                                branch.add(unplaced_node)
                                placed_nodes.add(unplaced_node)
                                placed_node = True
                                additional_fanned_triplets = [triplet for triplet in fanned_triplets if
                                                              triplet.labels.intersection(branch) != set()]
                                for additional_fanned_triplet in additional_fanned_triplets:
                                    resolve_fanned_triplet(additional_fanned_triplet)
                                break
                    if not placed_node:
                        branches.append({unplaced_node})
                placed_nodes.update(fanned_triplet.labels)

        for fanned_triplet in fanned_triplets:
            resolve_fanned_triplet(fanned_triplet)

        for node in set(self._labels) - placed_nodes:
            if node in placed_nodes:
                continue
            for fanned_triplet in fanned_triplets:
                if node in fanned_triplet.labels:
                    resolve_fanned_triplet(fanned_triplet)
            if node in set(self._labels) - placed_nodes:
                for branch in branches:
                    if ((node in children and len(branch) >= 2) or node not in children) and self.__D_sets[node].intersection(branch) == branch:
                        branch.add(node)
                        break
                else:
                    branches.append({node})
            placed_nodes.add(node)
        return branches

    def reconstruct(self):
        if len(self._labels) == 0:
            return {self.__root_name: {}}
        elif len(self._labels) == 2:
            return {self.__root_name: {label: {} for label in self._labels}}
        tree = {}
        children = self.__children_of_root()
        branches = self.__divide_in_branches(children)
        if len(branches) == 1:
            raise ValueError("The triplets are contradictory or the tree is not multifurcating")
        for branch in branches:
            possible_roots = branch.intersection(children)
            for root in possible_roots:
                if len(branch - self.__D_sets[root] - {root}) == 0:
                    root_node = root
                    subtree = MultifurcatingTreeReconstruction(
                        list(branch.difference({root_node})),
                        [triplet for triplet in self.__triplets if all(node in branch for node in triplet)],
                        self.__D_sets,
                        self.__numb_unlabelled_nodes,
                        root_name=root_node,
                    )
                    break
            else:
                root_node = f"*_{self.__numb_unlabelled_nodes + 1}"
                self.__numb_unlabelled_nodes += 1
                subtree = MultifurcatingTreeReconstruction(
                    list(branch),
                    [triplet for triplet in self.__triplets if all(node in branch for node in triplet)],
                    self.__D_sets,
                    self.__numb_unlabelled_nodes,
                    root_name=root_node,
                )
            tree.update(subtree.reconstruct())
            self.__numb_unlabelled_nodes = subtree.__numb_unlabelled_nodes
        return {self.__root_name: tree}