from copy import copy
from itertools import combinations, product
from random import choice

from ..__abstract import AbstractGraphReconstruction
from .level_one_network import NetworkTriplet


class LevelOneNetworkReconstruction(AbstractGraphReconstruction):
    def __init__(self, labels: list[str], triplets: list[str | NetworkTriplet], numb_unlabelled_nodes: int = 0, descendants: dict[str, set] = None, separations: dict[str, set] = None, sn_sets: set[frozenset[str]] = None):
        super().__init__(labels)
        self.__is_cycle: bool = None
        self.__triplets = [NetworkTriplet(triplet) for triplet in triplets]
        self.__numb_unlabelled_nodes = numb_unlabelled_nodes
        if descendants is not None and separations is not None and sn_sets is not None:
            self.__descendants = {label: descendants[label] for label in self._labels}
            self.__separation = {label: separations[label] for label in self._labels}
            self.__sn_sets = {sn_set for sn_set in sn_sets if sn_set.issubset(self._labels) and len(sn_set) != len(self._labels)}
        else:
            self.__descendants, self.__separation = self.__get_descendants_and_separations()
            self.__sn_sets = self.__get_non_trivial_sn_sets()
        self.__maximal_sn_sets = self.__get_maximal_sn_sets()

    def __get_non_trivial_sn_sets(self):
        sn_sets = {frozenset({label}) for label in self._labels}
        for i, j in combinations(self._labels, 2):
            sn_set = {i}
            z_set = {j}
            while len(z_set) != 0:
                z = z_set.pop()
                for label in sn_set:
                    for triplet in self.__triplets:
                        if not {label, z}.issubset(triplet.labels):
                            continue
                        other_label = (triplet.labels - {label, z}).pop()
                        if other_label in sn_set or other_label in z_set:
                            continue
                        if ((triplet.root != {other_label} and triplet._separations.get(other_label) != {label, z})
                                or triplet.type in ("1|2|3", r"1/2\3")
                                or (triplet.type in ("1|2,3", "1,2|3") and {other_label} not in triplet.branches)):
                            z_set.add(other_label)
                sn_set.add(z)
            if len(sn_set) != len(self._labels):
                sn_sets.add(frozenset(sn_set))
        return sn_sets

    def __get_maximal_sn_sets(self):
        maximal_sn_sets = []
        for sn_set in self.__sn_sets:
            if not any(sn_set.issubset(other_sn_set) for other_sn_set in self.__sn_sets if sn_set != other_sn_set):
                maximal_sn_sets.append(sn_set)
        return maximal_sn_sets

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
        possible_roots: set = {label for label in self._labels if
                               self.__separation[label].intersection(self._labels) == set() and all(
                                   label not in self.__descendants[other_label] for other_label in self._labels)}
        triplet_index = 0
        while len(possible_roots) >= 1 and triplet_index < len(self.__triplets):
            triplet = self.__triplets[triplet_index]
            if triplet.type == r"1/2\3" and triplet.root in possible_roots:
                triplet_branches = triplet.branches
                if any(set.union(*triplet_branches).issubset(self.__descendants[other_label]) for other_label in
                       set(self._labels) - triplet.root):
                    possible_roots.difference_update(triplet.root)
                    continue
                node_1, node_2 = triplet.labels - triplet.root
                descendants_1 = self.__descendants[node_1].union({node_1})
                descendants_2 = self.__descendants[node_2].union({node_2})
                if any({desc_1, desc_2} in other_triplet.branches for other_triplet in self.__triplets if
                       triplet.type in ["1|2,3", "1,2|3"] for desc_1 in
                       descendants_1 for desc_2 in descendants_2):
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
                branches_containing_descendants = [branch for branch in branches if descedants_and_label.intersection(branch) != set()]
                for branch_to_remove in branches_containing_descendants:
                    branches.remove(branch_to_remove)
                branches.append(set.union(*branches_containing_descendants, descedants_and_label))
            else:
                branches.append(descedants_and_label)
            placed_nodes.update(descedants_and_label)

        def resolve_fanned_triplet(triplet: NetworkTriplet) -> None | list:
            if any(sum([max_sn_set.issubset(self.__descendants[triplet_label]) for triplet_label in triplet.labels]) >= 1 for max_sn_set in self.__maximal_sn_sets):
                return None
            if len([other_triplet for other_triplet in self.__triplets if other_triplet.labels == triplet.labels]) > 1:
                return None
            for triplet_label in triplet.labels:
                for label in set(self._labels) - triplet.labels - set.union(*[self.__descendants[triplet_labels] for triplet_labels in triplet.labels]):
                    if triplet_label in self.__descendants[label] and not any(other_label in self.__descendants[label] for other_label in triplet.labels if other_label != triplet_label):
                        if any(max_sn_set.issubset(self.__descendants[label]) for max_sn_set in self.__maximal_sn_sets):
                            return None
            if len([max_sn_set for max_sn_set in self.__maximal_sn_sets if triplet.labels.intersection(max_sn_set) != set()]) == 3:
                return None
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
                other_triplets = [other_triplet for other_triplet in self.__triplets if
                                  other_triplet.type == r"1|2|3" and other_triplet.labels.intersection(
                                      triplet.labels) != set() and other_triplet != triplet]
                return other_triplets

        # Handle specific triplets
        for triplet in self.__triplets:
            if triplet.type == r"1|2|3":
                fanned_triplets.append(triplet)
            elif triplet.type in (r"1,2|3", r"1|2,3"):
                for triplet_branch in triplet.branches:
                    if len(triplet_branch) == 2:
                        branches_containing_branch = [branch for branch in branches if triplet_branch.intersection(branch) != set()]
                        if len(branches_containing_branch) == 2:
                            branch_1, branch_2 = branches_containing_branch
                            branches.remove(branch_1)
                            branches.remove(branch_2)
                            branches.append(branch_1.union(branch_2))
        for fanned_triplet in fanned_triplets:
            extra_triplets_to_resolve = resolve_fanned_triplet(fanned_triplet)
            fanned_triplets.extend(extra_triplets_to_resolve or [])
        return branches

    def __find_sink_of_cycle(self, source: str) -> list[set | frozenset]:
        sinks_and_descendants = set()
        if source in self._labels:
            for triplet in self.__triplets:
                if triplet.type == r"1/2\3" and set(source) == triplet.root:
                    node_1, node_2 = triplet.labels - triplet.root
                    common_descendants = self.__descendants[node_1].intersection(self.__descendants[node_2])
                    if node_1 in self.__descendants[node_2]:
                        common_descendants.update({node_1})
                    elif node_2 in self.__descendants[node_1]:
                        common_descendants.update({node_2})
                    if common_descendants != set():
                        sink_and_descendants = [max_sn_set for max_sn_set in self.__maximal_sn_sets if common_descendants.issubset(max_sn_set)][0]
                        sinks_and_descendants.update(sink_and_descendants)
        double_resolved_triplets = []
        for node1, node2, node3 in combinations(self._labels, 3):
            triplets_containing_nodes = [triplet for triplet in self.__triplets if all(node in triplet.labels for node in (node1, node2, node3))]
            if len(triplets_containing_nodes) == 2:
                if any(triplet.type in (r"1|2\3", r"1/2|3") for triplet in triplets_containing_nodes) and not any(triplet.type in (r"1/2\3", r"1\2\3", r"1/2/3") for triplet in triplets_containing_nodes):
                    triplet = \
                    [triplet for triplet in triplets_containing_nodes if triplet.type in (r"1|2\3", r"1/2|3")][0]
                    branch = [branch for branch in triplet.branches if len(branch) != 1][0]
                    sink = [node for node in branch if triplet._descendants.get(node, None) == None][0]
                    sinks_and_descendants.add(sink)
                elif any(triplet.type in (r"1|2\3", r"1/2|3") for triplet in triplets_containing_nodes) and any(
                        triplet.type in (r"1\2\3", r"1/2/3") for triplet in
                        triplets_containing_nodes):
                    triplet = [triplet for triplet in triplets_containing_nodes if triplet.type in (r"1\2\3", r"1/2/3")][0]
                    sink = [node for node in (node1, node2, node3) if triplet._descendants.get(node, None) == None][0]
                    sinks_and_descendants.add(sink)
                elif any(triplet.type == r"1/2\3" for triplet in triplets_containing_nodes) and any(triplet.type in (r"1|2\3", r"1/2|3") for triplet in triplets_containing_nodes):
                    triplet = [triplet for triplet in triplets_containing_nodes if triplet.type != r"1/2\3"][0]
                    branch = [branch for branch in triplet.branches if len(branch) == 1][0]
                    sink = branch.pop()
                    sinks_and_descendants.add(sink)
                elif all(triplet.type in (r"1|2,3", r"1,2|3") for triplet in triplets_containing_nodes):
                    double_resolved_triplets.extend(triplets_containing_nodes)
        for resolved_triplet in double_resolved_triplets:
            if resolved_triplet.labels.intersection(sinks_and_descendants) != set():
                continue
            a, b, c = resolved_triplet.labels
            for triplet in double_resolved_triplets:
                if len(triplet.labels.intersection({a,b,c})) == 2:
                    d = (triplet.labels - {a, b, c}).pop()
                    if d in sinks_and_descendants:
                        continue
                    abc_triplets = [triplet for triplet in double_resolved_triplets if triplet.labels == {a, b, c}]
                    abd_triplets = [triplet for triplet in double_resolved_triplets if triplet.labels == {a, b, d}]
                    acd_triplets = [triplet for triplet in double_resolved_triplets if triplet.labels == {a, c, d}]
                    bcd_triplets = [triplet for triplet in double_resolved_triplets if triplet.labels == {b, c, d}]
                    if sum([len(triplet_set) == 0 for triplet_set in
                            [abc_triplets, abd_triplets, acd_triplets, bcd_triplets]]) != 1:
                        continue
                    if len(abc_triplets) == 0:
                        sinks_and_descendants.add(d)
                    elif len(abd_triplets) == 0:
                        sinks_and_descendants.add(c)
                        break
                    elif len(acd_triplets) == 0:
                        sinks_and_descendants.add(b)
                        break
                    elif len(bcd_triplets) == 0:
                        sinks_and_descendants.add(a)
                        break

        if sinks_and_descendants == set():
            for label in set(self._labels) - {source}:
                only_one_triplet = [[triplet for triplet in self.__triplets if all(node in triplet for node in (label, node1, node2))][0] for node1, node2, in combinations(set(self._labels)-self.__descendants[label] - {label, source}, 2) if len([triplet for triplet in self.__triplets if all(node in triplet for node in (label, node1, node2))]) == 1]
                if len(only_one_triplet) == 0:
                    sink_and_descendants = [max_sn_set for max_sn_set in self.__maximal_sn_sets if label in max_sn_set][0]
                    sinks_and_descendants.update(sink_and_descendants)
                if len(only_one_triplet) >= 1:
                    for triplet in only_one_triplet:
                        if not all(len([other_triplet for other_triplet in self.__triplets if all(node in other_triplet for node in (other_label, *(triplet.labels - {label})))]) <= 1 for other_label in set(self._labels)-triplet.labels):
                            break
                    else:
                        sink_and_descendants = [max_sn_set for max_sn_set in self.__maximal_sn_sets if
                                                label in max_sn_set][0]
                        sinks_and_descendants.update(sink_and_descendants)
        sinks_and_descendants = [max_sn_set for max_sn_set in self.__maximal_sn_sets if max_sn_set.issubset(sinks_and_descendants)]
        return sinks_and_descendants

    def __find_singular_cycle_sink(self, sinks_and_descendants: list[set[str]], source: str) -> tuple[list[set[str]], list[set[str]]]:
        sinks_and_descendants = set(sinks_and_descendants)
        for sink_1, sink_2 in combinations(sinks_and_descendants, 2):
            s_1 = choice(list(sink_1))
            s_2 = choice(list(sink_2))
            for other_label in set(self._labels) - sink_1 - sink_2 - {source}:
                triplets = [triplet for triplet in self.__triplets if
                            all(node in triplet.labels for node in (s_1, s_2, other_label)) and triplet.type in (
                            r"1|2|3", r"1|2,3", r"1,2|3", r"1/2|3", r"1|2\3")]
                fanned_triplet = [triplet for triplet in triplets if triplet.type in (r"1|2|3", r"1/2|3", r"1|2\3")]
                resolved_triplet = [triplet for triplet in triplets if triplet.type in (r"1,2|3", r"1|2,3")]
                if len(fanned_triplet) == 0 and len(resolved_triplet) > 0:
                    if sink_1.issubset(self.__descendants[other_label]):
                        sinks_and_descendants.discard(sink_1)
                    elif sink_2.issubset(self.__descendants[other_label]):
                        sinks_and_descendants.discard(sink_2)
                elif len(fanned_triplet) >= 1 and len(resolved_triplet) > 0:
                    for res_triplet in resolved_triplet:
                        if {s_1} in res_triplet.branches and not any({s_2} in other_res_triplet.branches for other_res_triplet in resolved_triplet):
                            sinks_and_descendants.discard(sink_2)
                            break
                        elif {s_2} in res_triplet.branches and not any({s_1} in other_res_triplet.branches for other_res_triplet in resolved_triplet):
                            sinks_and_descendants.discard(sink_1)
                            break
        sinks_and_descendants = list(sinks_and_descendants)
        if len(sinks_and_descendants) > 1:
            labels_per_cycle = [set() for _ in range(len(sinks_and_descendants))]
            for sink_1, sink_2 in combinations(sinks_and_descendants, 2):
                s_1 = choice(list(sink_1))
                s_2 = choice(list(sink_2))
                for label in set(self._labels) - set().union(*sinks_and_descendants) - {source}:
                    resolved_triplet = [triplet for triplet in self.__triplets if
                               all(node in triplet.labels for node in (s_1, s_2, label)) and triplet.type in (r"1,2|3", r"1|2,3", r"1/2|3", r"1|2\3")]
                    if len(resolved_triplet) >= 1:
                        triplet = resolved_triplet[0]
                        if {s_1} in triplet.branches:
                            labels_per_cycle[sinks_and_descendants.index(sink_2)].add(label)
                        elif {s_2} in triplet.branches:
                            labels_per_cycle[sinks_and_descendants.index(sink_1)].add(label)
            return sinks_and_descendants, labels_per_cycle
        else:
            return sinks_and_descendants, [set(self._labels) - {source}]

    def __resolve_cycle(self, sink_and_descendants: set[str], cycle_labels: set[str], source: str = None) -> tuple[list[set[str]], set[str], set[str]]:
        internal_cycle_vertices = {label for label in cycle_labels if self.__descendants[label].intersection(sink_and_descendants) == sink_and_descendants - {label}}
        cycle_branches: list[set[str]] = []
        internal_cycle_vertex_to_branch = []
        for node1, node2 in combinations(cycle_labels - sink_and_descendants, 2):
            for sink_node in sink_and_descendants:
                triplets_containing_nodes = [triplet for triplet in self.__triplets if
                                             all(node in triplet.labels for node in (node1, node2, sink_node))]
                if len(triplets_containing_nodes) == 1:
                    branches_containing_nodes = [branch for branch in cycle_branches if
                                                 node1 in branch - internal_cycle_vertices or node2 in branch - internal_cycle_vertices]
                    if len(branches_containing_nodes) >= 1:
                        for branch in branches_containing_nodes:
                            cycle_branches.remove(branch)
                        new_branch = set.union(*branches_containing_nodes, {node1, node2} - {
                            source} - internal_cycle_vertices)
                        cycle_branches.append(new_branch)
                        break
                    else:
                        cycle_branches.append({node1, node2} - {source} - internal_cycle_vertices)
                        break
                else:
                    if any(triplet.type in (r"1|2\3", r"1/2|3") for triplet in triplets_containing_nodes) and any(triplet.type == r"1/2\3" for triplet in triplets_containing_nodes):
                        branches_containing_nodes = [branch for branch in cycle_branches if
                                                     node1 in branch - internal_cycle_vertices or node2 in branch - internal_cycle_vertices]
                        if len(branches_containing_nodes) >= 1:
                            for branch in branches_containing_nodes:
                                cycle_branches.remove(branch)
                            new_branch = set.union(*branches_containing_nodes, {node1, node2} - {source} - internal_cycle_vertices)
                            cycle_branches.append(new_branch)
                        else:
                            cycle_branches.append({node1, node2} - {source} - internal_cycle_vertices)
                    if not any(node1 in branch for branch in cycle_branches) and node1 not in {source} and node1 not in internal_cycle_vertices:
                        cycle_branches.append({node1})
                    if not any(node2 in branch for branch in cycle_branches) and node2 not in {source} and node2 not in internal_cycle_vertices:
                        cycle_branches.append({node2})
                    for triplet in triplets_containing_nodes:
                        if triplet.type == r"1/2\3":
                            if node1 in internal_cycle_vertices - {source} and node2 not in internal_cycle_vertices:
                                internal_cycle_vertex_to_branch.append((node1, node2))
                            elif node2 in internal_cycle_vertices - {source} and node1 not in internal_cycle_vertices:
                                internal_cycle_vertex_to_branch.append((node2, node1))
        for int_vertex, branch_vertex in internal_cycle_vertex_to_branch:
            branch = [branch for branch in cycle_branches if branch_vertex in branch]
            branch[0].add(int_vertex)
        if cycle_branches == [] and internal_cycle_vertex_to_branch == [] and len(cycle_labels - sink_and_descendants) == 2:
            cycle_branches = [{node} for node in cycle_labels - sink_and_descendants]
        return cycle_branches, sink_and_descendants, internal_cycle_vertices

    def __find_cycle_order(self, branches: list[set[str]], sink_branch: set[str], cycle_vertices: set[str], source: str) -> tuple[list[set[str]], list[set[str]]]:
        left, right = [], []

        def place_together(branch1, branch2):
            if left == [] and right == []:
                left.extend([branch1, branch2])
                return
            if (branch1 in left and branch2 in left) or (branch1 in right and branch2 in right):
                return
            elif branch1 in left:
                left.append(branch2)
            elif branch1 in right:
                right.append(branch2)
            elif branch2 in left:
                left.append(branch1)
            elif branch2 in right:
                right.append(branch1)

        def place_apart(branch1, branch2):
            if left == [] and right == []:
                left.append(branch1)
                right.append(branch2)
                return
            if (branch1 in left and branch2 in right) or (branch1 in right and branch2 in left):
                return
            elif branch1 in left:
                right.append(branch2)
            elif branch1 in right:
                left.append(branch2)
            elif branch2 in left:
                right.append(branch1)
            elif branch2 in right:
                left.append(branch1)

        branches += [{cycle_vertex} for cycle_vertex in cycle_vertices - sink_branch - {source} if not any(cycle_vertex in branch for branch in branches)]

        ordering = {str(branch): [] for branch in branches}

        def add_to_ordering(upper_branch, lower_branch):
            ordering[str(upper_branch)].append(lower_branch)
            for lower_lower_branch in ordering[str(lower_branch)]:
                add_to_ordering(upper_branch, lower_lower_branch)

        for branch_1, branch_2 in combinations(branches, 2):
            for node1, node2 in product(branch_1, branch_2):
                for sink in sink_branch.union({source}):
                    triplets_containing_nodes = [triplet for triplet in self.__triplets if all(node in triplet.labels for node in (node1, node2, sink))]
                    for triplet_type in (r"1/2/3", r"1\2\3", r"1|2\3", r"1/2|3", r"1|2,3", r"1,2|3", r"1/2\3"):
                        triplet_of_type = [triplet for triplet in triplets_containing_nodes if triplet.type == triplet_type]
                        if len(triplet_of_type) >= 1:
                            for triplet in triplet_of_type:
                                if triplet_type in (r"1/2/3", r"1\2\3"):
                                    if {node1} == triplet.root:
                                        place_together(branch_1, branch_2)
                                        add_to_ordering(branch_1, branch_2)
                                        break
                                    elif {node2} == triplet.root:
                                        place_together(branch_1, branch_2)
                                        add_to_ordering(branch_2, branch_1)
                                        break
                                elif triplet_type in (r"1|2\3", r"1/2|3"):
                                    if {sink} in triplet.branches:
                                        place_together(branch_1, branch_2)
                                        if node1 in triplet._descendants.keys():
                                            add_to_ordering(branch_1, branch_2)
                                        else:
                                            add_to_ordering(branch_2, branch_1)
                                        break
                                    elif node1 in cycle_vertices and node2 in cycle_vertices:
                                        place_apart(branch_1, branch_2)
                                        break
                                    else:
                                        other_triplet = [other_triplet for other_triplet in triplets_containing_nodes if triplet != other_triplet][0]
                                        if other_triplet.type in (r"1,2|3", r"1|2,3"):
                                            if {node1} in other_triplet.branches or {node2} in other_triplet.branches:
                                                place_apart(branch_1, branch_2)
                                                break
                                            else:
                                                place_together(branch_1, branch_2)
                                                if {node1} in triplet.branches:
                                                    add_to_ordering(branch_1, branch_2)
                                                else:
                                                    add_to_ordering(branch_2, branch_1)
                                                break
                                elif triplet_type in (r"1|2,3", r"1,2|3"):
                                    if {sink} in triplet.branches:
                                        place_together(branch_1, branch_2)
                                        other_triplet = [other_triplet for other_triplet in triplets_containing_nodes if triplet != other_triplet][0]
                                        if other_triplet.type in (r"1|2,3", r"1,2|3", r"1/2|3", r"1|2\3"):
                                            if {node1} in other_triplet.branches:
                                                add_to_ordering(branch_1, branch_2)
                                            else:
                                                add_to_ordering(branch_2, branch_1)
                                            break
                                        continue
                                    else:
                                        other_triplet = [other_triplet for other_triplet in triplets_containing_nodes if triplet != other_triplet][0]
                                        if other_triplet.type in (r"1|2,3", r"1,2|3") and {sink} not in other_triplet.branches:
                                            place_apart(branch_1, branch_2)
                                elif triplet_type == r"1/2\3":
                                    place_apart(branch_1, branch_2)
                                    break
                            else:
                                continue
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        if len(branches) == 1:
            left = branches

        left_ordered = sorted(left, key=lambda x: len(ordering[str(x)]), reverse=True)
        right_ordered = sorted(right, key=lambda x: len(ordering[str(x)]), reverse=True)
        return left_ordered, right_ordered

    def reconstruct(self) -> dict:
        self.__is_cycle = False
        tree = {}
        if len(self._labels) == 1:
            sub_dict = {self._labels[0]: {}}
            return sub_dict
        elif len(self._labels) == 2:
            label1, label2 = self._labels
            if label1 in self.__descendants[label2]:
                sub_dict = {label2: {label1: {}}}
            elif label2 in self.__descendants[label1]:
                sub_dict = {label1: {label2: {}}}
            else:
                sub_dict = {f"*_{self.__numb_unlabelled_nodes}": {node: {} for node in self._labels}}
                self.__numb_unlabelled_nodes += 1
            return sub_dict
        roots = self.__find_possible_roots()
        if len(roots) >= 1:
            root = choice(roots)
        else:
            root = f"*_{self.__numb_unlabelled_nodes}"
            self.__numb_unlabelled_nodes += 1
        tree[root] = {}
        branches = self.__divide_in_branches(root)
        if len(branches) == 1:
            if root in self._labels:
                subtree = LevelOneNetworkReconstruction(
                    branches[0],
                    [triplet for triplet in self.__triplets if all(node in branches[0] for node in triplet)],
                    self.__numb_unlabelled_nodes,
                    self.__descendants,
                    self.__separation,
                    self.__sn_sets
                )
                if len(subtree.__find_possible_roots()) == 1:
                    tree[root].update(subtree.reconstruct())
                    self.__numb_unlabelled_nodes = subtree.__numb_unlabelled_nodes
                    return tree
            sinks_and_descendants = self.__find_sink_of_cycle(root)
            if len(sinks_and_descendants) > 1:
                sinks_and_descendants, cycles_labels = self.__find_singular_cycle_sink(sinks_and_descendants, root)
            else:
                cycles_labels = [set(self._labels) - {root}]
            for i in range(len(sinks_and_descendants)):
                self.__is_cycle = True
                sink_and_descendants = sinks_and_descendants[i]
                cycle_labels = cycles_labels[i]
                cycle_branches, sink_and_descendants, cycle_vertices = self.__resolve_cycle(sink_and_descendants, cycle_labels, root)
                left, right = self.__find_cycle_order(cycle_branches, sink_and_descendants, cycle_vertices, root)
                left_nodes = set().union(*(left + [sink_and_descendants]))
                right_nodes = set().union(*(right + [sink_and_descendants]))
                left_triplets = []
                right_triplets = []
                for triplet in self.__triplets:
                    if all(node in left_nodes for node in triplet):
                        triplet_branch_containing_sink = [branch for branch in triplet.branches if sink_and_descendants.intersection(branch) != set()]
                        if len(triplet_branch_containing_sink) != 1 or triplet.type == r"1/2\3":
                            left_triplets.append(triplet)
                        else:
                            triplet_branch_containing_sink = triplet_branch_containing_sink[0]
                            if len(triplet_branch_containing_sink.intersection(sink_and_descendants)) == 2:
                                if triplet.type in (r"1\2\3", r"1/2/3"):
                                    left_triplets.append(triplet)
                                elif (triplet.labels - triplet_branch_containing_sink).pop() not in cycle_vertices:
                                    left_triplets.append(triplet)
                            elif len(triplet_branch_containing_sink) == 1:
                                branches_containing_other_triplet_labels = [branch for branch in left if (triplet.labels - triplet_branch_containing_sink).intersection(branch) != set()]
                                if len(branches_containing_other_triplet_labels) == 1: #the remaining nodes must be in the same branch for the sink to be in its own triplet branch
                                    if any(node in cycle_vertices for node in triplet.labels - triplet_branch_containing_sink) and triplet.type == r"1/2\3":
                                        left_triplets.append(triplet)
                                    elif not any(node in cycle_vertices for node in triplet.labels - triplet_branch_containing_sink):
                                        left_triplets.append(triplet)
                            else:
                                left_triplets.append(triplet)
                    if all(node in right_nodes for node in triplet):
                        triplet_branch_containing_sink = [branch for branch in triplet.branches if sink_and_descendants.intersection(branch) != set()]
                        if len(triplet_branch_containing_sink) != 1 or triplet.type in (r"1/2\3", r"1|2|3"):
                            right_triplets.append(triplet)
                        else:
                            triplet_branch_containing_sink = triplet_branch_containing_sink[0]
                            if len(triplet_branch_containing_sink.intersection(sink_and_descendants)) == 2:
                                if triplet.type in (r"1\2\3", r"1/2/3"):
                                    right_triplets.append(triplet)
                                elif (triplet.labels - triplet_branch_containing_sink).pop() not in cycle_vertices:
                                    right_triplets.append(triplet)
                            elif len(triplet_branch_containing_sink) == 1:
                                branches_containing_other_triplet_labels = [branch for branch in right if (triplet.labels - triplet_branch_containing_sink).intersection(branch) != set()]
                                if len(branches_containing_other_triplet_labels) == 1: #the remaining nodes must be in the same branch for the sink to be in its own triplet branch#aka check not of type 1|2|3
                                    if any(node in cycle_vertices for node in triplet.labels - triplet_branch_containing_sink) and triplet.type == r"1/2\3":
                                        right_triplets.append(triplet)
                                    elif not any(node in cycle_vertices for node in triplet.labels - triplet_branch_containing_sink):
                                        right_triplets.append(triplet)
                            else:
                                right_triplets.append(triplet)

                left_cycle = LevelOneNetworkReconstruction(
                    left_nodes,
                    left_triplets,
                    self.__numb_unlabelled_nodes,
                    self.__descendants if left_triplets == [] or left_nodes == sink_and_descendants else None,
                    self.__separation if left_triplets == [] or left_nodes == sink_and_descendants else None,
                    self.__sn_sets if left_triplets == [] or left_nodes == sink_and_descendants else None,
                )
                solved_left_cycle = left_cycle.reconstruct()
                self.__numb_unlabelled_nodes = left_cycle.__numb_unlabelled_nodes
                tree[root].update(solved_left_cycle)
                right_cycle = LevelOneNetworkReconstruction(
                    right_nodes,
                    right_triplets,
                    self.__numb_unlabelled_nodes,
                    self.__descendants if right_triplets == [] or right_nodes == sink_and_descendants else None,
                    self.__separation if right_triplets == [] or right_nodes == sink_and_descendants else None,
                    self.__sn_sets if right_triplets == [] or right_nodes == sink_and_descendants else None,
                )
                solved_right_cycle = right_cycle.reconstruct()
                self.__numb_unlabelled_nodes = right_cycle.__numb_unlabelled_nodes
                # Make sure the sink and descendants from the two different branches use the same unlabelled nodes
                def contains_sink(d, required_sink: set):
                    required_sink = set(required_sink)
                    for key, value in d.items():
                        if key in required_sink:
                            return True
                        if contains_sink(value, required_sink):
                            return True
                    if len(required_sink) == 0:
                        return True
                    else:
                        return False

                def find_sink_node(tree, cycle_path):
                    for key, value in tree.items():
                        if key in sink_and_descendants:
                            cycle_path.append(key)
                            return cycle_path
                        if key in cycle_vertices- sink_and_descendants or contains_sink(value, sink_and_descendants.copy()):
                            cycle_path.append(key)
                            return find_sink_node(value, cycle_path)
                    else:
                        try:
                            cycle_path.append(list(tree.keys())[0])
                        except IndexError:
                            pass
                        return cycle_path
                sink_node_left = find_sink_node(solved_left_cycle, [])
                sink_node_right = find_sink_node(solved_right_cycle, [])
                sub_cycle = solved_right_cycle
                for cycle_vertex in sink_node_right[:-1]:
                    sub_cycle = sub_cycle[cycle_vertex]
                sub_cycle.pop(sink_node_right[-1])
                sub_cycle[sink_node_left[-1]] = {}
                tree[root].update(solved_right_cycle)
        else:
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
                    subtree = LevelOneNetworkReconstruction(
                        list(branch),
                        [triplet for triplet in self.__triplets if all(node in branch for node in triplet)],
                        self.__numb_unlabelled_nodes,
                        self.__descendants,
                        self.__separation,
                        self.__sn_sets
                    )
                    sub_dict = subtree.reconstruct()
                    if subtree.__is_cycle:
                        for label1, label2 in combinations(subtree._labels, 2):
                            for other_label in set(self._labels) - branch:
                                if any(triplet.type == r"1|2|3" for triplet in self.__triplets if triplet.labels == {label1, label2, other_label}):
                                    sub_dict = list(sub_dict.values())[0]
                                    break
                            else:
                                continue
                            break
                    self.__numb_unlabelled_nodes = subtree.__numb_unlabelled_nodes
                tree[root].update(sub_dict)
        return tree