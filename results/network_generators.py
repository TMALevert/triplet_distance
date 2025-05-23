from random import sample, randint, random

from networkx.algorithms.components import biconnected_components
from networkx.classes import DiGraph
from networkx.generators.trees import random_labeled_rooted_tree
from phylox import DiNetwork
from phylox.generators.randomTC import generate_network_random_tree_child_sequence

def create_random_multifurcating_tree(n, a):
    undirected_tree = random_labeled_rooted_tree(n)
    root = undirected_tree.graph["root"]
    tree = DiGraph()
    tree.add_nodes_from(undirected_tree.nodes)
    tree_dict_final = {}

    def add_edge(u, tree_dict):
        tree_dict[str(u)] = {}
        neighbours = list(undirected_tree.neighbors(u))
        numb_of_children = 0
        for neighbour in neighbours:
            if not (neighbour, u) in tree.edges:
                numb_of_children += 1
                tree.add_edge(u, neighbour)
                tree_dict[str(u)][str(neighbour)] = add_edge(neighbour, tree_dict[str(u)])[str(neighbour)]
        if numb_of_children == 1:
            for _ in range(randint(1, a)):
                tree_dict[str(u)][str(max(tree.nodes) + 1)] = {}
                tree.add_edge(u, max(tree.nodes) + 1)
        return tree_dict

    add_edge(root, tree_dict_final)

    labels = [str(node) for node in tree.nodes]
    labels.remove(str(root))
    leaf_nodes = [str(node) for node in tree.nodes if tree.out_degree[node] == 0]
    final_labels = set(leaf_nodes)
    internal_labels = set(labels) - final_labels
    if len(internal_labels) > 0:
        final_labels = final_labels.union(set(sample(list(internal_labels), randint(1, len(internal_labels)))))
    return tree_dict_final, final_labels


def create_random_general_tree(n):
    undirected_tree = random_labeled_rooted_tree(n)
    root = undirected_tree.graph["root"]
    tree = DiGraph()
    tree.add_nodes_from(undirected_tree.nodes)
    tree_dict_final = {}

    def add_edge(u, tree_dict):
        tree_dict[str(u)] = {}
        neighbours = list(undirected_tree.neighbors(u))
        for neighbour in neighbours:
            if not (neighbour, u) in tree.edges:
                tree.add_edge(u, neighbour)
                tree_dict[str(u)][str(neighbour)] = add_edge(neighbour, tree_dict[str(u)])[str(neighbour)]
        return tree_dict

    add_edge(root, tree_dict_final)

    labels = [str(node) for node in tree.nodes]
    leaf_nodes = [str(node) for node in tree.nodes if tree.out_degree(node) == 0]
    for node in tree.nodes:
        if tree.out_degree(node) == 1:
            leaf_nodes.append(str(node))
            child = list(tree.successors(node))[0]
            leaf_nodes.append(str(child))
    final_labels = set(leaf_nodes)
    internal_labels = set(labels) - final_labels
    if len(internal_labels) > 0:
        final_labels = final_labels.union(set(sample(list(internal_labels), randint(max(1, 3-len(final_labels)), len(internal_labels)))))
    return tree_dict_final, final_labels


def create_random_level_1_network(n, n_reticulations):
    final_network = DiNetwork()
    for i_subnetwork in range(n_reticulations):
        current_numb_leaves = len([node for node in final_network.nodes if final_network.out_degree(node) == 0])
        if i_subnetwork != n_reticulations - 1:
            random_network = generate_network_random_tree_child_sequence(
                randint(2, max(n - current_numb_leaves - 2 * (n_reticulations - i_subnetwork - 1), 3)),
                1,
                label_leaves=False,
            )
        else:
            random_network = generate_network_random_tree_child_sequence(
                max(n - current_numb_leaves, 2), 1, label_leaves=False
            )
        nodes_to_positive_integers = {node: i for i, node in enumerate(random_network.nodes)}
        network = DiNetwork()
        network.add_edges_from(
            [
                (nodes_to_positive_integers[edge[0]], nodes_to_positive_integers[edge[1]])
                for edge in random_network.edges
            ]
        )
        root = [node for node in network.nodes if network.in_degree(node) == 0][0]
        if random() > 0.5:
            network.remove_node(root)
            root = [node for node in network.nodes if network.in_degree(node) == 0]
            assert len(root) == 1
            root = root[0]
        if len(final_network.nodes) == 0:
            final_network = network
        else:
            highest_node = max(final_network.nodes) + 1
            root += highest_node
            random_node = sample(list(final_network.nodes), 1)[0]
            final_network.add_edges_from([(edge[0] + highest_node, edge[1] + highest_node) for edge in network.edges])
            if random() > 0.5:
                final_network.add_edge(random_node, root)
            else:
                final_network.remove_node(root)
                final_network.add_edges_from(
                    [(random_node, root_child + highest_node) for root_child in network.successors(root - highest_node)]
                )

    final_leaves = [node for node in final_network.nodes if final_network.out_degree(node) == 0]
    for _ in range(n - len(final_leaves)):
        random_node = sample([node for node in final_network.nodes if final_network.out_degree(node) != 0], 1)[0]
        final_network.add_edge(random_node, max(final_network.nodes) + 1)
        final_leaves.append(max(final_network.nodes))

    labels = [node for node in final_leaves]
    for node in final_network.nodes:
        if final_network.out_degree(node) == 1:
            labels.append(node)
            child = list(final_network.successors(node))[0]
            labels.append(child)
    for cycle in biconnected_components(final_network.to_undirected()):
        if len(cycle) == 4 or len(cycle) == 3:
            internal_cycle_nodes = [
                node
                for node in cycle
                if len(set(final_network.successors(node)).intersection(set(cycle))) != 2
                and final_network.in_degree(node) != 2
            ]
            labels.append(sample(internal_cycle_nodes, 1)[0])
        for node in cycle:
            if final_network.in_degree(node) == 2:
                labels.append(node)
    final_labels = set(labels)
    internal_labels = set(final_network.nodes) - final_labels
    if len(internal_labels) > 0:
        final_labels = final_labels.union(
            set(sample(list(internal_labels), randint(max(0, 3-len(final_labels)), int(0.8 * len(internal_labels)))))
        )
    final_labels = [str(label) for label in final_labels]

    tree_dict_final = {}

    def add_edge(u, tree_dict):
        tree_dict[str(u)] = {}
        for child in final_network.successors(u):
            tree_dict[str(u)][str(child)] = add_edge(child, tree_dict[str(u)])[str(child)]
        return tree_dict

    add_edge([node for node in final_network.nodes if final_network.in_degree(node) == 0][0], tree_dict_final)
    return tree_dict_final, final_labels