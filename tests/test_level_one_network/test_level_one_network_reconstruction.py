import os

import pytest
from networkx import random_labeled_rooted_tree, draw_networkx, spring_layout, arf_layout, biconnected_component_edges, NetworkXError, biconnected_components
from phylox.generators.randomTC.random_tc_network import generate_network_random_tree_child_sequence
from phylox import DiNetwork
from phylox.generators.lgt.base import generate_network_lgt
from random import randint, sample, random
import matplotlib.pyplot as plt

from rooted_triplet_distance.level_one_network import Network, NetworkReconstruction

def create_random_level_1_network(n, n_reticulations):
    final_network = DiNetwork()
    for i_subnetwork in range(n_reticulations):
        if i_subnetwork != n_reticulations - 1 and n-len([node for node in final_network.nodes if final_network.out_degree(node) == 0]) - 2 * (n_reticulations - i_subnetwork - 1) != 2:
            random_network = generate_network_random_tree_child_sequence(randint(2, max(n-len([node for node in final_network.nodes if final_network.out_degree(node) == 0]) - 2 * (n_reticulations - i_subnetwork - 1), 3)), 1, label_leaves=False)
        else:
            random_network = generate_network_random_tree_child_sequence(n-len([node for node in final_network.nodes if final_network.out_degree(node) == 0]), 1, label_leaves=False)
        nodes_to_positive_integers = {node: i for i, node in enumerate(random_network.nodes)}
        network = DiNetwork()
        network.add_edges_from([(nodes_to_positive_integers[edge[0]], nodes_to_positive_integers[edge[1]]) for edge in random_network.edges])
        cycle_edges: list[tuple] = sorted(list(biconnected_component_edges(network.to_undirected())), key=len)[-1]
        if len(cycle_edges) == 3:
            for _ in range(randint(1, 4)):
                random_edge = sample(cycle_edges, 1)[0]
                cycle_edges.remove(random_edge)
                try:
                    network.remove_edge(*random_edge)
                except NetworkXError:
                    random_edge = (random_edge[1], random_edge[0])
                    network.remove_edge(*random_edge)
                new_node = network.number_of_nodes()+1
                network.add_node(new_node)
                network.add_edges_from([(random_edge[0], new_node), (new_node, random_edge[1])])
                cycle_edges.extend([(random_edge[0], new_node), (new_node, random_edge[1])])
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
                final_network.add_edges_from([(random_node, root_child + highest_node) for root_child in network.successors(root-highest_node)])

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
        if len(cycle) == 4:
            internal_cycle_nodes = [node for node in cycle if len(set(final_network.successors(node)).intersection(set(cycle))) != 2 and final_network.in_degree(node) != 2]
            labels.append(sample(internal_cycle_nodes, 1)[0])
    final_labels = set(labels)
    internal_labels = set(final_network.nodes) - final_labels
    if len(internal_labels) > 0:
        final_labels = final_labels.union(set(sample(list(internal_labels), randint(0, int(0.8 * len(internal_labels))))))
    final_labels = [str(label) for label in final_labels]

    tree_dict_final = {}
    def add_edge(u, tree_dict):
        tree_dict[str(u)] = {}
        for child in final_network.successors(u):
            tree_dict[str(u)][str(child)] = add_edge(child, tree_dict[str(u)])[str(child)]
        return tree_dict
    add_edge([node for node in final_network.nodes if final_network.in_degree(node) == 0][0], tree_dict_final)
    return tree_dict_final, final_labels

def test_find_possible_roots(network1, network2):
    labels, triplets = network1.labels, network1.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert ["p"] == reconstruction._LevelOneNetworkReconstruction__find_possible_roots()
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert ["p"] == reconstruction._LevelOneNetworkReconstruction__find_possible_roots()

def test_one_branch_when_cycle_with_labelled_root(network1, network2):
    labels, triplets = network1.labels, network1.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert len(reconstruction._LevelOneNetworkReconstruction__divide_in_branches("p")) == 1
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert len(reconstruction._LevelOneNetworkReconstruction__divide_in_branches("p")) == 1

def test_find_sink_of_cycle_with_labelled_source(network1, network2):
    labels, triplets = network1.labels, network1.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("p") == {"d", "c"}
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("p") == {"b", "e"}

def test_resolve_cycle(network1, network2):
    labels, triplets = network1.labels, network1.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    print(reconstruction._LevelOneNetworkReconstruction__resolve_cycle("p"))
    network1.visualize(show=False, save=True, save_name="network1.png")
    network2.visualize(show=False, save=True, save_name="network2.png")
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    print(reconstruction._LevelOneNetworkReconstruction__resolve_cycle("p"))

def test_reconstruct(network1, network2):
    labels, triplets = network1.labels, network1.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert Network(reconstruction.reconstruct(), labels) == network1
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert Network(reconstruction.reconstruct(), labels) == network2

def test_find_possible_roots_no_labelled_root(network1_no_root, network2_no_root):
    labels, triplets = network1_no_root.labels, network1_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert [] == reconstruction._LevelOneNetworkReconstruction__find_possible_roots()
    labels, triplets = network2_no_root.labels, network2_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert [] == reconstruction._LevelOneNetworkReconstruction__find_possible_roots()

def test_one_branch_when_cycle_no_labelled_root(network1_no_root, network2_no_root):
    labels, triplets = network1_no_root.labels, network1_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert len(reconstruction._LevelOneNetworkReconstruction__divide_in_branches("*")) == 1
    labels, triplets = network2_no_root.labels, network2_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert len(reconstruction._LevelOneNetworkReconstruction__divide_in_branches("*")) == 1

def test_find_sink_of_cycle_no_labelled_source(network1_no_root, network2_no_root):
    labels, triplets = network1_no_root.labels, network1_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("*") == {"d", "c"}
    labels, triplets = network2_no_root.labels, network2_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("*") == {"b", "e"}

def run_test_random_network(_):
    tree_dict, labels = create_random_level_1_network(randint(3, 6), 1)

    tree = Network(tree_dict, labels)
    # tree.visualize(show=False, save=True, save_name=f"random_network_{_}")
    tree.visualize(show=True)
    triplets = tree.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    obtained_tree = Network(reconstruction.reconstruct(), labels)
    obtained_tree.visualize(show=True)
    try:
        # obtained_tree.visualize(show=False, save=True, save_name=f"obtained_network_{_}")
        # os.remove(f"obtained_network_{_}.png")
        assert tree == obtained_tree
        assert len(obtained_tree.triplets) == len(triplets)
        for triplet in obtained_tree.triplets:
            assert triplet in triplets
    except AssertionError:
        obtained_tree.visualize(show=False, save=True, save_name=f"obtained_network_{_}")
        tree.visualize(show=False, save=True, save_name=f"random_network_{_}_{tree_dict}_assert_error")
        reconstruction.reconstruct()
        raise AssertionError
    except:
        tree.visualize(show=False, save=True, save_name=f"random_network_{_}_{tree_dict}")
        reconstruction.reconstruct()
        raise AssertionError


@pytest.mark.parametrize("_", range(300))
def test_random_network_often(_):
    # from cProfile import Profile
    # with Profile() as profile:
    #     for _ in range(100):
    #         run_test_random_tree()
    # profile.dump_stats("test_random_tree_often.prof")
    run_test_random_network(_)