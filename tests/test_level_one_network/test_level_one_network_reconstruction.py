import pytest
from random import randint

from results.network_generators import create_random_level_1_network
from rooted_triplet_distance.level_one_network import Network, NetworkReconstruction


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
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("p") == [{"d", "c"}]
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("p") == [{"b", "e"}]


def test_resolve_cycle(network1, network2):
    labels, triplets = network1.labels, network1.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    (
        cycle_branches,
        sink_and_descendants,
        internal_cycle_vertices,
    ) = reconstruction._LevelOneNetworkReconstruction__resolve_cycle({"d", "c"}, "p")
    assert all(cycle_branch in [{"f"}, {"1", "a"}, {"g", "h"}, {"e", "b"}] for cycle_branch in cycle_branches)
    assert sink_and_descendants == {"c", "d"}
    assert internal_cycle_vertices == {"d", "1"}
    network1.visualize(show=False, save=True, save_name="network1.png")
    network2.visualize(show=False, save=True, save_name="network2.png")
    labels, triplets = network2.labels, network2.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    (
        cycle_branches,
        sink_and_descendants,
        internal_cycle_vertices,
    ) = reconstruction._LevelOneNetworkReconstruction__resolve_cycle({"b", "e"}, "p")
    assert all(cycle_branch in [{"f"}, {"1", "a"}, {"g", "h"}, {"c", "d"}] for cycle_branch in cycle_branches)
    assert sink_and_descendants == {"e", "b"}
    assert internal_cycle_vertices == {"b", "1"}


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
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("*") == [{"d", "c"}]
    labels, triplets = network2_no_root.labels, network2_no_root.triplets
    reconstruction = NetworkReconstruction(labels, triplets)
    assert reconstruction._LevelOneNetworkReconstruction__find_sink_of_cycle("*") == [{"b", "e"}]


def run_test_random_network(_):
    tree_dict, labels = create_random_level_1_network(randint(6, 12), randint(1, 3))
    tree = Network(tree_dict, labels)
    triplets = tree.triplets

    reconstruction = NetworkReconstruction(labels, triplets)
    try:
        reconstructed_dict = reconstruction.reconstruct()
    except RecursionError:
        reconstructed_dict = reconstruction.reconstruct()
    obtained_tree = Network(reconstructed_dict, labels)
    assert tree == obtained_tree
    assert len(obtained_tree.triplets) == len(triplets)
    for triplet in obtained_tree.triplets:
        assert triplet in triplets


@pytest.mark.parametrize("_", range(300))
def test_random_network_often(_):
    run_test_random_network(_)
