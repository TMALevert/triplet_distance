from random import randint, sample

import pytest

from results.network_generators import create_random_general_tree
from rooted_triplet_distance import GeneralTree, GeneralTreeReconstruction


@pytest.mark.parametrize(
    "triplets, labels",
    [
        (["A,B|C", "A|B|C"], ["A", "B", "C"]),
        (["A,B|C", "A|B|C", "A|B,C"], ["A", "B", "C"]),
        (["C,A|B", "A|B|C"], ["A", "B", "C"]),
        (["A,B|C", "B|C,E", "E|A|B"], ["A", "B", "C", "E"]),
    ],
)  # Todo: Add more test cases
def test_divide_in_branches_conflicting_triplets(triplets, labels):
    assert len(GeneralTreeReconstruction(labels, triplets)._GeneralTreeReconstruction__divide_in_branches("*")) == 1


@pytest.mark.parametrize(
    "triplets, labels, children, branches, root",
    [
        (["A,B|C"], ["A", "B", "C"], ["C"], [{"A", "B"}, {"C"}], "*"),
        (["A,B|C", "A|C,D"], ["A", "B", "C", "D"], [], [{"A", "B"}, {"C", "D"}], "*"),
        (["A,B|C", "A|C,D", "A|C|D"], ["A", "B", "C", "D"], [], [{"A", "B", "C", "D"}], "*"),
        (["A,B|C", "A|C,D", "A|C|D", "A|B,C"], ["A", "B", "C", "D"], [], [{"A", "B", "C", "D"}], "*"),
        (["A|B,C", "C,B|D", "A|C|D"], ["A", "B", "C", "D"], [], [{"A"}, {"B", "C"}, {"D"}], "*"),
        (["A|B,C", "C,B|D"], ["A", "B", "C", "D"], ["A", "D"], [{"A"}, {"B", "C"}, {"D"}], "*"),
        (["A|B,C", "C,B|D", "A|C|D"], ["A", "B", "C", "D"], ["B"], [{"A"}, {"B", "C"}, {"D"}], "*"),
        (["A|B,C", "C,B|D", "A|C|D", "A|B,D"], ["A", "B", "C", "D"], ["C"], [{"A", "B", "C", "D"}], "*"),
        (
            ["A|B,C", "C,B|D", "A|C|D"],
            ["A", "B", "C", "D", "E"],
            ["A", "D", "E"],
            [{"A"}, {"B", "C"}, {"E"}, {"D"}],
            "*",
        ),
        (["A|B|C", "C,D|E", "A,E|B"], ["A", "B", "C", "D", "E"], ["C"], [{"A", "E"}, {"D", "C"}, {"B"}], "*"),
        (
            ["A|B|C", "C,D|E", "A,E|B", "A,E|C", "A,E|D", "D,C|A", "D,C|B", "A|D|B", "E|D|B", "E|C|B"],
            ["A", "B", "C", "D", "E"],
            ["C"],
            [{"A", "E"}, {"D", "C"}, {"B"}],
            "*",
        ),
        ([], ["A", "B", "C"], ["A", "B", "C"], [{"A"}, {"B"}, {"C"}], "*"),
    ],
)  # Todo: Add more test cases
def test_divide_in_branches(triplets, labels, children, branches, root):
    obtained_branches = GeneralTreeReconstruction(labels, triplets)._GeneralTreeReconstruction__divide_in_branches(root)
    assert len(obtained_branches) == len(branches)
    for branch in branches:
        assert branch in obtained_branches
    for branch in obtained_branches:
        assert branch in branches


def test_find_possible_root():
    ...


def test_tree():
    tree_dict = {"A": {"B": {"C": {}, "D": {}}, "*_0": {"E": {}, "F": {}}}}
    tree = GeneralTree(tree_dict, ["A", "B", "C", "D", "E", "F"])
    triplets = tree.triplets
    reconstructed_tree = GeneralTreeReconstruction(["A", "B", "C", "D", "E", "F"], triplets).reconstruct()
    assert reconstructed_tree == tree_dict
    for triplet in GeneralTree(reconstructed_tree, ["A", "B", "C", "D", "E", "F"]).triplets:
        assert triplet in triplets


def run_test_random_tree():
    tree_dict, labels = create_random_general_tree(randint(3, 20))

    tree = GeneralTree(tree_dict, labels)

    triplets = tree.triplets
    reconstruction = GeneralTreeReconstruction(labels, triplets)
    obtained_tree = GeneralTree(reconstruction.reconstruct(), labels)

    assert tree == obtained_tree
    assert len(obtained_tree.triplets) == len(triplets)
    for triplet in obtained_tree.triplets:
        assert triplet in triplets


@pytest.mark.parametrize("_", range(100))
def test_random_tree_often(_):
    run_test_random_tree()


def run_test_random_tree_partial_triplets():
    tree_dict, labels = create_random_general_tree(randint(3, 20))
    tree = GeneralTree(tree_dict, labels)

    triplets = tree.triplets
    if len(triplets) > 0:
        triplet_subset = sample(triplets, randint(1, len(triplets)))
    else:
        triplet_subset = triplets
    reconstruction = GeneralTreeReconstruction(labels, triplet_subset)
    obtained_tree = GeneralTree(reconstruction.reconstruct(), labels)

    for triplet in triplet_subset:
        assert triplet in obtained_tree.triplets


@pytest.mark.parametrize("_", range(100))
def test_random_tree_partial_triplets(_):
    run_test_random_tree_partial_triplets()
