from random import randint, sample

import pytest
from networkx import random_labeled_rooted_tree, DiGraph

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


def create_random_tree(n):
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
        final_labels = final_labels.union(set(sample(list(internal_labels), randint(1, len(internal_labels)))))
    return tree_dict_final, final_labels


def run_test_random_tree():
    tree_dict, labels = create_random_tree(randint(3, 20))

    tree = GeneralTree(tree_dict, labels)

    triplets = tree.triplets
    reconstruction = GeneralTreeReconstruction(labels, triplets)
    obtained_tree = GeneralTree(reconstruction.reconstruct(), labels)

    assert tree == obtained_tree
    assert len(obtained_tree.triplets) == len(triplets)
    for triplet in obtained_tree.triplets:
        assert triplet in triplets


@pytest.mark.parametrize("_", range(3000))
def test_random_tree_often(_):
    # from cProfile import Profile
    # with Profile() as profile:
    #     for _ in range(100):
    #         run_test_random_tree()
    # profile.dump_stats("test_random_tree_often.prof")
    run_test_random_tree()


def run_test_random_tree_partial_triplets():
    tree_dict, labels = create_random_tree(randint(3, 20))
    tree = GeneralTree(tree_dict, labels)

    triplets = tree.triplets
    if len(triplets) > 0:
        triplet_subset = sample(triplets, randint(1, len(triplets)))
    else:
        triplet_subset = triplets
    reconstruction = GeneralTreeReconstruction(labels, triplet_subset)
    obtained_tree = GeneralTree(reconstruction.reconstruct(), labels)

    try:
        for triplet in triplet_subset:
            assert triplet in obtained_tree.triplets
    except AssertionError:
        for triplet in triplet_subset:
            print(triplet)
        tree.visualize()
        obtained_tree.visualize()
        reconstruction.reconstruct()
        raise AssertionError


@pytest.mark.parametrize("_", range(3000))
def test_random_tree_partial_triplets(_):
    # from cProfile import Profile
    # with Profile() as profile:
    #     for _ in range(20):
    #         run_test_random_tree_partial_triplets()
    # profile.dump_stats("test_random_tree_partial_triplets.prof")
    run_test_random_tree_partial_triplets()
