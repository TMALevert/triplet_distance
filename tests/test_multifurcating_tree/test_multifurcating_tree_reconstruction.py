import timeit
from random import sample, randint

import numpy as np
import pytest
from matplotlib import pyplot as plt, scale
from networkx import random_labeled_rooted_tree, DiGraph

from rooted_triplet_distance import MultifurcatingTreeReconstruction, MultifurcatingTree, GeneralTreeReconstruction, \
    GeneralTree


def create_random_tree(n, a):
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


def test_D_sets():
    tree_dict = {
        "root": {
            "*1": {"A": {"B": {}, "C": {}}, "D": {}},
            "*2": {"F": {"G": {}, "H": {}, "E": {}}, "I": {}, "J": {}},
        }
    }
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tree = MultifurcatingTree(tree_dict, labels)
    reconstruction = MultifurcatingTreeReconstruction(labels, tree.triplets)
    D_sets = {
        "A": ["B", "C"],
        "B": ["A"],
        "C": ["A"],
        "D": [],
        "E": ["F"],
        "F": ["G", "H", "E"],
        "G": ["F"],
        "H": ["F"],
        "I": [],
        "J": [],
    }
    for node in labels:
        assert len(reconstruction._MultifurcatingTreeReconstruction__D_sets[node]) == len(D_sets[node])
        for other in D_sets[node]:
            assert other in reconstruction._MultifurcatingTreeReconstruction__D_sets[node]
        for other in reconstruction._MultifurcatingTreeReconstruction__D_sets[node]:
            assert other in D_sets[node]


def test_find_children_of_root():
    tree = MultifurcatingTree({"root": {"*1": {"A": {"B": {}, "C": {}}, "D": {}}, "E": {}}}, ["A", "B", "C", "D", "E"])
    triplets = tree.triplets
    reconstruction = MultifurcatingTreeReconstruction(tree.labels, triplets)
    children = reconstruction._MultifurcatingTreeReconstruction__children_of_root()
    assert children == {"E"}


def run_test_incomplete_triplet_set():
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Z"]
    triplets = [
        "Z|F|G",
        "Z|I,C",
        "Z|A,B",
        "Z|C,D",
        "Z|I,E",
        "Z|G,H",
        "Z,G|A",
        "A,B|C",
        "A,B|D",
        "A,B|E",
        "A,B|F",
        "A,B|G",
        "A,B|H",
        "A,B|J",
        "A|C,D",
        "A|C|E",
        "A,C|F",
        "A,C|G",
        "A,C|H",
        "A,C|J",
        "A|D|E",
        "A,D|F",
        "A,D|G",
        "A,D|H",
        "A,D|J",
        "A,E|F",
        "A,E|G",
        "A,E|H",
        "A,E|J",
        "A|F,G",
        "A|F,H",
        "A|G,H",
        "B|C,D",
        "B|C|E",
        "B,C|F",
        "B,C|G",
        "B,C|H",
        "B,C|J",
        "B|D|E",
        "B,D|F",
        "B,D|G",
        "B,D|H",
        "B,D|J",
        "B,E|F",
        "B,E|G",
        "B,E|H",
        "B,E|J",
        "B|F,G",
        "B|F,H",
        "B|G,H",
        "C,D|E",
        "C,D|F",
        "C,D|G",
        "C,D|H",
        "C,D|I",
        "C,D|J",
        "C,E|F",
        "C,E|G",
        "C,E|H",
        "C|E|I",
        "C,E|J",
        "C|F,G",
        "C|F,H",
        "C,I|F",
        "C|G,H",
        "C,I|G",
        "C,I|H",
        "C,I|J",
        "D,E|F",
        "D,E|G",
        "D,E|H",
        "D|E|I",
        "D,E|J",
        "D|G,F",
        "D|F,H",
        "D,I|F",
        "D|G,H",
        "D,I|G",
        "D,I|H",
        "D,I|J",
        "E|F,G",
        "E|F,H",
        "E,I|F",
        "E|G,H",
        "E,I|G",
        "E,I|H",
        "E,I|J",
        "F|G,H",
        "F,G|I",
        "F,H|I",
        "G,H|I",
    ]
    triplet_subset = sample(triplets, randint(1, len(triplets)))
    tree = MultifurcatingTreeReconstruction(labels, triplet_subset).reconstruct()
    obtained_triplets = MultifurcatingTree(tree, labels).triplets
    for triplet in triplet_subset:
        assert triplet in obtained_triplets


@pytest.mark.parametrize("_", range(100))
def test_incomplete_triplet_set_often(_):
    run_test_incomplete_triplet_set()


@pytest.mark.parametrize(
    "triplets, labels, children, branches",
    [
        (["A,B|C"], ["A", "B", "C"], ["C"], [{"A", "B"}, {"C"}]),
        (["A,B|C", "A|C,D"], ["A", "B", "C", "D"], [], [{"A", "B"}, {"C", "D"}]),
        (["A,B|C", "A|C,D", "A|C|D"], ["A", "B", "C", "D"], [], [{"A", "B", "C", "D"}]),
        (["A,B|C", "A|C,D", "A|C|D", "A|B,C"], ["A", "B", "C", "D"], [], [{"A", "B", "C", "D"}]),
        (["A|B,C", "C,B|D", "A|C|D"], ["A", "B", "C", "D"], [], [{"A"}, {"B", "C"}, {"D"}]),
        (["A|B,C", "C,B|D"], ["A", "B", "C", "D"], ["A", "D"], [{"A"}, {"B", "C"}, {"D"}]),
        (["A|B,C", "C,B|D", "A|C|D"], ["A", "B", "C", "D"], ["B"], [{"A"}, {"B", "C"}, {"D"}]),
        (["A|B,C", "C,B|D", "A|C|D", "A|B,D"], ["A", "B", "C", "D"], ["C"], [{"A", "B", "C", "D"}]),
        (["A|B,C", "C,B|D", "A|C|D"], ["A", "B", "C", "D", "E"], ["A", "D", "E"], [{"A"}, {"B", "C", "E"}, {"D"}]),
        (["A|B|C", "C,D|E", "A,E|B"], ["A", "B", "C", "D", "E"], ["C"], [{"A", "E"}, {"D", "C"}, {"B"}]),
        (
            ["A|B|C", "C,D|E", "A,E|B", "A,E|C", "A,E|D", "D,C|A", "D,C|B", "A|D|B", "E|D|B", "E|C|B"],
            ["A", "B", "C", "D", "E"],
            ["C"],
            [{"A", "E"}, {"D", "C"}, {"B"}],
        ),
        ([], ["A", "B", "C"], ["A", "B", "C"], [{"A"}, {"B"}, {"C"}]),
    ],
)
def test_divide_in_branches(triplets, labels, children, branches):
    obtained_branches = MultifurcatingTreeReconstruction(
        labels, triplets
    )._MultifurcatingTreeReconstruction__divide_in_branches(children=children)
    assert len(obtained_branches) == len(branches)
    for branch in branches:
        assert branch in obtained_branches
    for branch in obtained_branches:
        assert branch in branches


@pytest.mark.parametrize(
    "triplets, labels",
    [
        (["A,B|C", "A|B|C"], ["A", "B", "C"]),
        (["A,B|C", "A|B|C", "A|B,C"], ["A", "B", "C"]),
        (["C,A|B", "A|B|C"], ["A", "B", "C"]),
        (["A,B|C", "B|C,E", "E|A|B"], ["A", "B", "C", "E"]),
    ],
)
def test_divide_in_branches_conflicting_triplets(triplets, labels):
    assert (
        len(MultifurcatingTreeReconstruction(labels, triplets)._MultifurcatingTreeReconstruction__divide_in_branches([]))
        == 1
    )


def run_test_random_tree():
    tree_dict, labels = create_random_tree(randint(3, 20), 4)

    tree = MultifurcatingTree(tree_dict, labels)

    triplets = tree.triplets
    reconstruction = MultifurcatingTreeReconstruction(labels, triplets)
    obtained_tree = MultifurcatingTree(reconstruction.reconstruct(), labels)

    assert tree == obtained_tree
    assert len(obtained_tree.triplets) == len(triplets)
    for triplet in obtained_tree.triplets:
        assert triplet in triplets


@pytest.mark.parametrize("_", range(3000))
def test_random_tree_often(_):
    # from cProfile import Profile
    # with Profile() as profile:
    #     for _ in range(20):
    #         run_test_random_tree()
    # profile.dump_stats("test_random_multifurcating_tree_often.prof")
    run_test_random_tree()


def run_test_random_tree_partial_triplets():
    tree_dict, labels = create_random_tree(randint(3, 20), 4)

    tree = MultifurcatingTree(tree_dict, labels)

    triplets = tree.triplets
    if len(triplets) > 0:
        triplet_subset = sample(triplets, randint(1, len(triplets)))
    else:
        triplet_subset = triplets
    reconstruction = MultifurcatingTreeReconstruction(labels, triplet_subset)
    obtained_tree = MultifurcatingTree(reconstruction.reconstruct(), labels)
    try:
        for triplet in triplet_subset:
            assert triplet in obtained_tree.triplets
    except AssertionError as e:
        print(triplet_subset)
        print(tree_dict)
        print(labels)
        raise e


@pytest.mark.parametrize("_", range(3000))
def test_random_tree_partial_triplets(_):
    # from cProfile import Profile
    # with Profile() as profile:
    #     for _ in range(20):
    #         run_test_random_tree_partial_triplets()
    # profile.dump_stats("test_random_multifurcating_tree_partial_triplets.prof")
    run_test_random_tree_partial_triplets()


def test_compare_to_general_tree_reconstruction():
    n = 1000
    size = randint(15,20)
    # size = 8
    print(size)
    times_multifurcating: list[float] = []
    times_general: list[float] = []
    times_per_triplet_multifurcating: list[float] = []
    times_per_triplet_general: list[float] = []
    numb_multifurcating_triplets: list[int] = []
    numb_general_triplets: list[int] = []
    numb_labels: list[int] = []
    for _ in range(n):
        tree_dict, labels = create_random_tree(size, 4)

        tree = MultifurcatingTree(tree_dict, labels)
        general_tree = GeneralTree(tree_dict, labels)

        triplets = tree.triplets
        general_triplets = general_tree.triplets
        numb_multifurcating_triplets.append(len(triplets))
        numb_general_triplets.append(len(general_triplets))
        numb_labels.append(len(labels))

        multifurcating_reconstruction = MultifurcatingTreeReconstruction(labels, triplets)
        obtained_tree = MultifurcatingTree(multifurcating_reconstruction.reconstruct(), labels)

        assert tree == obtained_tree
        # assert len(obtained_tree.triplets) == len(triplets)
        # for triplet in obtained_tree.triplets:
        #     assert triplet in triplets

        times_multifurcating.append(timeit.timeit(lambda: multifurcating_reconstruction.reconstruct(), number=5)/5)
        times_per_triplet_multifurcating.append(times_multifurcating[-1] / len(triplets))

        general_reconstruction = GeneralTreeReconstruction(labels, general_triplets)
        obtained_tree = MultifurcatingTree(general_reconstruction.reconstruct(), labels)

        assert tree == obtained_tree
        # assert len(obtained_tree.triplets) == len(triplets)
        # for triplet in obtained_tree.triplets:
        #     assert triplet in triplets

        times_general.append(timeit.timeit(lambda: general_reconstruction.reconstruct(), number=5) / 5)
        times_per_triplet_general.append(times_general[-1] / len(general_triplets))

    plt.figure()
    plt.hist(numb_multifurcating_triplets, alpha=0.5, label="Multifurcating triplets")
    plt.hist(numb_general_triplets, alpha=0.5, label="General triplets")
    plt.ylabel("Number of trees")
    plt.xlabel("Number of triplets")
    plt.legend()
    plt.title(f"Number of triplets in the triplet subset for {n} random trees")
    plt.savefig(f"histogram_numb_triplets_{size}.png")

    plt.figure()
    # Get numb of triplets per numb of labels
    plt.scatter(numb_labels, numb_multifurcating_triplets, alpha=0.5, label="Multifurcating triplets")
    plt.scatter(numb_labels, numb_general_triplets, alpha=0.5, label="General triplets")
    plt.plot(np.linspace(min(numb_labels), max(numb_labels), 100), 1/6*np.linspace(min(numb_labels), max(numb_labels), 100)**3, label=r"$\frac{n^3}{3!}$")
    plt.xlabel("Number of labels")
    plt.ylabel("Number of triplets")
    plt.legend()
    plt.title("Number of triplets in the triplet subset per number of labels")
    plt.savefig("scatter_numb_triplets_per_numb_labels.png")

    # Create a plot showing the average time and the standard deviation for each method on the y axis and the number of labels on the x axis
    plt.figure()
    times_per_numb_labels_multi = []
    unique_numb_labels = list(set(numb_labels))
    for numb in unique_numb_labels:
        times_per_numb_labels_multi.append(
            [times_multifurcating[i] for i in range(n) if numb_labels[i] == numb]
        )

    times_per_numb_labels_general = []
    for numb in unique_numb_labels:
        times_per_numb_labels_general.append(
            [times_general[i] for i in range(n) if numb_labels[i] == numb]
        )
    plt.errorbar(
        unique_numb_labels,
        [np.mean(times) for times in times_per_numb_labels_multi],
        [np.std(times) for times in times_per_numb_labels_multi],
        label="Multifurcating"
    )
    plt.errorbar(
        unique_numb_labels,
        [np.mean(times) for times in times_per_numb_labels_general],
        [np.std(times) for times in times_per_numb_labels_general],
        label="General"
    )
    plt.xlabel("Number of labels")
    plt.ylabel("Average time [s]")
    plt.legend()
    plt.title("Average time per number of labels")
    plt.savefig("errorbar_average_time_per_numb_labels.png")

    plt.figure()
    plt.yscale(scale.FuncScale(None, functions=(lambda x: np.sign(x)*abs(x)**(4/7), lambda x: np.sign(x)*abs(x)**(7/4))))
    domain = np.linspace(min(min(numb_general_triplets), min(numb_multifurcating_triplets)), max(max(numb_general_triplets), max(numb_multifurcating_triplets)), 100)
    plt.plot(domain, (max(max(times_multifurcating), max(times_general))/max(domain**(7/4)))*domain**(7/4), label=r"~$n^{\frac{7}{4}}$")
    plt.scatter(numb_multifurcating_triplets, times_multifurcating, alpha=0.5, label="Multifurcating")
    plt.scatter(numb_general_triplets, times_general, alpha=0.5, label="General")
    plt.xlabel("Number of triplets")
    plt.ylabel("Average time [s]")
    plt.legend()
    plt.title("Time per number of triplets")
    plt.savefig("scatter_time_per_numb_triplets.png")

    # print(f"Multifurcating average [s]: {sum(times_multifurcating) / n}")
    # print(f"General average [s]: {sum(times_general) / n}")
    # print(f"Percentage of time saved using general: {100 * (1 - sum(times_general) / sum(times_multifurcating))}%")
    # print(f"Percentage of times general is faster: {100 * (sum(np.array(times_general) <= np.array(times_multifurcating))) / n}%")
    # print(f"Multifurcating average per triplet [s]: {sum(times_per_triplet_multifurcating) / n}")
    # print(f"General average per triplet [s]: {sum(times_per_triplet_general) / n}")
