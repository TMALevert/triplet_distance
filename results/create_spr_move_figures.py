import os
from math import ceil
from random import sample, randint

import numpy as np
from networkx import balanced_tree, descendants
from networkx.algorithms.dag import dag_longest_path
from networkx.classes import DiGraph
import matplotlib.pyplot as plt

from rooted_triplet_distance.__abstract import _get_tree_dict
from rooted_triplet_distance.general_tree import Tree


def create_balanced_tree(r: int, h: int, fully_labelled=False) -> Tree:
    """
    Create a balanced tree with r branches and height h.
    """
    tree = balanced_tree(r, h, DiGraph)
    if fully_labelled:
        labels = set(tree.nodes)
    else:
        labels = {node for node in tree.nodes if tree.out_degree(node) == 0}
        remaining_labels = set(tree.nodes) - labels
        labels = labels.union(sample(list(remaining_labels), ceil(len(remaining_labels) / 2)))
    tree_dict = _get_tree_dict(tree)
    return Tree(tree_dict, labels)


def create_catepillar_tree(r: int, h: int, fully_labelled=False) -> Tree:
    """
    Create a caterpillar tree with r branches and height h.
    """
    tree = DiGraph()
    for i in range(h):
        tree.add_node(f"{i}")
        if i > 0:
            tree.add_edge(f"{i-1}", f"{i}")
        for j in range(r - 1):
            tree.add_edge(f"{i}", f"{i}-{j}")
        if i == h - 1:
            tree.add_edge(f"{i}", f"{i}-{r-1}")
    if fully_labelled:
        labels = set(tree.nodes)
    else:
        labels = {node for node in tree.nodes if tree.out_degree(node) == 0}
        remaining_labels = set(tree.nodes) - labels
        labels = labels.union(sample(list(remaining_labels), ceil(len(remaining_labels) / 2)))
    tree_dict = _get_tree_dict(tree)
    return Tree(tree_dict, labels)


def plot_spr_moves_to_root(tree: Tree, save_name: str, numb_skips: int = 0) -> None:
    plt.figure()
    node = dag_longest_path(tree._tree)[-1]
    while tree._tree.in_degree(node) > 0:
        points = []
        size_of_subtree = len(descendants(tree._tree, node).union({node}).intersection(tree.labels))
        parent = list(tree._tree.predecessors(node))[0]
        while len(list(tree._tree.predecessors(parent))) >= 1:
            insert_edge = (list(tree._tree.predecessors(parent))[0], parent)
            new_dict, distance = tree.perform_spr_move(node, insert_edge=insert_edge)
            new_tree = Tree(new_dict, tree.labels)
            parent = insert_edge[0]
            if new_tree - tree == 0:
                continue
            points.append((distance, new_tree - tree))
        node = sample(list(tree._tree.predecessors(node)), 1)[0]
        for _ in range(numb_skips):
            if tree._tree.in_degree(node) > 0:
                node = sample(list(tree._tree.predecessors(node)), 1)[0]
        if len(points) == 0:
            continue
        plt.plot(*zip(*points), marker="o", label=f"$n={size_of_subtree}$")
    plt.xlabel("SPR move length")
    plt.ylabel("Triplet distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def plot_all_spr_moves(tree: Tree, save_name: str) -> None:
    """
    Create figures comparing the SPR distance and triplet distance after performing SPR moves.
    """
    plt.figure()
    node = dag_longest_path(tree._tree)[-1]
    while tree._tree.in_degree(node) > 0:
        points = []
        size_of_subtree = len(descendants(tree._tree, node).union({node}).intersection(tree.labels))
        for insert_edge in tree._tree.edges():
            try:
                new_dict, distance = tree.perform_spr_move(node, insert_edge=insert_edge)
            except ValueError:
                continue
            new_tree = Tree(new_dict, tree.labels)
            if new_tree - tree == 0:
                continue
            points.append((distance, new_tree - tree))
        node = sample(list(tree._tree.predecessors(node)), 1)[0]
        if tree._tree.in_degree(node) > 0:
            node = sample(list(tree._tree.predecessors(node)), 1)[0]
        if len(points) == 0:
            continue
        plt.scatter(*zip(*points), label=f"$n={size_of_subtree}$")
        poly = np.polyfit(*zip(*points), 1)
        x = np.linspace(min(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[0])[0], 100)
        line = plt.plot(x, np.polyval(poly, x), linestyle="--")[0]
        colour = line.get_color() # Get the color of the line for the legend
        plt.plot([], [], marker="o", linestyle="--", color=colour, label=f"$n={size_of_subtree}$")
    plt.xlabel("SPR move length")
    plt.ylabel("Triplet distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def plot_caterpillar_spr_triplet_distance(r: int, h: int, fully_labelled=False) -> None:
    """
    Create figures comparing the SPR distance and triplet distance after performing SPR moves.
    """
    tree = create_catepillar_tree(r, h, fully_labelled=fully_labelled)
    plot_spr_moves_to_root(
        tree,
        f"spr_move_figures/{'fully_labelled_' if fully_labelled else ''}caterpillar_tree_spr_move_r{r}_h{h}.png",
        numb_skips=1,
    )
    plot_all_spr_moves(
        tree,
        f"spr_move_figures/{'fully_labelled_' if fully_labelled else ''}caterpillar_tree_all_spr_move_r{r}_h{h}.png",
    )


def plot_fully_balanced_tree_spr_triplet_distance(r: int, h: int, fully_labelled=False) -> None:
    """
    Create figures comparing the SPR distance and triplet distance after performing SPR moves.
    """
    tree = create_balanced_tree(r, h, fully_labelled=fully_labelled)
    plot_spr_moves_to_root(
        tree, f"spr_move_figures/{'fully_labelled_' if fully_labelled else ''}balanced_tree_spr_move_r{r}_h{h}.png"
    )


if __name__ == "__main__":
    os.makedirs("spr_move_figures", exist_ok=True)
    r = 4  # Number of branches
    h = 15  # Height of the tree
    plot_caterpillar_spr_triplet_distance(r, h, fully_labelled=True)
    plot_caterpillar_spr_triplet_distance(r, h, fully_labelled=False)

    r = 2
    h = 6
    plot_fully_balanced_tree_spr_triplet_distance(r, h, fully_labelled=True)
    plot_fully_balanced_tree_spr_triplet_distance(r, h, fully_labelled=False)
