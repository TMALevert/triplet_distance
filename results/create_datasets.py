from datetime import datetime
from io import TextIOWrapper
from random import randint
from timeit import Timer
from pprint import pprint
import csv

import numpy as np
from networkx import dag_longest_path
from networkx.algorithms.components import biconnected_components

from rooted_triplet_distance import MultifurcatingTree, GeneralTree, LevelOneNetwork, MultifurcatingTreeReconstruction, GeneralTreeReconstruction, LevelOneNetworkReconstruction
from network_generators import create_random_multifurcating_tree, create_random_general_tree, create_random_level_1_network


def get_timing_gen_alg(tree: GeneralTree):
    timer = Timer(lambda: GeneralTreeReconstruction(tree.labels, tree.triplets).reconstruct())
    time = min(timer.repeat(5, 1))
    try:
        reconstructed_dict = GeneralTreeReconstruction(tree.labels, tree.triplets).reconstruct()
        assert tree == GeneralTree(reconstructed_dict, tree.labels)
    except AssertionError:
        print("General reconstruction failed")
        pprint(tree._tree_dict)
        pprint(tree.labels)
        pprint(reconstructed_dict)
    return time


def get_timing_multi_alg(tree: MultifurcatingTree):
    timer = Timer(lambda: MultifurcatingTreeReconstruction(tree.labels, tree.triplets).reconstruct())
    time = min(timer.repeat(5, 1))
    try:
        reconstructed_dict = MultifurcatingTreeReconstruction(tree.labels, tree.triplets).reconstruct()
        assert tree == MultifurcatingTree(reconstructed_dict, tree.labels)
    except AssertionError:
        print("Multifurcating reconstruction failed")
        pprint(tree._tree_dict)
        pprint(tree.labels)
        pprint(reconstructed_dict)
    return time


def get_timing_network_alg(network: LevelOneNetwork):
    timer = Timer(lambda: LevelOneNetworkReconstruction(network.labels, network.triplets).reconstruct())
    time = min(timer.repeat(5, 1))
    try:
        reconstructed_dict = LevelOneNetworkReconstruction(network.labels, network.triplets).reconstruct()
        assert network == LevelOneNetwork(reconstructed_dict, network.labels)
    except AssertionError:
        print("Network reconstruction failed")
        pprint(network._tree_dict)
        pprint(network.labels)
        pprint(reconstructed_dict)
    return time


def create_multifurcating_tree_dataframe(file: TextIOWrapper, numb_trees: int, min_labels: int, max_labels: int):
    columns=["network_id", "network_dict", "labels", "numb_labels", "numb_nodes", "numb_cycles", "max_cycle_size",
                 "max_depth", "numb_multi_triplets", "numb_gen_triplets", "time_multi_alg", "time_gen_alg",
                 "time_network_alg"]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    for id in range(numb_trees):
        print(id)
        dict, labels = create_random_multifurcating_tree(randint(min_labels, max_labels), 4)
        multi_tree = MultifurcatingTree(dict, labels)
        gen_tree = GeneralTree(dict, labels)
        network = LevelOneNetwork(dict, labels)
        row_data = dict.fromkeys(columns)
        row_data["network_id"] = id
        row_data["network_dict"] = dict
        row_data["labels"] = labels
        row_data["numb_labels"] = len(labels)
        row_data["numb_nodes"] = len(multi_tree._tree.nodes)
        row_data["numb_cycles"] = 0
        row_data["max_cycle_size"] = 0
        row_data["max_depth"] = len(dag_longest_path(multi_tree._tree))
        row_data["numb_multi_triplets"] = len(multi_tree.triplets)
        row_data["numb_gen_triplets"] = len(gen_tree.triplets)
        row_data["time_multi_alg"] = get_timing_multi_alg(multi_tree)
        row_data["time_gen_alg"] = get_timing_gen_alg(gen_tree)
        row_data["time_network_alg"] = get_timing_network_alg(network)
        writer.writerow(row_data)


def create_general_tree_dataframe(file: TextIOWrapper, numb_trees: int, min_labels: int, max_labels: int):
    columns=["network_id", "network_dict", "labels", "numb_labels", "numb_nodes", "numb_cycles", "max_cycle_size",
                 "max_depth", "numb_multi_triplets", "numb_gen_triplets", "time_multi_alg", "time_gen_alg",
                 "time_network_alg"]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    for id in range(numb_trees):
        print(id)
        dict, labels = create_random_general_tree(randint(min_labels, max_labels))
        gen_tree = GeneralTree(dict, labels)
        network = LevelOneNetwork(dict, labels)
        row_data = dict.fromkeys(columns)
        row_data["network_id"] = id
        row_data["network_dict"] = dict
        row_data["labels"] = labels
        row_data["numb_labels"] = len(labels)
        row_data["numb_nodes"] = len(gen_tree._tree.nodes)
        row_data["numb_cycles"] = 0
        row_data["max_cycle_size"] = 0
        row_data["max_depth"] = len(dag_longest_path(gen_tree._tree))
        row_data["numb_multi_triplets"] = len(gen_tree.triplets)
        row_data["numb_gen_triplets"] = len(gen_tree.triplets)
        row_data["time_multi_alg"] = np.nan
        row_data["time_gen_alg"] = get_timing_gen_alg(gen_tree)
        row_data["time_network_alg"] = get_timing_network_alg(network)
        writer.writerow(row_data)


def create_level_1_network_dataframe(file: TextIOWrapper, numb_networks: int, min_labels: int, max_labels: int, min_n_reticulations: int, max_n_reticulations: int):
    columns=["network_id", "network_dict", "labels", "numb_labels", "numb_nodes", "numb_cycles", "max_cycle_size",
                 "max_depth", "numb_multi_triplets", "numb_gen_triplets", "time_multi_alg", "time_gen_alg",
                 "time_network_alg"]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    for id in range(numb_networks):
        print(id)
        dict, labels = create_random_level_1_network(randint(min_labels, max_labels), randint(min_n_reticulations, max_n_reticulations))
        network = LevelOneNetwork(dict, labels)
        row_data = dict.fromkeys(columns)
        row_data["network_id"] = id
        row_data["network_dict"] = dict
        row_data["labels"] = labels
        row_data["numb_labels"] = len(labels)
        row_data["numb_nodes"] = len(network._tree.nodes)
        numb_cycles = 0
        max_cycle_size = 0
        for cycle in biconnected_components(network._tree.to_undirected()):
            if len(cycle) > 2:
                numb_cycles += 1
                max_cycle_size = max(max_cycle_size, len(cycle))
        row_data["numb_cycles"] = numb_cycles
        row_data["max_cycle_size"] = max_cycle_size
        row_data["max_depth"] = len(dag_longest_path(network._tree))
        row_data["numb_multi_triplets"] = np.nan
        row_data["numb_gen_triplets"] = len(network.triplets)
        row_data["time_multi_alg"] = np.nan
        row_data["time_gen_alg"] = np.nan
        row_data["time_network_alg"] = get_timing_network_alg(network)
        writer.writerow(row_data)


if __name__ == "__main__":
    with open(f"network_data_{datetime.now().strftime('%d_%m_%Hu%Mm%Ss')}.csv", "w", newline='') as file:
        create_level_1_network_dataframe(file,100, 3, 30, 1, 4)
    with open(f"multifurcating_tree_data_{datetime.now().strftime('%d_%m_%Hu%Mm%Ss')}.csv", "w", newline='') as file:
        create_multifurcating_tree_dataframe(file,150, 4, 40)
    with open(f"general_tree_data_{datetime.now().strftime('%d_%m_%Hu%Mm%Ss')}.csv", "w", newline='') as file:
        create_general_tree_dataframe(file, 150, 3, 40)