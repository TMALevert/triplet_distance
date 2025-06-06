import os
from pathlib import Path

from pandas import read_csv, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

__time_types = ["time_multi_alg", "time_gen_alg", "time_network_alg"]
__time_to_reconstruction = {
    "time_multi_alg": "Multifurcating Tree Reconstruction",
    "time_gen_alg": "General Tree Reconstruction",
    "time_network_alg": "Network Reconstruction",
}


def load_data(network_type: str):
    """
    Load data from CSV files based on the specified network type.
    :param network_type: Either "multifurcating_tree", "general_tree", or "network".
    :return: DataFrame containing the concatenated data from all relevant CSV files.
    """
    data = DataFrame()
    for file in os.listdir(Path(__file__).parent / "data"):
        if file.endswith(".csv") and file.startswith(f"{network_type}_data"):
            try:
                x = read_csv(Path(__file__).parent / "data" / file)
            except Exception:
                print(file)
                continue
            data = pd.concat([data, x], ignore_index=True)
    return data


def plot_time_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["numb_labels"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.yscale("log")
    plt.xlabel("Number of labels")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_numb_labels.png")
    plt.close()


def plot_time_vs_numb_gen_triplets(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["numb_gen_triplets"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.yscale("log")
    plt.xlabel("Number of generated triplets")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_numb_triplets.png")
    plt.close()


def plot_time_vs_numb_multi_triplets(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["numb_multi_triplets"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.yscale("log")
    plt.xlabel("Number of generated triplets")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_numb_triplets.png")
    plt.close()


def plot_time_vs_numb_cycles(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["numb_cycles"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.xlabel("Number of cycles")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_numb_cycles.png")
    plt.close()


def plot_time_vs_max_cycle_size(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["max_cycle_size"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.xlabel("Maximum cycle size")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_max_cycle_size.png")
    plt.close()


def plot_time_vs_numb_nodes(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["numb_nodes"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.yscale("log")
    plt.xlabel("Number of nodes")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_numb_nodes.png")
    plt.close()


def plot_time_vs_max_depth(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(data["max_depth"], data[time_type], label=__time_to_reconstruction[time_type])
    plt.yscale("log")
    plt.xlabel("Maximum depth")
    plt.ylabel("Time [s]")
    plt.title(f"Reconstruction time for {network_type.replace('_', ' ')}s")
    plt.legend()
    plt.savefig(f"figures/{network_type}_time_vs_max_depth.png")
    plt.close()


def plot_numb_multi_triplets_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_labels"], data["numb_multi_triplets"], alpha=0.5)
    plt.xlabel("Number of labels")
    plt.ylabel("Number of multi triplets")
    plt.title(f"Multi triplets vs Labels for {network_type.replace('_', ' ')}s")
    plt.savefig(f"figures/{network_type}_numb_multi_triplets_vs_numb_labels.png")
    plt.close()


def plot_numb_gen_triplets_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_labels"], data["numb_gen_triplets"], alpha=0.5)
    plt.xlabel("Number of labels")
    plt.ylabel("Number of generated triplets")
    plt.title(f"Generated triplets vs Labels for {network_type.replace('_', ' ')}s")
    plt.savefig(f"figures/{network_type}_numb_gen_triplets_vs_numb_labels.png")
    plt.close()


def plot_numb_gen_triplets_vs_numb_multi_triplets(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_multi_triplets"], data["numb_gen_triplets"], alpha=0.5)
    plt.xlabel("Number of multi triplets")
    plt.ylabel("Number of generated triplets")
    plt.title(f"Generated triplets vs Multi triplets for {network_type.replace('_', ' ')}s")
    plt.savefig(f"figures/{network_type}_numb_gen_triplets_vs_numb_multi_triplets.png")
    plt.close()


def plot_numb_gen_triplets_vs_numb_cycles(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_cycles"], data["numb_gen_triplets"], alpha=0.5)
    plt.xlabel("Number of cycles")
    plt.ylabel("Number of generated triplets")
    plt.title(f"Generated triplets vs Cycles for {network_type.replace('_', ' ')}s")
    plt.savefig(f"figures/{network_type}_numb_gen_triplets_vs_numb_cycles.png")
    plt.close()


def plot_time_vs_numb_labels_per_algorithm(datas: dict):
    plt.figure()
    for time_type in __time_types:
        for network_type, data in datas.items():
            if not all(np.isnan(data[time_type])):
                plt.scatter(
                    data["numb_labels"],
                    data[time_type],
                    label=f"{network_type} - {__time_to_reconstruction[time_type]}",
                )
        plt.yscale("log")
        plt.xlabel("Number of labels")
        plt.ylabel("Time [s]")
        plt.title(f"Reconstruction time for {__time_to_reconstruction[time_type]}\n per network type")
        plt.legend()
        plt.savefig(f"figures/{time_type}_vs_numb_labels_per_algorithm.png")
        plt.close()


def histogram_numb_labels(datas):
    plt.figure()
    # for network_type, data in datas.items():
    plt.hist(
        [data["numb_labels"] for data in datas.values()],
        bins=15,
        weights=[np.ones(len(data["numb_labels"])) / len(data["numb_labels"]) for data in datas.values()],
        label=[" ".join(network_type.split("_")) + "s" for network_type in datas.keys()],
    )
    plt.xlabel("Number of labels")
    plt.ylabel("Frequency")
    plt.title("Distribution of number of labels per network type")
    plt.legend()
    plt.savefig(f"figures/numb_labels_histogram.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    datas = {}
    for type in ["multifurcating_tree", "general_tree", "network"]:
        data = load_data(type)
        datas[type] = data
        print(f"Loaded {len(data)} rows for {type}.")

        plot_time_vs_numb_labels(data, type)
        plot_time_vs_numb_gen_triplets(data, type)
        plot_time_vs_numb_nodes(data, type)
        plot_time_vs_max_depth(data, type)
        plot_numb_gen_triplets_vs_numb_labels(data, type)
        if type == "multifurcating_tree":
            plot_time_vs_numb_multi_triplets(data, type)
            plot_numb_multi_triplets_vs_numb_labels(data, type)
            plot_numb_gen_triplets_vs_numb_multi_triplets(data, type)
        if type == "network":
            plot_time_vs_numb_cycles(data, type)
            plot_time_vs_max_cycle_size(data, type)
            plot_numb_gen_triplets_vs_numb_cycles(data, type)
    plot_time_vs_numb_labels_per_algorithm(datas)
    histogram_numb_labels(datas)
