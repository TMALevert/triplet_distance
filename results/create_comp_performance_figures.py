import os
from pathlib import Path
from timeit import Timer
from random import sample

from pandas import read_csv, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from rooted_triplet_distance import (
    MultifurcatingTree,
    GeneralTree,
    GeneralTreeReconstruction,
    MultifurcatingTreeReconstruction,
)

__time_types = ["time_multi_alg", "time_gen_alg", "time_network_alg"]
__time_to_reconstruction = {
    "time_multi_alg": "Multifurcating Tree Reconstruction",
    "time_gen_alg": "General Tree Reconstruction",
    "time_network_alg": "Network Reconstruction",
}
__time_to_colour = {
    "time_multi_alg": "#1A659E",
    "time_gen_alg": "#F58300",
    "time_network_alg": "#058A22",
}
__time_to_fit_colour = {
    "time_multi_alg": "#018ADF",
    "time_gen_alg": "#F0C808",
    "time_network_alg": "#82C62F",
}
__time_to_runtime = {
    "time_multi_alg": {"X": 7, "triplets": 7 / 3},
    "time_gen_alg": {"X": 8, "triplets": 8 / 3},
    "time_network_alg": {"X": 8, "triplets": 8 / 3},
}


def get_timing_gen_alg_partial_triplet_set(tree: GeneralTree, frac: float):
    triplets = tree.triplets
    labels = tree.labels
    min_time = np.inf
    for _ in range(5):
        subset_triplets = sample(triplets, round(frac * len(triplets)))
        try:
            timer = Timer(lambda: GeneralTreeReconstruction(labels, subset_triplets).reconstruct())
            time = min(timer.repeat(1, 1))
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"Error during general reconstruction with {frac} fraction: {e}\n")
                f.write(f"Triplets: {subset_triplets}\n")
                f.write(f"Labels: {labels}\n")
                f.write(f"Tree dict: {tree._tree_dict}\n")
            continue
        min_time = min(min_time, time)
    return min_time


def get_timing_multi_alg_partial_triplet_set(tree: MultifurcatingTree, frac: float):
    triplets = tree.triplets
    labels = tree.labels
    min_time = np.inf
    for _ in range(5):
        subset_triplets = sample(triplets, round(frac * len(triplets)))
        try:
            timer = Timer(lambda: MultifurcatingTreeReconstruction(labels, subset_triplets).reconstruct())
            time = min(timer.repeat(1, 1))
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"Error during multi reconstruction with {frac} fraction: {e}\n")
                f.write(f"Triplets: {subset_triplets}\n")
                f.write(f"Labels: {labels}\n")
                f.write(f"Tree dict: {tree._tree_dict}")
            continue
        min_time = min(min_time, time)
    return min_time


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


def plot_runtime_fit(x, y, power, colour, x_string: str):
    """
    Fit the theoretical runtime on the given x and y values and the power.
    :param x: The x values of the data.
    :param y: The y values of the data.
    :param power: The power to which the x values are raised based on the theoretical running time.
    :param colour: The colour of the line in the plot.
    :param x_string: The string representation of the x values for the label.
    """

    def __theoretical_runtime(x, a, b, p):
        return a * (x**p) + b

    def latex_float(numb: float) -> str:
        float_str = "{0:.2g}".format(numb)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return rf"{base} \cdot 10^{{{int(exponent)}}}"
        else:
            return float_str

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    popt, _ = curve_fit(__theoretical_runtime, x, y, p0=[1, 1, power], bounds=(0, np.inf))
    x_range = np.linspace(min(x), max(x), 100)
    y_fit = __theoretical_runtime(x_range, *popt)
    plt.plot(
        x_range,
        y_fit,
        color=colour,
        label=f"Best fit: ${latex_float(popt[0])}{x_string}^{'{'}{round(popt[2], 2)}{'}'}+{latex_float(popt[1])}$",
        linestyle="dashdot",
        zorder=3,
    )
    plt.legend(loc="lower right")


def plot_time_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["numb_labels"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                color=__time_to_colour[time_type],
            )
            plot_runtime_fit(
                data["numb_labels"],
                data[time_type],
                __time_to_runtime[time_type]["X"],
                __time_to_fit_colour[time_type],
                "$|X|$",
            )
    plt.yscale("log")
    plt.xlabel("Number of labels")
    plt.ylabel("Time [s]")
    if network_type != "network":
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_numb_labels.png")
    plt.close()


def plot_time_vs_numb_labels_only_multi_and_gen_alg(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in ["time_multi_alg", "time_gen_alg"]:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["numb_labels"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                color=__time_to_colour[time_type],
            )
            plot_runtime_fit(
                data["numb_labels"],
                data[time_type],
                __time_to_runtime[time_type]["X"],
                __time_to_fit_colour[time_type],
                "$|X|$",
            )
    plt.yscale("log")
    plt.xlabel("Number of labels")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_numb_labels_only_multi_and_gen_alg.png")
    plt.close()


def plot_time_vs_numb_gen_triplets(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["numb_gen_triplets"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                color=__time_to_colour[time_type],
            )
            plot_runtime_fit(
                data["numb_gen_triplets"],
                data[time_type],
                __time_to_runtime[time_type]["triplets"],
                __time_to_fit_colour[time_type],
                f"$|t({'N' if 'network' in network_type else 'T'})|$",
            )
    plt.yscale("log")
    plt.xlabel("Number of general triplets")
    plt.ylabel("Time [s]")
    if network_type != "network":
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_numb_triplets.png")
    plt.close()


def plot_time_vs_numb_multi_triplets(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["numb_multi_triplets"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                color=__time_to_colour[time_type],
            )
            plot_runtime_fit(
                data["numb_multi_triplets"],
                data[time_type],
                __time_to_runtime[time_type]["triplets"],
                __time_to_fit_colour[time_type],
                f"$|t({'N' if 'network' in network_type else 'T'})|$",
            )
    plt.yscale("log")
    plt.xlabel("Number of general triplets")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_numb_triplets.png")
    plt.close()


def plot_time_vs_numb_cycles(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["numb_cycles"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                alpha=0.8,
                c=data["numb_labels"],
                cmap="jet",
            )
    plt.colorbar(label="Number of labels")
    plt.xlabel("Number of cycles")
    plt.ylabel("Time [s]")
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_numb_cycles.png")
    plt.close()


def plot_time_vs_max_cycle_size(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["max_cycle_size"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                alpha=0.8,
                c=data["numb_labels"],
                cmap="jet",
            )
    plt.colorbar(label="Number of labels")
    plt.xlabel("Maximum cycle size")
    plt.ylabel("Time [s]")
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_max_cycle_size.png")
    plt.close()


def plot_time_vs_numb_nodes(data: DataFrame, network_type: str):
    plt.figure()
    for time_type in __time_types:
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["numb_nodes"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                alpha=0.8,
                color=__time_to_colour[time_type],
            )
    plt.yscale("log")
    plt.xlabel("Number of nodes")
    plt.ylabel("Time [s]")
    if network_type != "network":
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_numb_nodes.png")
    plt.close()


def plot_time_vs_max_depth(data: DataFrame, network_type: str):
    plt.figure()
    markers = ["+", "x", "1"]
    for ind, time_type in enumerate(__time_types):
        if not all(np.isnan(data[time_type])):
            plt.scatter(
                data["max_depth"],
                data[time_type],
                label=__time_to_reconstruction[time_type],
                alpha=0.8,
                marker=markers[ind],
                c=data["numb_labels"],
                cmap="jet",
            )
    plt.colorbar(label="Number of labels")
    plt.yscale("log")
    plt.xlabel("Maximum depth")
    plt.ylabel("Time [s]")
    if network_type != "network":
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_time_vs_max_depth.png")
    plt.close()


def plot_numb_multi_triplets_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_labels"], data["numb_multi_triplets"], alpha=0.5)
    plt.xlabel("Number of labels")
    plt.ylabel("Number of multifurcating triplets")
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_numb_multi_triplets_vs_numb_labels.png")
    plt.close()


def plot_numb_gen_triplets_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_labels"], data["numb_gen_triplets"], alpha=0.5)
    plt.xlabel("Number of labels")
    plt.ylabel("Number of general triplets")
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_numb_gen_triplets_vs_numb_labels.png")
    plt.close()


def plot_numb_multi_and_gen_triplets_vs_numb_labels(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_labels"], data["numb_multi_triplets"], alpha=0.5, label="Multifurcating triplets")
    plt.scatter(data["numb_labels"], data["numb_gen_triplets"], alpha=0.5, label="General triplets")
    plt.xlabel("Number of labels")
    plt.ylabel("Number of triplets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_numb_multi_and_gen_triplets_vs_numb_labels.png")
    plt.close()


def plot_numb_gen_triplets_vs_numb_multi_triplets(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_multi_triplets"], data["numb_gen_triplets"], alpha=0.5)
    plt.xlabel("Number of multifurcating triplets")
    plt.ylabel("Number of general triplets")
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_numb_gen_triplets_vs_numb_multi_triplets.png")
    plt.close()


def plot_numb_gen_triplets_vs_numb_cycles(data: DataFrame, network_type: str):
    plt.figure()
    plt.scatter(data["numb_cycles"], data["numb_gen_triplets"], alpha=0.8, c=data["numb_labels"], cmap="jet")
    plt.colorbar(label="Number of labels")
    plt.xlabel("Number of cycles")
    plt.ylabel("Number of general triplets")
    plt.tight_layout()
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
                    label=f"{(' '.join(network_type.split('_')) + ' instances').title()}",
                )
        plot_runtime_fit(
            np.concatenate([data["numb_labels"] for data in datas.values() if not all(np.isnan(data[time_type]))]),
            np.concatenate([data[time_type] for data in datas.values() if not all(np.isnan(data[time_type]))]),
            __time_to_runtime[time_type]["X"],
            "red",
            "$|X|$",
        )
        plt.yscale("log")
        plt.xlabel("Number of labels")
        plt.ylabel("Time [s]")
        if time_type != "time_multi_alg":
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/{time_type}_vs_numb_labels_per_algorithm.png")
        plt.close()


def plot_time_vs_numb_triplets_per_algorithm(datas: dict):
    for time_type in __time_types:
        plt.figure()
        for network_type, data in datas.items():
            if not all(np.isnan(data[time_type])):
                plt.scatter(
                    data["numb_gen_triplets" if time_type != "time_multi_alg" else "numb_multi_triplets"],
                    data[time_type],
                    label=f"{(' '.join(network_type.split('_')) + ' instances').title()}",
                )
        plot_runtime_fit(
            np.concatenate(
                [
                    data["numb_gen_triplets" if time_type != "time_multi_alg" else "numb_multi_triplets"]
                    for data in datas.values()
                    if not all(np.isnan(data[time_type]))
                ]
            ),
            np.concatenate([data[time_type] for data in datas.values() if not all(np.isnan(data[time_type]))]),
            __time_to_runtime[time_type]["triplets"],
            "red",
            f"$|t({'N' if 'network' in time_type else 'T'})|$",
        )
        plt.yscale("log")
        plt.xlabel(f"Number of {'general' if time_type != 'time_multi_alg' else 'multifurcating'} triplets")
        plt.ylabel("Time [s]")
        if time_type != "time_multi_alg":
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/{time_type}_vs_numb_triplets_per_algorithm.png")
        plt.close()


def plot_time_vs_max_depth_per_algorithm(datas: dict):
    plt.figure()
    markers = ["+", "x", "1"]
    for time_type in __time_types:
        min_c = min([data["numb_labels"].min() for data in datas.values()])
        max_c = max([data["numb_labels"].max() for data in datas.values()])
        for ind, (network_type, data) in enumerate(datas.items()):
            if not all(np.isnan(data[time_type])):
                plt.scatter(
                    data["max_depth"],
                    data[time_type],
                    label=f"{(' '.join(network_type.split('_')) + ' instances').title()}",
                    alpha=0.8,
                    marker=markers[ind],
                    c=data["numb_labels"],
                    cmap="jet",
                )
                plt.clim(min_c, max_c)
        plt.colorbar(label="Number of labels")
        plt.yscale("log")
        plt.xlabel("Maximum depth")
        plt.ylabel("Time [s]")
        if time_type != "time_multi_alg":
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/{time_type}_vs_max_depth_per_algorithm.png")
        plt.close()


def histogram_numb_labels(datas):
    plt.figure()
    plt.hist(
        [data["numb_labels"] for data in datas.values()],
        bins=15,
        weights=[np.ones(len(data["numb_labels"])) / len(data["numb_labels"]) for data in datas.values()],
        label=[(" ".join(network_type.split("_")) + "s").title() for network_type in datas.keys()],
    )
    plt.xlabel("Number of labels")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/numb_labels_histogram.png")
    plt.close()


def plot_timing_alg_partial_triplet_sets(data: DataFrame, network_type: str):
    fracs = [0.8, 0.6, 0.4, 0.2]
    plt.figure()
    if network_type == "multifurcating_tree":
        for frac in fracs:
            times = data.apply(
                lambda row: get_timing_multi_alg_partial_triplet_set(
                    MultifurcatingTree(eval(row["network_dict"]), eval(row["labels"])), frac
                )
                / row["time_multi_alg"],
                axis=1,
            )
            plt.scatter(data["numb_labels"], times, label=rf"$\alpha={frac}$", alpha=0.8)
    elif network_type == "general_tree":
        for frac in fracs:
            times = data.apply(
                lambda row: get_timing_gen_alg_partial_triplet_set(
                    GeneralTree(eval(row["network_dict"]), eval(row["labels"])), frac
                )
                / row["time_gen_alg"],
                axis=1,
            )
            plt.scatter(data["numb_labels"], times, label=rf"$\alpha={frac}$", alpha=0.8)
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of original runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{network_type}_timing_partial_triplet_sets_fracs.png")
    plt.close()


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
            plot_numb_multi_and_gen_triplets_vs_numb_labels(data, type)
            plot_time_vs_numb_labels_only_multi_and_gen_alg(data, type)
            plot_timing_alg_partial_triplet_sets(data, type)
        if type == "general_tree":
            plot_timing_alg_partial_triplet_sets(data, type)
        if type == "network":
            plot_time_vs_numb_cycles(data, type)
            plot_time_vs_max_cycle_size(data, type)
            plot_numb_gen_triplets_vs_numb_cycles(data, type)
    plot_time_vs_numb_labels_per_algorithm(datas)
    plot_time_vs_numb_triplets_per_algorithm(datas)
    plot_time_vs_max_depth_per_algorithm(datas)
    histogram_numb_labels(datas)
