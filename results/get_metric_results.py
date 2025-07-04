from rooted_triplet_distance import LevelOneNetwork


"""
The networks used here (except manual_improved) are taken from the paper
"Evaluating methods for computer-assisted stemmatology using artificial benchmark data sets" (2009) by Roos and Heikkila.
The networks are reconstructions of the "Notre Besoin" tradition.
"""

labels = ["T1", "T2", "A", "J", "C", "U", "M", "S", "F", "V", "D", "B", "L"]

true_stemma = LevelOneNetwork(
    {
        "T1": {
            "T2": {
                "A": {
                    "J": {},
                    "C": {"M": {}, "S": {"D": {}}, "F": {}},
                    "U": {"V": {}, "F": {}, "B": {"L": {}}},
                }
            }
        }
    },
    labels,
)

class_meth_b = LevelOneNetwork(
    {
        "T1": {
            "T2": {
                "A": {
                    "*1": {"J": {}, "*4": {"C": {"M": {}}, "S": {"D": {}}, "F": {}}},
                    "*2": {"*3": {"V": {}, "B": {"L": {}}}, "U": {"F": {}}},
                }
            }
        }
    },
    labels,
)

manual_improved = LevelOneNetwork(
    {
        "T1": {
            "T2": {
                "A": {
                    "*1": {"J": {}, "C": {"M": {}, "S": {"D": {}}, "F": {}}},
                    "*2": {"*3": {"V": {}, "B": {"L": {}}}, "U": {"F": {}}},
                }
            }
        }
    },
    labels,
)

neighbour_joining = LevelOneNetwork(
    {
        "T1": {
            "*1": {
                "T2": {},
                "*2": {
                    "*3": {"A": {}, "J": {}},
                    "*4": {
                        "*5": {"*7": {"M": {}, "C": {}}, "*8": {"D": {}, "S": {}}},
                        "*6": {"F": {}, "*9": {"U": {}, "V": {}, "*10": {"L": {}, "B": {}}}},
                    },
                },
            }
        }
    },
    labels,
)

rhm = LevelOneNetwork(
    {
        "T1": {
            "*1": {
                "T2": {},
                "*2": {
                    "*3": {"A": {}, "J": {}},
                    "*4": {
                        "*5": {"*7": {"M": {}, "C": {}}, "*8": {"D": {}, "S": {}}},
                        "*6": {"*9": {"F": {}, "U": {}}, "*10": {"V": {}, "*11": {"L": {}, "B": {}}}},
                    },
                },
            }
        }
    },
    labels,
)

if __name__ == "__main__":
    print("Triplet distances:")
    print(f"True stemma vs Classical Method B: {true_stemma - class_meth_b}")
    print(f"True stemma vs Neighbour Joining : {true_stemma - neighbour_joining}")
    print(f"True stemma vs RHM               : {true_stemma - rhm}\n")

    print(f"Classical Method B vs Neighbour Joining : {class_meth_b - neighbour_joining}")
    print(f"Classical Method B vs RHM               : {class_meth_b - rhm}")
    print(f"Neighbour Joining vs RHM                : {neighbour_joining - rhm}\n")

    print(f"True stemma vs Manual Improved        : {true_stemma - manual_improved}")
    print(f"Classical Method B vs Manual Improved : {manual_improved - class_meth_b}\n")

    print("Average sign distances:")
    print(f"True stemma vs Classical Method B: {true_stemma.average_sign_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.average_sign_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.average_sign_distance(rhm)}\n")

    print(f"Classical Method B vs Neighbour Joining : {class_meth_b.average_sign_distance(neighbour_joining)}")
    print(f"Classical Method B vs RHM               : {class_meth_b.average_sign_distance(rhm)}")
    print(f"Neighbour Joining vs RHM                : {neighbour_joining.average_sign_distance(rhm)}\n")

    print(f"True stemma vs Manual Improved        : {true_stemma.average_sign_distance(manual_improved)}")
    print(f"Classical Method B vs Manual Improved : {manual_improved.average_sign_distance(class_meth_b)}\n")

    print("Robinson-Foulds distances:")
    print(f"True stemma vs Classical Method B: {true_stemma.robinson_foulds_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.robinson_foulds_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.robinson_foulds_distance(rhm)}\n")

    print(f"Classical Method B vs Neighbour Joining : {class_meth_b.robinson_foulds_distance(neighbour_joining)}")
    print(f"Classical Method B vs RHM               : {class_meth_b.robinson_foulds_distance(rhm)}")
    print(f"Neighbour Joining vs RHM                : {neighbour_joining.robinson_foulds_distance(rhm)}\n")

    print(f"True stemma vs Manual Improved        : {true_stemma.robinson_foulds_distance(manual_improved)}")
    print(f"Classical Method B vs Manual Improved : {manual_improved.robinson_foulds_distance(class_meth_b)}\n")

    print("Tripartition distances:")
    print(f"True stemma vs Classical Method B: {true_stemma.tripartition_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.tripartition_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.tripartition_distance(rhm)}\n")

    print(f"Classical Method B vs Neighbour Joining : {class_meth_b.tripartition_distance(neighbour_joining)}")
    print(f"Classical Method B vs RHM               : {class_meth_b.tripartition_distance(rhm)}")
    print(f"Neighbour Joining vs RHM                : {neighbour_joining.tripartition_distance(rhm)}\n")

    print(f"True stemma vs Manual Improved        : {true_stemma.tripartition_distance(manual_improved)}")
    print(f"Classical Method B vs Manual Improved : {manual_improved.tripartition_distance(class_meth_b)}\n")

    print("Mu distances:")
    print(f"True stemma vs Classical Method B: {true_stemma.mu_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.mu_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.mu_distance(rhm)}\n")

    print(f"Classical Method B vs Neighbour Joining : {class_meth_b.mu_distance(neighbour_joining)}")
    print(f"Classical Method B vs RHM               : {class_meth_b.mu_distance(rhm)}")
    print(f"Neighbour Joining vs RHM                : {neighbour_joining.mu_distance(rhm)}\n")

    print(f"True stemma vs Manual Improved        : {true_stemma.mu_distance(manual_improved)}")
    print(f"Classical Method B vs Manual Improved : {manual_improved.mu_distance(class_meth_b)}")
