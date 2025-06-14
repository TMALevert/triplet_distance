from rooted_triplet_distance import LevelOneNetwork


"""
The networks used here are taken from the paper
"Evaluating methods for computer-assisted stemmatology using artificial benchmark data sets" (2009) by Roos and Heikkila.
The networks are reconstructions of the "Notre Besoin" tradition.
"""

labels = ["T1", "T2", "A", "J", "C", "U", "M", "S", "F", "V", "D", "B", "L"]

true_stemma = LevelOneNetwork({"T1": {"T2": {"A":{"J": {},
                                                       "C": {"M": {},
                                                             "S": {"D": {}},
                                                             "F": {}},
                                                       "U": {"V": {},
                                                             "F": {},
                                                             "B": {"L": {}}
                                                             },
                                                  }
                                             }
                                      }
                               },
                              labels)

class_meth_b = LevelOneNetwork({"T1": {"T2": {"A":{"*1": {"J": {},
                                                          "*4": {"C": {"M": {}},
                                                                 "S": {"D": {}},
                                                                 "F": {}
                                                                 }
                                                          },
                                                   "*2": {"*3": {"V": {},
                                                                 "B": {"L": {}}
                                                                 },
                                                          "U": {"F": {}}
                                                          }
                                                   }
                                              }
                                       }
                                },
                               labels)

neighbour_joining = LevelOneNetwork({"T1": {"*1": {"T2": {},
                                                   "*2": {"*3": {"A": {},
                                                                 "J": {}},
                                                          "*4": {"*5": {"*7": {"M": {},
                                                                               "C": {}},
                                                                        "*8": {"D": {},
                                                                               "S": {}}},
                                                                 "*6": {"F": {},
                                                                        "*9": {"U": {},
                                                                               "V": {},
                                                                               "*10": {"L": {},
                                                                                       "B": {}}
                                                                               }
                                                                        }
                                                                 }
                                                          }
                                                   }
                                            }
                                     },
                                    labels)

rhm = LevelOneNetwork({"T1": {"*1": {"T2": {},
                                                   "*2": {"*3": {"A": {},
                                                                 "J": {}},
                                                          "*4": {"*5": {"*7": {"M": {},
                                                                               "C": {}},
                                                                        "*8": {"D": {},
                                                                               "S": {}}},
                                                                 "*6": {"*9": {"F": {},
                                                                               "U": {}},
                                                                        "*10": {"V": {},
                                                                                "*11": {"L": {},
                                                                                        "B": {}}}
                                                                        }
                                                                 }
                                                          }
                                                   }
                                            }
                                     },
                                    labels)

if __name__ == "__main__":
    print("Triplet distances:")
    print(f"True stemma vs Classical Method B: {true_stemma - class_meth_b}")
    print(f"True stemma vs Neighbour Joining : {true_stemma - neighbour_joining}")
    print(f"True stemma vs RHM               : {true_stemma - rhm}")

    print("Robinson-Foulds distances:\n")
    print(f"True stemma vs Classical Method B: {true_stemma.robinson_foulds_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.robinson_foulds_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.robinson_foulds_distance(rhm)}")

    print("Tripartition distances:\n")
    print(f"True stemma vs Classical Method B: {true_stemma.tripartition_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.tripartition_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.tripartition_distance(rhm)}")

    print("Mu distances:\n")
    print(f"True stemma vs Classical Method B: {true_stemma.mu_distance(class_meth_b)}")
    print(f"True stemma vs Neighbour Joining : {true_stemma.mu_distance(neighbour_joining)}")
    print(f"True stemma vs RHM               : {true_stemma.mu_distance(rhm)}")