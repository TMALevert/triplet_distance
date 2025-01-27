from rooted_triplet_distance import TreeReconstruction, Tree

if __name__ == "__main__":
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Z"]
    triplets = [
        "Z|F|G",
        "Z|IC",
        "Z|AB",
        # "Z|CD",
        # "Z|IE",
        # "Z|GH",
        # "ZG|A",
        # "AB|C",
        # "AB|D",
        # "AB|E",
        # "AB|F",
        # "AB|G",
        # "AB|H",
        # "AB|J",
        "A|CD",
        # "A|C|E",
        # "AC|F",
        # "AC|G",
        "AC|H",
        "AC|J",
        # "A|D|E",
        # "AD|F",
        # "AD|G",
        # "AD|H",
        # "AD|J",
        # "AE|F",
        # "AE|G",
        # "AE|H",
        # "AE|J",
        # "A|FG",
        # "A|FH",
        # "A|GH",
        # "B|CD",
        # "B|C|E",
        "BC|F",
        "BC|G",
        # "BC|H",
        # "BC|J",
        # "B|D|E",
        # "BD|F",
        # "BD|G",
        # "BD|H",
        # "BD|J",
        # "BE|F",
        # "BE|G",
        # "BE|H",
        # "BE|J",
        "B|FG",
        "B|FH",
        "B|GH",
        "CD|E",
        "CD|F",
        "CD|G",
        # "CD|H",
        # "CD|I",
        # "CD|J",
        # "CE|F",
        # "CE|G",
        # "CE|H",
        # "C|E|I",
        # "CE|J",
        # "C|FG",
        # "C|FH",
        # "CI|F",
        # "C|GH",
        "CI|G",
        "CI|H",
        "CI|J",
        "DE|F",
        "DE|G",
        "DE|H",
        "D|E|I",
        # "DE|J",
        # "D|GF",
        # "D|FH",
        # "DI|F",
        # "D|GH",
        # "DI|G",
        # "DI|H",
        # "DI|J",
        # "E|FG",
        # "E|FH",
        # "EI|F",
        # "E|GH",
        # "EI|G",
        "EI|H",
        "EI|J",
        "F|GH",
        # "FG|I",
        "FH|I",
        "GH|I",
    ]

    tree = TreeReconstruction(labels, triplets).reconstruct()
    print(tree)
    tree = Tree(tree, labels)
    tree.visualize()

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tree_dict = {
        "root": {"*1": {"A": {"B": {}, "C": {}}, "D": {}}, "*2": {"F": {"G": {}, "H": {}, "E": {}}, "I": {}, "J": {}}}
    }
    tree = Tree(tree_dict, labels)
    tree.visualize()
    print(tree.triplets)
    tree = TreeReconstruction(labels, tree.triplets).reconstruct()
    print(tree)
    tree = Tree(tree, labels)
    tree.visualize()
