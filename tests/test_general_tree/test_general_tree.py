import pytest

from rooted_triplet_distance.general_tree import Tree


def test_graph_creation():
    tree = {
        "A": {
            "B": {"D": {"H": {}, "I": {}}, "E": {"J": {}, "K": {}}},
            "C": {"F": {"L": {}, "M": {}}, "G": {"N": {}, "O": {}}},
        }
    }
    t = Tree(tree)
    assert all(
        label in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"] for label in t.labels
    )


@pytest.mark.parametrize(
    "tree, labels, triplets",
    [
        ({"root": {"1": {"A": {}, "B": {}}, "C": {}}}, ["A", "B", "C"], ["A,B|C"]),
        ({"root": {"A": {}, "B": {}, "C": {}}}, ["A", "B", "C"], ["A|B|C"]),
        ({"root": {"A": {}, "B": {}}}, ["A", "B"], []),
        (
            {"root": {"A": {}, "B": {"C": {}, "D": {}, "E": {}}}},
            ["A", "B", "C", "D", "E"],
            ["C|D|E", "C,D|A", "C,E|A", "D,E|A", r"A|B\C", r"A|B\D", r"A|B\E", r"C/B\D", r"C/B\E", r"D/B\E"],
        ),
        (
            {
                "root": {
                    "*1": {"A": {"B": {}, "C": {}}, "D": {}},
                    "*2": {"F": {"G": {}, "H": {}, "E": {}}, "I": {}, "J": {}},
                }
            },
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            [
                "A,D|F",
                "A,D|I",
                "A,D|J",
                "A,D|G",
                "A,D|H",
                "A,D|E",
                "F,I|A",
                "F,J|A",
                "I,J|A",
                "I,G|A",
                "I,H|A",
                "I,E|A",
                "J,G|A",
                "J,H|A",
                "J,E|A",
                "G,H|A",
                "G,E|A",
                "H,E|A",
                "B,C|D",
                "D,B|F",
                "D,B|I",
                "D,B|J",
                "D,B|G",
                "D,B|H",
                "D,B|E",
                "D,C|F",
                "D,C|I",
                "D,C|J",
                "D,C|G",
                "D,C|H",
                "D,C|E",
                "F,I|D",
                "F,J|D",
                "I,J|D",
                "I,G|D",
                "I,H|D",
                "I,E|D",
                "J,G|D",
                "J,H|D",
                "J,E|D",
                "G,H|D",
                "G,E|D",
                "H,E|D",
                "B,C|F",
                "B,C|I",
                "B,C|J",
                "B,C|G",
                "B,C|H",
                "B,C|E",
                "F,I|B",
                "F,J|B",
                "I,J|B",
                "I,G|B",
                "I,H|B",
                "I,E|B",
                "J,G|B",
                "J,H|B",
                "J,E|B",
                "G,H|B",
                "G,E|B",
                "H,E|B",
                "F,I|C",
                "F,J|C",
                "I,J|C",
                "I,G|C",
                "I,H|C",
                "I,E|C",
                "J,G|C",
                "J,H|C",
                "J,E|C",
                "G,H|C",
                "G,E|C",
                "H,E|C",
                "F|I|J",
                "I|J|G",
                "I|J|H",
                "I|J|E",
                "G,H|I",
                "G,E|I",
                "H,E|I",
                "G,H|J",
                "G,E|J",
                "H,E|J",
                "G|H|E",
                r"G/F\H",
                r"G/F\E",
                r"H/F\E",
                r"G/F|I",
                r"G/F|J",
                r"H/F|I",
                r"H/F|J",
                r"E/F|I",
                r"E/F|J",
                r"G/F|D",
                r"H/F|D",
                r"E/F|D",
                r"G/F|C",
                r"H/F|C",
                r"E/F|C",
                r"G/F|B",
                r"H/F|B",
                r"E/F|B",
                r"G/F|A",
                r"H/F|A",
                r"E/F|A",
                r"B/A|D",
                r"C/A|D",
                r"B/A\C",
                r"B/A|F",
                r"B/A|I",
                r"B/A|J",
                r"B/A|G",
                r"B/A|H",
                r"B/A|E",
                r"C/A|F",
                r"C/A|I",
                r"C/A|J",
                r"C/A|G",
                r"C/A|H",
                r"C/A|E",
            ],
        ),
        ({"A": {"B": {"C": {}}}}, ["A", "B", "C"], [r"A\B\C"]),
        (
            {"A": {"B": {"C": {}, "D": {}}, "*1": {"E": {}, "F": {}}}},
            ["A", "B", "C", "D", "E", "F"],
            [
                r"A\B\C",
                r"A\B\D",
                r"C/B\D",
                r"B/A\E",
                r"B/A\F",
                r"D/A\E",
                r"D/A\F",
                r"C/A\E",
                r"C/A\F",
                r"C/B|E",
                r"C/B|F",
                r"D/B|E",
                r"D/B|F",
                r"E,F|B",
                r"E,F|C",
                r"E,F|D",
                r"C,D|E",
                r"C,D|F",
            ],
        ),
    ],
)
def test_find_triplets(tree, labels, triplets):
    t = Tree(tree, labels)
    assert all(triplet in triplets for triplet in t.triplets)
