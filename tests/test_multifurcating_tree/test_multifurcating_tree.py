import pytest

from rooted_triplet_distance import MultifurcatingTree


def test_graph_creation():
    tree = {
        "A": {
            "B": {"D": {"H": {}, "I": {}}, "E": {"J": {}, "K": {}}},
            "C": {"F": {"L": {}, "M": {}}, "G": {"N": {}, "O": {}}},
        }
    }
    t = MultifurcatingTree(tree)
    assert t.layers == {
        "0": ["A"],
        "1": ["B", "C"],
        "2": ["D", "E", "F", "G"],
        "3": ["H", "I", "J", "K", "L", "M", "N", "O"],
    }
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
            ["C|D|E", "C,D|A", "C,E|A", "D,E|A"],
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
            ],
        ),
    ],
)
def test_find_triplets(tree, labels, triplets):
    t = MultifurcatingTree(tree, labels)
    assert all(triplet in triplets for triplet in t.triplets)
