from itertools import combinations

import pytest

from rooted_triplet_distance import GeneralTriplet


@pytest.mark.parametrize(
    "triplet, possible_root, branches, relation",
    [
        (r"1|2,3", set(), [{"1"}, {"2", "3"}], (None, ("1", (None, ("2", "3"))))),
        (r"1|2|3", set(), [{"1"}, {"2"}, {"3"}], (None, ("1", "2", "3"))),
        (r"1/2|3", set(), [{"1", "2"}, {"3"}], (None, ("3", ("2", tuple("1"))))),
        (r"1/2/3", {"3"}, [{"1", "2", "3"}], ("3", ("2", tuple("1")))),
        (r"1/2\3", {"2"}, [{"1"}, {"3"}], ("2", ("1", "3"))),
        (r"1|2\3", set(), [{"1"}, {"3", "2"}], (None, ("1", ("2", tuple("3"))))),
        (r"1,2|3", set(), [{"1", "2"}, {"3"}], (None, ("3", (None, ("1", "2"))))),
        (r"1\2\3", {"1"}, [{"1", "2", "3"}], ("1", ("2", tuple("3")))),
    ],
)
def test_init(triplet, possible_root, branches, relation):
    t = GeneralTriplet(triplet)
    assert t.labels == {"1", "2", "3"}
    assert t.type == triplet
    assert t._tree_relation == relation
    for branch in t._branches:
        assert branch in branches
    assert t._possible_root == possible_root


@pytest.mark.parametrize(
    "triplet, type, possible_root, branches, relation",
    [
        (r"11|22,33", r"1|2,3", set(), [{"11"}, {"22", "33"}], (None, ("11", (None, ("22", "33"))))),
        (r"11|22|33", r"1|2|3", set(), [{"11"}, {"22"}, {"33"}], (None, ("11", "22", "33"))),
        (r"11/22|33", r"1/2|3", set(), [{"11", "22"}, {"33"}], (None, ("33", ("22", tuple({"11"}))))),
        (r"11/22/33", r"1/2/3", {"33"}, [{"11", "22", "33"}], ("33", ("22", tuple({"11"})))),
        (r"11/22\33", r"1/2\3", {"22"}, [{"11"}, {"33"}], ("22", ("11", "33"))),
        (r"11|22\33", r"1|2\3", set(), [{"33", "22"}, {"11"}], (None, ("11", ("22", tuple({"33"}))))),
        (r"11,22|33", r"1,2|3", set(), [{"11", "22"}, {"33"}], (None, ("33", (None, ("11", "22"))))),
        (r"11\22\33", r"1\2\3", {"11"}, [{"11", "22", "33"}], ("11", ("22", tuple({"33"})))),
    ],
)
def test_init_longer_labels(triplet, type, possible_root, branches, relation):
    t = GeneralTriplet(triplet)
    assert t.labels == {"11", "22", "33"}
    assert t.type == type
    assert t._tree_relation == relation
    for branch in t._branches:
        assert branch in branches
    assert t._possible_root == possible_root


def test_equal_type_error():
    with pytest.raises(TypeError):
        GeneralTriplet("A|B|C") == 1


@pytest.mark.parametrize(
    "triplet, other, equal",
    [
        ("A|B|C", "A|B|C", True),
        ("A,B|C", "C|A,B", True),
        ("B,A|C", "C|A,B", True),
        ("A|B,C", "A|C,B", True),
        ("A,B|C", "A|B|C", False),
        ("A,B|C", "A,B|D", False),
        ("A|B|C", "C|A|B", True),
        ("A|C|B", "B|C|A", True),
        ("B|C|B", "B|C|A", False),
        (r"1/2|3", r"3|2\1", True),
        (r"1/2|3", r"3|1\2", False),
        (r"1/2|3", r"3/2\1", False),
        (r"1/2|3", r"3|2|1", False),
        (r"1/2|3", r"3|2,1", False),
        (r"1/2/3", r"3\2\1", True),
        (r"1/2/3", r"2\3\1", False),
        (r"1/2/3", r"3/2\1", False),
        (r"1/2/3", r"2/1|3", False),
        (r"1/2/3", r"3|2\1", False),
        (r"1/2\3", r"3/2\1", True),
        (r"1/2\3", r"3/1\2", False),
        (r"1/2\3", r"3/2/1", False),
        (r"1/2\3", r"3\2\1", False),
        (r"1/2\3", r"3|2\1", False),
        (r"1|2\3", r"3/2|1", True),
        (r"1|2\3", r"3,2|1", False),
        (r"1|2\3", r"3|2|1", False),
        (r"1|2\3", r"3/2\1", False),
        (r"1|2\3", r"3/2/1", False),
        (r"1|2\3", r"3/2|1", True),
        (r"1|2\3", r"3,2|1", False),
        (r"1|2\3", r"3|2|1", False),
        (r"1|2\3", r"3/2/1", False),
        (r"1|2\3", r"3/2\1", False),
    ],
)
def test_equal(triplet, other, equal: bool):
    assert (GeneralTriplet(triplet) == GeneralTriplet(other)) is equal


@pytest.mark.parametrize(
    "triplet, element, contains",
    [
        ("A|B|C", "A", True),
        ("A|B|C", "B", True),
        ("A|B|C", "C", True),
        ("A|B|C", "D", False),
        ("A,B|C", "A", True),
        ("A,B|C", "B", True),
        ("A,B|C", "C", True),
        ("A,B|C", "D", False),
        ("A|B,C", "A", True),
        ("A|B,C", "B", True),
        ("A|B,C", "C", True),
        ("A|B,C", "D", False),
        (r"1/2|3", "1", True),
        (r"1/2|3", "2", True),
        (r"1/2|3", "3", True),
        (r"1/2|3", "4", False),
        (r"1/2/3", "1", True),
        (r"1/2/3", "2", True),
        (r"1/2/3", "3", True),
        (r"1/2/3", "4", False),
        (r"1/2\3", "1", True),
        (r"1/2\3", "2", True),
        (r"1/2\3", "3", True),
        (r"1/2\3", "4", False),
        (r"1|2\3", "1", True),
        (r"1|2\3", "2", True),
        (r"1|2\3", "3", True),
        (r"1|2\3", "4", False),
        (r"1\2\3", "1", True),
        (r"1\2\3", "2", True),
        (r"1\2\3", "3", True),
        (r"1\2\3", "4", False),
    ],
)
def test_contains(triplet, element, contains: bool):
    assert (element in GeneralTriplet(triplet)) is contains


@pytest.mark.parametrize(
    "triplet",
    [
        "A|B|C",
        "A,B|C",
        "A|B,C",
        r"1/2|3",
        r"1/2/3",
        r"1/2\3",
        r"1|2\3",
        r"1\2\3",
    ],
)
def test_iter(triplet):
    t = GeneralTriplet(triplet)
    for element in t:
        assert element in t


def test_hash():
    triplet1 = GeneralTriplet("A|B|C")
    triplet2 = GeneralTriplet("A,B|C")
    triplet3 = GeneralTriplet("A|B,C")
    triplet4 = GeneralTriplet("A/B|C")
    triplet5 = GeneralTriplet("A/B/C")
    triplet6 = GeneralTriplet(r"A/B\C")
    triplet7 = GeneralTriplet(r"A|B\C")
    triplet8 = GeneralTriplet(r"A\B\C")
    triplets = [triplet1, triplet2, triplet3, triplet4, triplet5, triplet6, triplet7, triplet8]
    for triplet in triplets:
        assert triplet.__hash__() == triplet.__hash__()
    for triplet_i, triplet_j in combinations(triplets, 2):
        assert triplet_i.__hash__() != triplet_j.__hash__()
