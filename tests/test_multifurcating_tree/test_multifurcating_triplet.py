import pytest

from rooted_triplet_distance import MultifurcatingTriplet


@pytest.mark.parametrize(
    "triplet, parts, labels",
    [
        ("A|B|C", {"A", "B", "C"}, {"A", "B", "C"}),
        ("A,B|C", {("A", "B"), "C"}, {"A", "B", "C"}),
        ("A|B,C", {"A", ("B", "C")}, {"A", "B", "C"}),
    ],
)
def test_init(triplet, parts, labels):
    t = MultifurcatingTriplet(triplet)
    assert t.parts == parts
    assert t.labels == labels


def test_apart():
    t = MultifurcatingTriplet("A|B|C")
    assert t.apart("A")
    assert t.apart("B")
    assert t.apart("C")
    assert t.apart("D")
    t = MultifurcatingTriplet("A,B|C")
    assert not t.apart("A")
    assert not t.apart("B")
    assert t.apart("C")
    assert t.apart("D")
    t = MultifurcatingTriplet("A|B,C")
    assert t.apart("A")
    assert not t.apart("B")
    assert not t.apart("C")
    assert t.apart("D")


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
    ],
)
def test_equal(triplet, other, equal: bool):
    assert (MultifurcatingTriplet(triplet) == MultifurcatingTriplet(other)) is equal


def test_equal_type_error():
    with pytest.raises(TypeError):
        MultifurcatingTriplet("A|B|C") == 1


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
    ],
)
def test_contains(triplet, element, contains: bool):
    assert (element in MultifurcatingTriplet(triplet)) is contains


@pytest.mark.parametrize(
    "triplet",
    [
        "A|B|C",
        "A,B|C",
        "A|B,C",
    ],
)
def test_iter(triplet):
    t = MultifurcatingTriplet(triplet)
    for element in t:
        assert element in t


def test_hash():
    triplet = MultifurcatingTriplet("A|B|C")
    triplet2 = MultifurcatingTriplet("A,B|C")
    triplet3 = MultifurcatingTriplet("A|B,C")
    assert triplet.__hash__() == triplet.__hash__()
    assert triplet2.__hash__() == triplet2.__hash__()
    assert triplet3.__hash__() == triplet3.__hash__()
    assert triplet.__hash__() != triplet2.__hash__()
    assert triplet.__hash__() != triplet3.__hash__()
    assert triplet2.__hash__() != triplet3.__hash__()
