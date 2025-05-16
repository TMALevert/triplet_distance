import pytest

from rooted_triplet_distance.level_one_network import Network


@pytest.fixture(scope="module")
def network1():
    network_dict = {
        "p": {"1": {"a": {}, "2": {"b": {"e": {}}, "d": {}}}, "3": {"f": {}, "4": {"g": {"h": {}}, "d": {"c": {}}}}}
    }
    network_labels = ["a", "b", "c", "e", "p", "d", "f", "g", "h", "1"]
    return Network(network_dict, network_labels)


@pytest.fixture(scope="module")
def network2():
    network_dict = {
        "p": {
            "1": {"a": {}, "2": {"b": {"e": {}}, "d": {"c": {}}}},
            "3": {"f": {}, "4": {"g": {"h": {}}, "b": {"e": {}}}},
        }
    }
    network_labels = ["a", "b", "c", "e", "p", "d", "f", "g", "h", "1"]
    return Network(network_dict, network_labels)


@pytest.fixture(scope="module")
def network1_no_root():
    network_dict = {
        "p": {"1": {"a": {}, "2": {"b": {"e": {}}, "d": {}}}, "3": {"f": {}, "4": {"g": {"h": {}}, "d": {"c": {}}}}}
    }
    network_labels = ["a", "b", "c", "e", "d", "f", "g", "h", "1"]
    return Network(network_dict, network_labels)


@pytest.fixture(scope="module")
def network2_no_root():
    network_dict = {
        "p": {
            "1": {"a": {}, "2": {"b": {"e": {}}, "d": {"c": {}}}},
            "3": {"f": {}, "4": {"g": {"h": {}}, "b": {"e": {}}}},
        }
    }
    network_labels = ["a", "b", "c", "e", "d", "f", "g", "h", "1"]
    return Network(network_dict, network_labels)
