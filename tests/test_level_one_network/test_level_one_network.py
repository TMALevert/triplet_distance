def test_different_triplets(network1, network2):
    assert network1 != network2
    assert len([triplet for triplet in network1.triplets if triplet not in network2.triplets]) > 0
    assert len([triplet for triplet in network2.triplets if triplet not in network1.triplets]) > 0

def test_numb_spanning_trees(network1, network2):
    assert len(network1.spanning_trees) == 2
    assert len(network2.spanning_trees) == 2

def test_correctness_spanning_trees(network1, network2):
    for spanning_tree in network1.spanning_trees:
        assert all(triplet in network1.triplets for triplet in spanning_tree.triplets)
    for spanning_tree in network2.spanning_trees:
        assert all(triplet in network2.triplets for triplet in spanning_tree.triplets)