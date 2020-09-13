from poker.utils import argmax


def test_argmax():

    values = [0, 1, 2]
    assert argmax(values) == [2]

    values = [0, 1, 2, 1, 1]
    assert argmax(values) == [2]

    values = [0, 1, 1]
    assert argmax(values) == [1, 2]

    values = [0, 100, 5, 200, 99, 200]
    assert argmax(values) == [3, 5]

    values = [0, 55, 99, 99]
    assert argmax(values) == [2, 3]
