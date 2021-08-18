""" Unit tests for the fileIO module """

import numpy as np

from tables_io import arrayUtils


def test_array_length():
    """ Test the pandas reading """
    assert arrayUtils.arrayLength(4) == 0
    assert arrayUtils.arrayLength(np.ones(5)) == 5
    assert arrayUtils.arrayLength(np.ones((5, 5, 5))) == 5
    assert arrayUtils.arrayLength([3, 4, 4]) == 3
