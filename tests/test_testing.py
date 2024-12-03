import pytest

from tables_io import io


def test_read():
    breakpoint()
    tab = io.read("./tests/data/no_groupname_test.hdf5")

    assert len(tab) > 0
