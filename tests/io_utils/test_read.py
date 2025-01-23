"""Tests for Reading Functions
"""

import os
import pytest
import numpy as np

from tables_io import io_utils
from tests.testUtils import check_deps
from tables_io.lazy_modules import h5py, pq, pq


@pytest.mark.skipif(not check_deps([h5py, pq]), reason="Missing HDF5 or parquet")
def test_pandas_read(h5_data_file, parquet_data_file):
    """Test the pandas reading from HDF5 and Parquet files"""
    _ = io_utils.read.read_H5_to_dict(h5_data_file)
    _ = io_utils.read.read_pq_to_dict(parquet_data_file)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_no_groupname(h5_no_group_file):
    """Test the load_training_data function for an HDF5 file with no groupname"""
    _ = io_utils.read.read_HDF5_to_dict(h5_no_group_file, groupname=None)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_get_group_names(h5_test_outfile):
    """Create a mock file with some data, ensure that the group names are read correctly."""
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    md_dict = dict(zgrid=zgrid)
    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf, md=md_dict)

    io_utils.write.write_dict_to_HDF5(data_dict, h5_test_outfile, groupname=None)

    group_names = io_utils.read.read_HDF5_group_names(h5_test_outfile)
    assert len(group_names) == 3
    assert "md" in group_names
    assert "zmode" in group_names
    assert "pz_pdf" in group_names

    subgroup_names = io_utils.read.read_HDF5_group_names(h5_test_outfile, "md")
    assert "zgrid" in subgroup_names

    with pytest.raises(KeyError) as excinfo:
        _ = io_utils.read.read_HDF5_group_names(h5_test_outfile, "dummy")
        assert "dummy" in str(excinfo.value)

    os.unlink(h5_test_outfile)
