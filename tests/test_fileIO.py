""" Unit tests for the fileIO module """

import os
import numpy as np
import pytest

from tables_io import io_utils
from tests.testUtils import check_deps
from tables_io.lazy_modules import apTable, fits, pd, h5py, pq, pq


h5_data_file = "tests/data/pandas_test_hdf5.h5"
parquet_data_file = "tests/data/parquet_test.parquet"
no_group_file = "tests/data/no_groupname_test.hdf5"

test_outfile = "./test_out.h5"


@pytest.mark.skipif(not check_deps([h5py, pq]), reason="Missing HDF5 or parquet")
def test_pandas_readin():
    """Test the pandas reading"""
    _ = io_utils.read.read_H5_to_dict(h5_data_file)
    _ = io_utils.read.read_pq_to_dict(parquet_data_file)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_no_groupname():
    """Test the load_training_data function for a file with no groupname"""
    _ = io_utils.read.read_HDF5_to_dict(no_group_file, groupname=None)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_get_input_data_length_hdf5():
    """Test the get_input_data_size_hdf5 function"""
    assert io_utils.iterator.get_input_data_length_HDF5(no_group_file) == 10
    try:
        _ = io_utils.iterator.get_input_data_length_HDF5(h5_data_file, "df")
    except ValueError:
        pass
    else:
        raise ValueError("Failed to catch ValueError for mismatched column lengths")


@pytest.mark.skipif(not check_deps([h5py, pq]), reason="Missing HDF5 or parquet")
def test_get_input_data_length():
    """Test the get_input_data_size function"""
    assert io_utils.iterator.get_input_data_length(parquet_data_file) == 10
    assert io_utils.iterator.get_input_data_length(no_group_file) == 10
    with pytest.raises(NotImplementedError):
        io_utils.iterator.get_input_data_length("dummy.fits")


@pytest.mark.skipif(not check_deps([pq]), reason="Missing parquet")
def test_get_input_data_length_pq():
    """Test the get_input_data_size_pq function"""
    assert io_utils.iterator.get_input_data_length_pq(parquet_data_file) == 10


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_iter_chunk_hdf5_data():
    """Test the hdf5 data chunk iterator"""
    for itr in io_utils.iterator.iter_HDF5_to_dict(no_group_file, chunk_size=2):
        for val in itr[2].values():
            assert np.size(val) == 2

    for itr in io_utils.iterator.iter_HDF5_to_dict(no_group_file, chunk_size=3):
        for val in itr[2].values():
            assert np.size(val) <= 3


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_file():
    """Test writing an output file"""
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = {"data": dict(zmode=zmode, pz_pdf=pz_pdf)}

    groups, outf = io_utils.write.initialize_HDF5_write(
        test_outfile,
        data=dict(photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4")),
    )
    io_utils.write.write_dict_to_HDF5_chunk(
        groups, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)
    os.unlink(test_outfile)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_parallel_file():
    from mpi4py import MPI

    """ Testing parallel write """
    comm = MPI.COMM_WORLD
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = {"data": dict(zmode=zmode, pz_pdf=pz_pdf)}

    groups, outf = io_utils.write.initialize_HDF5_write(
        test_outfile,
        data=dict(photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4")),
        comm=comm,
    )
    io_utils.write.write_dict_to_HDF5_chunk(
        groups, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)
    os.unlink(test_outfile)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_file_single():
    """Test writing an output file"""
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        test_outfile,
        "data",
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        test_outfile,
        None,
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_get_group_names():
    """Create a mock file with some data, ensure that the group names are read correctly."""
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    md_dict = dict(zgrid=zgrid)
    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf, md=md_dict)

    io_utils.write.write_dict_to_HDF5(data_dict, test_outfile, groupname=None)

    group_names = io_utils.read.read_HDF5_group_names(test_outfile)
    assert len(group_names) == 3
    assert "md" in group_names
    assert "zmode" in group_names
    assert "pz_pdf" in group_names

    subgroup_names = io_utils.read.read_HDF5_group_names(test_outfile, "md")
    assert "zgrid" in subgroup_names

    with pytest.raises(KeyError) as excinfo:
        _ = io_utils.read.read_HDF5_group_names(test_outfile, "dummy")
        assert "dummy" in str(excinfo.value)

    os.unlink(test_outfile)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_parallel_file_single():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    """ Test writing an output file """
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        test_outfile,
        "data",
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
        comm=comm,
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        test_outfile,
        None,
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
        comm=comm,
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)
