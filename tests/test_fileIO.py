""" Unit tests for the fileIO module """

import os
import numpy as np
import pytest

from tables_io import io
from tables_io.testUtils import check_deps
from tables_io.lazy_modules import apTable, fits, pd, h5py, pq, pq


h5_data_file = "tests/data/pandas_test_hdf5.h5"
parquet_data_file = "tests/data/parquet_test.parquet"
no_group_file = "tests/data/no_groupname_test.hdf5"

test_outfile = "./test_out.h5"


@pytest.mark.skipif(not check_deps([h5py, pq]), reason="Missing HDF5 or parquet")
def test_pandas_readin():
    """Test the pandas reading"""
    _ = io.readH5ToDict(h5_data_file)
    _ = io.readPqToDict(parquet_data_file)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_no_groupname():
    """Test the load_training_data function for a file with no groupname"""
    _ = io.readHdf5ToDict(no_group_file, groupname=None)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_get_input_data_length_hdf5():
    """Test the get_input_data_size_hdf5 function"""
    assert io.getInputDataLengthHdf5(no_group_file) == 10
    try:
        _ = io.getInputDataLengthHdf5(h5_data_file, "df")
    except ValueError:
        pass
    else:
        raise ValueError("Failed to catch ValueError for mismatched column lengths")

@pytest.mark.skipif(not check_deps([h5py, pq]), reason="Missing HDF5 or parquet")
def test_get_input_data_length():
    """Test the get_input_data_size function"""
    assert io.getInputDataLength(parquet_data_file) == 10
    assert io.getInputDataLength(no_group_file) == 10
    with pytest.raises(NotImplementedError):
        io.getInputDataLength("dummy.fits") 

@pytest.mark.skipif(not check_deps([pq]), reason="Missing parquet")
def test_get_input_data_length_pq():
    """Test the get_input_data_size_pq function"""
    assert io.getInputDataLengthPq(parquet_data_file) == 10

@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_iter_chunk_hdf5_data():
    """Test the hdf5 data chunk iterator"""
    for itr in io.iterHdf5ToDict(no_group_file, chunk_size=2):
        for val in itr[2].values():
            assert np.size(val) == 2

    for itr in io.iterHdf5ToDict(no_group_file, chunk_size=3):
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

    groups, outf = io.initializeHdf5Write(
        test_outfile, data=dict(photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4"))
    )
    io.writeDictToHdf5Chunk(groups, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf")
    io.finalizeHdf5Write(outf, "md", zgrid=zgrid)
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

    groups, outf = io.initializeHdf5Write(
        test_outfile, data=dict(photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4")), comm=comm
    )
    io.writeDictToHdf5Chunk(groups, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf")
    io.finalizeHdf5Write(outf, "md", zgrid=zgrid)
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

    group, outf = io.initializeHdf5WriteSingle(
        test_outfile, "data", photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4")
    )
    io.writeDictToHdf5ChunkSingle(group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf")
    io.finalizeHdf5Write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)

    group, outf = io.initializeHdf5WriteSingle(
        test_outfile, None, photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4")
    )
    io.writeDictToHdf5ChunkSingle(group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf")
    io.finalizeHdf5Write(outf, "md", zgrid=zgrid)

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

    io.writeDictToHdf5(data_dict, test_outfile, groupname=None)

    group_names = io.readHdf5GroupNames(test_outfile)
    assert len(group_names) == 3
    assert "md" in group_names
    assert "zmode" in group_names
    assert "pz_pdf" in group_names

    subgroup_names = io.readHdf5GroupNames(test_outfile, "md")
    assert "zgrid" in subgroup_names

    with pytest.raises(KeyError) as excinfo:
        _ = io.readHdf5GroupNames(test_outfile, "dummy")
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

    group, outf = io.initializeHdf5WriteSingle(
        test_outfile, "data", photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4"), comm=comm
    )
    io.writeDictToHdf5ChunkSingle(group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf")
    io.finalizeHdf5Write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)

    group, outf = io.initializeHdf5WriteSingle(
        test_outfile, None, photoz_mode=((npdf,), "f4"), photoz_pdf=((npdf, nbins), "f4"), comm=comm
    )
    io.writeDictToHdf5ChunkSingle(group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf")
    io.finalizeHdf5Write(outf, "md", zgrid=zgrid)

    os.unlink(test_outfile)
