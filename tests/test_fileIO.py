""" Unit tests for the fileIO module """

import pytest
import os
import numpy as np

from tables_io import io_layer

h5_data_file = 'tests/data/pandas_test_hdf5.h5'
parquet_data_file = 'tests/data/parquet_test.parquet'
no_group_file = 'tests/data/no_groupname_test.hdf5'

test_outfile = './test_out.h5'


def test_pandas_readin():
    """ Test the  """
    _ = io_layer.readFileToArrayDict(h5_data_file)
    _ = io_layer.readFileToArrayDict(parquet_data_file)


def test_no_groupname():
    """ Test the load_training_data function for a file with no groupname """
    _ = io_layer.readFileToArrayDict(no_group_file, groupname='None')

def test_get_input_data_size_hdf5():
    """ Test the get_input_data_size_hdf5 function """
    assert io_layer.getInputDataSizeHdf5(no_group_file) == 10
    assert io_layer.getInputDataSizeHdf5(h5_data_file, 'df') == 14

def test_missing_file_ext():
    """ Test that we refuse to read csv files """
    with pytest.raises(NotImplementedError):
        _ = io_layer.readFileToArrayDict(no_group_file, fmt='csv')

def test_iter_chunk_hdf5_data():
    """ Test the hdf5 data chunk iterator """
    for itr in io_layer.iterChunkHdf5Data(no_group_file, chunk_size=2):
        for val in itr[2].values():
            assert np.size(val) == 2

    for itr in io_layer.iterChunkHdf5Data(no_group_file, chunk_size=3):
        for val in itr[2].values():
            assert np.size(val) <= 3


def test_write_output_file():
    """ Test writing an output file """
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    outf = io_layer.initializeHdf5Writeout(test_outfile, photoz_mode=((npdf,), 'f4'), photoz_pdf=((npdf, nbins), 'f4'))
    io_layer.writeoutHdf5Chunk(outf, data_dict, 0, npdf, zmode='photoz_mode', pz_pdf='photoz_pdf')
    io_layer.finalizeHdf5Writeout(outf, zgrid=zgrid)

    os.unlink(test_outfile)
