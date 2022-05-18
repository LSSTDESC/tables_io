""" Unit tests for the fileIO module """

import os
import numpy as np

from tables_io import io

h5_data_file = 'tests/data/pandas_test_hdf5.h5'
parquet_data_file = 'tests/data/parquet_test.parquet'
no_group_file = 'tests/data/no_groupname_test.hdf5'

test_outfile = './test_out.h5'


def test_pandas_readin():
    """ Test the pandas reading """
    _ = io.readH5ToDict(h5_data_file)
    _ = io.readPqToDict(parquet_data_file)


def test_no_groupname():
    """ Test the load_training_data function for a file with no groupname """
    _ = io.readHdf5ToDict(no_group_file, groupname=None)

def test_get_input_data_length_hdf5():
    """ Test the get_input_data_size_hdf5 function """
    assert io.getInputDataLengthHdf5(no_group_file) == 10
    try:
        _ = io.getInputDataLengthHdf5(h5_data_file, 'df')
    except ValueError:
        pass
    else:
        raise ValueError("Failed to catch ValueError for mismatched column lengths")


def test_iter_chunk_hdf5_data():
    """ Test the hdf5 data chunk iterator """
    for itr in io.iterHdf5ToDict(no_group_file, chunk_size=2):
        for val in itr[2].values():
            assert np.size(val) == 2

    for itr in io.iterHdf5ToDict(no_group_file, chunk_size=3):
        for val in itr[2].values():
            assert np.size(val) <= 3


def test_write_output_file():
    """ Test writing an output file """
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = {'data': dict(zmode=zmode, pz_pdf=pz_pdf)}

    groups, outf = io.initializeHdf5Write(test_outfile, data = dict(photoz_mode=((npdf,), 'f4'), photoz_pdf=((npdf, nbins), 'f4')))
    io.writeDictToHdf5Chunk(groups, data_dict, 0, npdf, zmode='photoz_mode', pz_pdf='photoz_pdf')
    io.finalizeHdf5Write(outf, 'md', zgrid=zgrid)

    os.unlink(test_outfile)
    
def test_write_output_file_single():
    """ Test writing an output file """
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    group, outf = io.initializeHdf5WriteSingle(test_outfile, 'data', photoz_mode=((npdf,), 'f4'), photoz_pdf=((npdf, nbins), 'f4'))
    io.writeDictToHdf5ChunkSingle(group, data_dict, 0, npdf, zmode='photoz_mode', pz_pdf='photoz_pdf')
    io.finalizeHdf5Write(outf, 'md', zgrid=zgrid)

    os.unlink(test_outfile)

    group, outf = io.initializeHdf5WriteSingle(test_outfile, None, photoz_mode=((npdf,), 'f4'), photoz_pdf=((npdf, nbins), 'f4'))
    io.writeDictToHdf5ChunkSingle(group, data_dict, 0, npdf, zmode='photoz_mode', pz_pdf='photoz_pdf')
    io.finalizeHdf5Write(outf, 'md', zgrid=zgrid)

    os.unlink(test_outfile)

    
