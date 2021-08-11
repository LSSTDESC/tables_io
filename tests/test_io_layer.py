"""
Unit tests for PDF class
"""
import numpy as np
import unittest
import tables_io

from astropy.table import Table as apTable
from astropy.io import fits

import pandas as pd
import h5py
import pyarrow.parquet as pq


def compare_table_dicts(d1, d2):
    identical = True
    for k, v in d1.items():
        try:
            vv = d2[k]
        except KeyError:
            vv = d2[k.upper()]
        identical |= report_diff_values(v, vv)
    return identical


class IoLayerTestCase(unittest.TestCase):
    """ Test the utility functions """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self._nrow = 1000
        self._vect_size = 20
        self._mat_size = 5
        self._scalar = np.random.uniform(self._nrow)
        self._vect = np.random.uniform(self._nrow*self._vect_size).reshape(self._nrow, self._vect_size)
        self._matrix = np.random.uniform(self._nrow*self._mat_size*self._mat_size).reshape(self._nrow, self._mat_size, self._mat_size)
        self._data = dict(scalar=self._scalar, vect=self._vect, matrix=self._matrix)
        self._table = apTable(self._data)
        self._table.meta['a'] = 1
        self._table.meta['b'] = None
        self._table.meta['c'] = [3, 4, 5]
        self._small_table = apTable(dict(a=np.ones(21), b=np.zeros(21)))
        self._small_table.meta['small'] = True
        self._tables = dict(data=self._table, md=self._small_table)
        self._files = []


    def tearDown(self):
        """ Clean up any mock data files created by the tests. """
        for ff in self._files:
            os.unlink(ff)

    def testFitsLoopback(self):
        
        tables_io.writeTablesToFits(self._tables, "test_out.fits", overwrite=True)
        tables_r_fits = tables_io.readFitsToTables("test_out.fits")
        assert compare_table_dicts(self._tables, tables_r_fits)

    def testHdf5Loopback(self):
        
        tables_io.writeTablesToHdf5(self._tables, "test_out.fits", overwrite=True)
        tables_r_hdf5 = tables_io.readHdf5ToTables("test_out.hdf5")
        assert compare_table_dicts(self._tables, tables_r_hdf5)

    def testParquetLoopback(self):

        dataframe_in = tables_io.tablesToDataframes(self._tables)
        tables_io.writeDataframesToPq(dataframe_in, "test_out")
        dataframes_in = tables_io.readPqToDataframes("test_out", keys=['data', 'md']))
        tables_r_pq = tables_io.dataframesToTables(dataframes_in)
        assert compare_table_dicts(self._tables, tables_r_pq)


if __name__ == '__main__':
    unittest.main()
