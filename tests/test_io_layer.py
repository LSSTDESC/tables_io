"""
Unit tests for io_layer module
"""

import os

import numpy as np
import unittest
from tables_io import io_layer

from astropy.table import Table as apTable
from astropy.utils.diff import report_diff_values


def compare_table_dicts(d1, d2):
    """ Compare all the tables in two dictionaries """
    identical = True
    for k, v in d1.items():
        try:
            vv = d2[k]
        except KeyError:
            vv = d2[k.upper()]
        identical |= report_diff_values(v, vv)
    return identical


class IoLayerTestCase(unittest.TestCase):  #pylint: disable=too-many-instance-attributes
    """ Test the utility functions """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self._nrow = 1000
        self._vect_size = 20
        self._mat_size = 5
        self._scalar = np.random.uniform(size=self._nrow)
        self._vect = np.random.uniform(size=self._nrow*self._vect_size).reshape(self._nrow, self._vect_size)
        self._matrix = np.random.uniform(size=self._nrow*self._mat_size*self._mat_size).reshape(self._nrow, self._mat_size, self._mat_size)
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
            try:
                os.unlink(ff)
            except:
                pass

    def testFitsLoopback(self):
        """ Test writing / reading to FITS """
        io_layer.writeTablesToFits(self._tables, "test_out.fits", overwrite=True)
        self._files.append('test_out.fits')
        tables_r_fits = io_layer.readFitsToTables("test_out.fits")
        assert compare_table_dicts(self._tables, tables_r_fits)

    def testHdf5Loopback(self):
        """ Test writing / reading to HDF5 """
        io_layer.writeTablesToHdf5(self._tables, "test_out.hdf5", overwrite=True)
        self._files.append('test_out.hdf5')
        tables_r_hdf5 = io_layer.readHdf5ToTables("test_out.hdf5")
        assert compare_table_dicts(self._tables, tables_r_hdf5)

    def testParquetLoopback(self):
        """ Test writign / reading to parquet """
        dataframe_in = io_layer.tablesToDataframes(self._tables)
        io_layer.writeDataframesToPq(dataframe_in, "test_out_")
        self._files.append('test_out_data.pq')
        self._files.append('test_out_md.pq')
        dataframes_in = io_layer.readPqToDataframes("test_out_", keys=['data', 'md'])
        tables_r_pq = io_layer.dataframesToTables(dataframes_in)
        assert compare_table_dicts(self._tables, tables_r_pq)


if __name__ == '__main__':
    unittest.main()
