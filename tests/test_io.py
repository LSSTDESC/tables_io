"""
Unit tests for io_layer module
"""

import os

import numpy as np
import unittest
from tables_io import types, forceTo, read, write
from tables_io.testUtils import compare_table_dicts, make_test_data

from astropy.table import Table as apTable
from astropy.utils.diff import report_diff_values


class IoTestCase(unittest.TestCase):  #pylint: disable=too-many-instance-attributes
    """ Test the utility functions """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self._tables = make_test_data()        
        self._files = []

    def tearDown(self):
        """ Clean up any mock data files created by the tests. """
        for ff in self._files:
            if os.path.exists(ff):
                os.unlink(ff)

    def _do_loopback(self, tType, basepath, fmt, keys=None):
        """ Utility function to do loopback tests """
        odict_c = forceTo(self._tables, tType)
        filepath = write(odict_c, basepath, fmt)
        if keys is None:
            self._files.append(filepath)
        else:
            for key in keys:
                self._files.append("%s%s.%s" % (basepath, key, fmt))
        odict_r = read(filepath, tType=tType, fmt=fmt, keys=keys)
        tables_r = forceTo(odict_r, types.AP_TABLE)
        assert compare_table_dicts(self._tables, tables_r)

    def testFitsLoopback(self):
        """ Test writing / reading to FITS """
        self._do_loopback(types.AP_TABLE, 'test_out', 'fits')

    def testHf5Loopback(self):
        """ Test writing / reading astropy tables to HDF5 """
        self._do_loopback(types.AP_TABLE, 'test_out', 'hf5')

    def testHdf5Loopback(self):
        """ Test writing / reading numpy arrays to HDF5 """
        self._do_loopback(types.NUMPY_DICT, 'test_out', 'hdf5')

    def testH5Loopback(self):
        """ Test writing / reading pandas dataframes to HDF5 """
        self._do_loopback(types.PD_DATAFRAME, 'test_out', 'h5')

    def testPQLoopback(self):
        """ Test writing / reading pandas dataframes to parquet """
        self._do_loopback(types.PD_DATAFRAME, 'test_out', 'pq', list(self._tables.keys()))


if __name__ == '__main__':
    unittest.main()
