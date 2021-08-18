"""
Unit tests for io_layer module
"""

import numpy as np
import unittest
from tables_io import types, forceTo, forceObjTo
from tables_io.testUtils import compare_table_dicts, make_test_data

from astropy.table import Table as apTable
from astropy.utils.diff import report_diff_values

class ConvTestCase(unittest.TestCase):  #pylint: disable=too-many-instance-attributes
    """ Test the utility functions """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self._tables = make_test_data()
        self._table = self._tables['data']

    def tearDown(self):
        """ Clean up any mock data files created by the tests. """

    def _do_loopback(self, tType1, tType2):
        """ Utility function to do loopback tests """
        odict_1 = forceTo(self._tables, tType1)
        odict_2 = forceTo(odict_1, tType2)
        tables_r = forceTo(odict_2, types.AP_TABLE)
        assert compare_table_dicts(self._tables, tables_r)
        t1 = forceObjTo(self._table, tType1)
        t2 = forceObjTo(t1, tType2)
        _ = forceObjTo(t2, types.AP_TABLE)

    def testAstropyLoopback(self):
        """ Test writing / reading astropy tables to HDF5 """
        self._do_loopback(types.AP_TABLE, types.NUMPY_DICT)
        self._do_loopback(types.AP_TABLE, types.PD_DATAFRAME)

    def testNumpyLoopback(self):
        """ Test writing / reading numpy arrays to HDF5 """
        self._do_loopback(types.NUMPY_DICT, types.AP_TABLE)
        self._do_loopback(types.NUMPY_DICT, types.PD_DATAFRAME)

    def testPandasLoopback(self):
        """ Test writing / reading pandas dataframes to HDF5 """
        self._do_loopback(types.PD_DATAFRAME, types.AP_TABLE)
        self._do_loopback(types.PD_DATAFRAME, types.NUMPY_DICT)


if __name__ == '__main__':
    unittest.main()
