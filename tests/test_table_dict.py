"""
Unit tests for io_layer module
"""

import os

import unittest

from tables_io import types, convert, TableDict
from tables_io.testUtils import compare_table_dicts, make_test_data


class TableDictTestCase(unittest.TestCase):  #pylint: disable=too-many-instance-attributes
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
        odict_c = convert(self._tables, tType)
        td_c = TableDict(odict_c)
        filepath = td_c.write(basepath, fmt)
        if keys is None:
            self._files.append(filepath)
        else:
            for key in keys:
                self._files.append("%s%s.%s" % (basepath, key, fmt))
        td_r = TableDict.read(filepath, tType=tType, fmt=fmt, keys=keys)
        tables_r = td_r.convert(types.AP_TABLE)
        assert compare_table_dicts(self._tables, tables_r)
        if fmt in ['pq', 'h5']:
            return
        basepath2 = "%s_v2" % basepath
        filepath2 = td_c.write(basepath2)
        self._files.append(filepath2)
        td_r2 = TableDict.read(filepath2, tType=tType)
        tables_r2 = td_r2.convert(types.AP_TABLE)
        assert compare_table_dicts(self._tables, tables_r2)

    def testFitsLoopback(self):
        """ Test writing / reading to FITS """
        self._do_loopback(types.AP_TABLE, 'test_td_out', 'fits')

    def testHf5Loopback(self):
        """ Test writing / reading astropy tables to HDF5 """
        self._do_loopback(types.AP_TABLE, 'test_td_out', 'hf5')

    def testHdf5Loopback(self):
        """ Test writing / reading numpy arrays to HDF5 """
        self._do_loopback(types.NUMPY_DICT, 'test_td_out', 'hdf5')

    def testH5Loopback(self):
        """ Test writing / reading pandas dataframes to HDF5 """
        self._do_loopback(types.PD_DATAFRAME, 'test_td_out', 'h5')

    def testPQLoopback(self):
        """ Test writing / reading pandas dataframes to parquet """
        self._do_loopback(types.PD_DATAFRAME, 'test_td_out', 'pq', list(self._tables.keys()))

    def testBadType(self):
        """ Test adding bad types to TableDict """
        td = TableDict(self._tables)
        try:
            td['aa'] = 4
        except TypeError:
            pass
        else:
            raise TypeError("Failed to catch TypeError")

if __name__ == '__main__':
    unittest.main()
