"""
Unit tests for io_layer module
"""

import os

import unittest
from tables_io import types, convert, read, write
from tables_io.testUtils import compare_table_dicts, compare_tables, make_test_data


class IoTestCase(unittest.TestCase):  #pylint: disable=too-many-instance-attributes
    """ Test the utility functions """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self._tables = make_test_data()
        self._table = self._tables['data']
        self._files = []

    def tearDown(self):
        """ Clean up any mock data files created by the tests. """
        for ff in self._files:
            if os.path.exists(ff):
                os.unlink(ff)

    def _do_loopback(self, tType, basepath, fmt, keys=None):
        """ Utility function to do loopback tests """
        odict_c = convert(self._tables, tType)
        filepath = write(odict_c, basepath, fmt)
        if keys is None:
            self._files.append(filepath)
        else:
            for key in keys:
                self._files.append("%s%s.%s" % (basepath, key, fmt))
        odict_r = read(filepath, tType=tType, fmt=fmt, keys=keys)
        tables_r = convert(odict_r, types.AP_TABLE)
        assert compare_table_dicts(self._tables, tables_r)

    def _do_loopback_single(self, tType, basepath, fmt, keys=None):
        """ Utility function to do loopback tests """
        obj_c = convert(self._table, tType)
        filepath = write(obj_c, basepath, fmt)
        if keys is None:
            self._files.append(filepath)
        else:
            for key in keys:
                self._files.append("%s%s.%s" % (basepath, key, fmt))
        obj_r = read(filepath, tType=tType, fmt=fmt, keys=keys)
        table_r = convert(obj_r, types.AP_TABLE)
        assert compare_tables(self._table, table_r)
        if types.FILE_FORMAT_SUFFIXS[fmt] not in types.NATIVE_TABLE_TYPE:
            return

        basepath_native = "%s_native" % basepath
        filepath_native = write(obj_c, basepath_native)
        if keys is not None:
            filepath_native += ".pq"
        self._files.append(filepath_native)
        obj_r_native = read(filepath_native, tType=tType, keys=keys)
        table_r_native = convert(obj_r_native, types.AP_TABLE)
        assert compare_tables(self._table, table_r_native)


    def testFitsLoopback(self):
        """ Test writing / reading to FITS """
        self._do_loopback(types.AP_TABLE, 'test_out', 'fits')
        self._do_loopback_single(types.AP_TABLE, 'test_out_single', 'fits')

    def testHf5Loopback(self):
        """ Test writing / reading astropy tables to HDF5 """
        self._do_loopback(types.AP_TABLE, 'test_out', 'hf5')
        self._do_loopback_single(types.AP_TABLE, 'test_out_single', 'hf5')

    def testHdf5Loopback(self):
        """ Test writing / reading numpy arrays to HDF5 """
        self._do_loopback(types.NUMPY_DICT, 'test_out', 'hdf5')
        self._do_loopback_single(types.NUMPY_DICT, 'test_out_single', 'hdf5')

    def testH5Loopback(self):
        """ Test writing / reading pandas dataframes to HDF5 """
        self._do_loopback(types.PD_DATAFRAME, 'test_out', 'h5')
        self._do_loopback_single(types.PD_DATAFRAME, 'test_out_single', 'h5')

    def testPQLoopback(self):
        """ Test writing / reading pandas dataframes to parquet """
        self._do_loopback(types.PD_DATAFRAME, 'test_out', 'pq', list(self._tables.keys()))
        self._do_loopback_single(types.PD_DATAFRAME, 'test_out_single', 'pq', [''])

    def testBad(self):  #pylint: disable=no-self-use
        """ Test that bad calls to write are dealt with """
        try:
            write('aa', 'null', 'fits')
        except TypeError:
            pass
        else:
            raise TypeError("Failed to catch unwritable type")
        assert write(False, 'null', 'fits') is None

if __name__ == '__main__':
    unittest.main()
