"""
Unit tests for metadata facilities (tablePlus module)
"""

import unittest
from copy import deepcopy

import pandas as pd
import numpy as np
from astropy.table import Table as ApTable

from tables_io.tablePlus import TablePlus
from tables_io.convUtils import convertToRecarray

class MetadataTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Make a table of each supported type
        '''
        d = {}
        d['col1'] = np.array([1,2,3])
        d['col2'] = np.array([4.1,5.2,6.3])

        self._npDict = TablePlus(d, name='npDict')
        self._df = TablePlus(pd.DataFrame(deepcopy(d)), name='dataFrame')
        self._apTable = TablePlus(ApTable(deepcopy(d)), name='apTable')
        recA = convertToRecarray(ApTable(deepcopy(d)))
        self._npRecArray = TablePlus(recA, name='recArray')
        self._globalMeta = {'a' : 1, 'b' : 2}

        self._tables = [self._npDict, self._df,
                        self._apTable, self._npRecArray]

    def tearDown(self):
        pass                  # nothing to do

    def testNative(self):
        '''
        Verify that TablePlus properly delegates to native type
        '''
        assert self._npDict['col1'][0] == 1
        assert self._df['col1'][0] == 1
        assert self._apTable['col1'][0] == 1
        assert self._npRecArray['col1'][0] == 1
        cols_df = self._df.columns
        cols_ap = self._apTable.colnames
        maxes = self._df.max()

        #dfPlus = tablePlus.TablePlus(df, 'dfTable')
        #print(dir(dfPlus))

    def testTableMeta(self):
        for t in self._tables:
            t.setTableMeta(self._globalMeta)

        for t in self._tables:
            assert t.getTableMeta() == self._globalMeta

    ## Still to do:  column meta; type conversion

if __name__ == '__main__':
    unittest.main()
