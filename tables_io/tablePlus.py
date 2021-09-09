import pandas as pd
import numpy as np

from tables_io import types

def _custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add

class MetadataBase(object):
    '''
    This class
    1.  Includes utilities for handling metadata
    2.  Allows derived classes to delegate table operations to a
        standard table of one of the supported types
    '''

    @property
    def _delegate(self):
        pub = [o for o in dir(self.default) if not o.startswith('_')]
        return pub
    def __getattr__(self, k):
        if k in self._delegate:
            return getattr(self.default, k)
        if k in self._metadataAttributes:
            return getattr(self, k)
        raise AttributeError(k)
    def __dir__(self):
        return _custom_dir(self, self._delegate)

    def _checkColumnMeta(self, d):
        # Check that each value key is a simple type
        for v in d.items():
            if not isinstance(v, (int, float, str, bool)):
                raise ValueException(f'Value {v} is not of simple type')
    def addColumnMeta(self, columnName, d):
        self._checkColumnMeta(d)
        if columnName in self._columnMeta:
            self._columnMeta[columnName].update(d)
        else:
            self._columnMeta[columnName] = d

    def getColumnMeta(self, columnName):
        return self._columnMeta.get(columnName)

    def __init__(self, columnMeta={}, tableMeta={}):
        self._columnMeta = columnMeta
        self._tableMeta = tableMeta
        self._metadataAttributes = [self.addColumnMeta, self.getColumnMeta]

class TablePlus(MetadataBase):
    def __init__(self, tbl, name):
        '''
        Parameters
        ----------
        tbl      An instance of one of the supported table types
        name     Name to associate with the table
        '''
        # If tbl is not appropriate, following will raise an exception
        self._tableType = types.tableType(tbl)

        self._name = name
        self._tbl = tbl

        super().__init__()

        @property
        def tableType(self):
            return self._tableType

        # Should the name be kept here or in MetadataPlus?
        @property
        def tableName(self):
            return self._name

        # Not sure this one is a good idea
        @property
        def rawTable(self):
            return self._tbl

        # Native table methods called on this object will be delegated
        # to the table
        self.default = tbl

    def someFunc(self):
        print('Hello from TablePlus.someFunc')

    # Cannot delegate special methods; must override explicitly
    def __getitem__(self, k):
        return self._tbl.__getitem__(k)
