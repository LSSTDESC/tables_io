""" Handle table plus associated metadata"""

from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .lazy_modules import apTable, pd

from tables_io import types, convUtils

# Map (ttypeFrom, ttypeTo) to conversion routine

def _custom_dir(c, add):
    '''
    dir should return
       * functions defined for c's class
       * instance variables belonging to c
       * public stuff from the contained object (some table-like thing)
    '''
    return dir(type(c)) + list(c.__dict__.keys()) + add

class DelegatorBase:
    '''
    This class allows derived classes to delegate operations to a
    member.   Here it is used to delegate table operations to
    a table-like member
    '''

    @property
    def _delegate(self):
        pub = [o for o in dir(self.default) if not o.startswith('_')]
        return pub
    def __getattr__(self, k):
        if k in self._delegate:
            return getattr(self.default, k)
        raise AttributeError(k)
    def __dir__(self):
        return _custom_dir(self, self._delegate)

    def __init__(self):
        pass

class TablePlus(DelegatorBase):
    """
    Instance of the class represents a table in one of the supported
    formats plus (optional) associated metadata.
    """
    _apReadonly = ['dtype', 'name']
    _apPredefined = ['unit', 'format', 'description']

    def __init__(self, tbl, name=None):
        '''
        Parameters
        ----------
        tbl      An instance of one of the supported table types
        name     Name to associate with the table
        '''
        # If tbl is not appropriate, following will raise an exception
        self._tableType = types.tableType(tbl)

        if self._tableType == types.AP_TABLE:
            self._columnMeta = None
            self._tableMeta = None
        else:
            self._columnMeta = {}   # columnMeta
            self._tableMeta = {}    # tableMeta

        self._name = name
        self._tbl = tbl

        super().__init__()

        @property
        def tableType(self):
            return self._tableType

        @property
        def tableName(self):
            return self._name

        # Not sure this one is a good idea. Or necessary
        ##@property
        ##def rawTable(self):
        ##    return self._tbl

        # Native table methods called on this object will be delegated
        # to the table
        self.default = tbl

    def __getitem__(self, k):
        '''
        Specific override of __getitem__, delegating to table member
        so that [ ] syntax will be handled properly
        '''
        return self._tbl.__getitem__(k)

    def _checkColumnMeta(d):
        '''
        Check that each value key is a simple type
        '''
        for v in d.items():
            if not isinstance(v, (int, float, str, bool)):
                raise ValueError(f'Value {v} is not of simple type')

    def _syncColumnMeta(self):
        '''
        See if all column metadata is associated with an actual column.
        If not, delete entries for non-columns
        Returns:
        True if all column metadata is ok
        False if deletions were needed to sync
        '''

        sync = True
        if self._tableType == types.AP_TABLE:    # nothing to do
            return sync

        metaColumns = list(self._columnMeta.keys())
        actualColumns = self.getColumnNames()

        for m in metaColumns:
            if m not in actualColumns:
                sync = False
                del self._columnMeta[m]
        return sync

    def _validateTableMeta(self, meta):
        # Check that meta is convertible to something which can be
        # written to yaml or json; say require it to be dict or list
        # If not raise an exception
        pass

    def getColumnNames(self):
        '''
        Return a set of column names belonging to the underlying table
        '''
        if self._tableType == types.NUMPY_DICT:
            return set(self._tbl.keys())
        if self._tableType == types.NUMPY_RECARRAY:
            return set(self._tbl.dtype.fields.keys())
        if self._tableType == types.PD_DATAFRAME:
            return set(self._tbl.columns)
        if self._tableType == types.AP_TABLE:
            return set(self._tbl.colnames)
        return NotImplementedError('Unsupported native table type')

    def addColumnMeta(self, columnName, d):
        '''
        Add or update metadata (key,value pairs) for a column
        Parameters
        ----------
        columnName     A column in the table
        d              Dict-like where keys and values are of simple types
        '''
        if not columnName in self.getColumnNames():
            raise ValueError('Cannot add metadata for non-existent column')
        TablePlus._checkColumnMeta(d)

        # For astropy use native mechanism
        # Built-in attributes are name, unit, dtype, description, format
        # and meta (ordered dict).   Treat name and dtype as read-only.
        if self._tableType == types.AP_TABLE:
            dCopy = deepcopy(d)
            for k in TablePlus._apReadonly:
                if k in dCopy:
                    raise ValueError(
                        f'addColumnMeta: Changing value of {k} not allowed')
            for k in TablePlus._apPredefined:
                if k in dCopy:
                    self.setattr(k, d[k])
                    del dCopy[k]
            self.meta.update(dCopy)
            return

        if columnName in self._columnMeta:
            self._columnMeta[columnName].update(d)
        else:
            self._columnMeta[columnName] = d

    def getColumnMeta(self, columnName):
        '''
        Get metadata associated with a particular column
        returns:
        a dict
        '''
        if self._tableType == types.AP_TABLE:
            c = self[columnName]
            d = deepcopy(c.meta)
            for k in TablePlus._apPredefined:
                d[k] = c.getattr(k)
            return d

        return self._columnMeta.get(columnName)

    def getTableMeta(self):
        """
        Return a dict of table metadata
        """
        if self._tableType == types.AP_TABLE:
            return self._tbl.meta
        return self._tableMeta

    def setTableMeta(self, meta):
        '''
        Stash the meta.  If there already was a reference table metadata
        it will be overwritten
        '''
        self._validateTableMeta(meta)
        if self._tableType == types.AP_TABLE:
            self._tbl.meta = meta
        else:
            self._tableMeta = meta

    def convertTo(self, tType):
        '''
        Convert a table-with-metadata of one type to another.
        Use convUtils to convert the table itself. Metadata only
        needs conversion if from- to to-type is AP_TABLE
        '''
        if tType == self._tableType:
            return self

        tabPlus = TablePlus(convUtils.convertObj(self, tType))
        print(type(self._columnMeta))

        # Converting to AP_TABLE
        if tType == types.AP_TABLE:
            tabPlus._tbl.meta = self._tableMeta
            for c,d in self._columnMeta.items():
                tabPlus.addColumnMeta(c, d)
            return tabPlus

        # Converting from AP_TABLE
        if self._tableType == types.AP_TABLE:
            tabPlus._tableMeta == self._tbl.meta
            for c in self.getColumnNames():
                d = self._tbl[c].meta
                for k in TablePlus._apPredefined:
                    d[k] = self._tbl[c].getattr(k)
                tabPlus._columnMeta[c] = d
            return tabPlus

        # Otherwise just copy metadata
        tabPlus._columnMeta = deepcopy(self._columnMeta)
        tabPlus._tableMeta = deepcopy(self._tableMeta)
        return tabPlus
