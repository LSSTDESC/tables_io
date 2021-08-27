""" Lazy loading modules """

import sys
import importlib.util


class DeferredModuleError:
    """ Class to throw an error if you try to use a modules that wasn't loaded """

    def __init__(self, moduleName):
        self._moduleName = moduleName

    @property
    def moduleName(self):
        """ Return the name of the module this is associated to """
        return self._moduleName

    def __getattr__(self, item):
        raise ImportError("Module %s was not loaded, so call to %s.%s fails" %
                          (self.moduleName, self.moduleName, item))



def lazyImport(modulename):
    """ This will allow us to lazy import various modules

    Parameters
    ----------
    modulename : `str`
        The name of the module in question

    Returns
    -------
    module : `importlib.LazyModule`
        A lazy loader for the module in question
    """
    try:
        return sys.modules[modulename]
    except KeyError:
        spec = importlib.util.find_spec(modulename)
        if spec is None:
            print("Can't find module %s" % modulename)
            return DeferredModuleError(modulename)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)
        # Make module with proper locking and get it inserted into sys.modules.
        loader.exec_module(module)
        try:
            _ = dir(module)
        except ValueError:
            pass
    return module


tables = lazyImport('tables')
apTable = lazyImport('astropy.table')
fits = lazyImport('astropy.io.fits')
h5py = lazyImport('h5py')
pd = lazyImport('pandas')
pq = lazyImport('pyarrow.parquet')
mpi4py = lazyImport('mpi4py')

HAS_TABLES = not isinstance(tables, DeferredModuleError)
HAS_PQ = not isinstance(pq, DeferredModuleError)
HAS_FITS = not isinstance(fits, DeferredModuleError)
HAS_ASTROPY = not isinstance(apTable, DeferredModuleError)
HAS_HDF5 = not isinstance(h5py, DeferredModuleError)
HAS_PANDAS = not isinstance(pd, DeferredModuleError)
HAS_PARQUET = not isinstance(pq, DeferredModuleError)
HAS_MPI = not isinstance(mpi4py, DeferredModuleError)
