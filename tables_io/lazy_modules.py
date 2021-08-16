""" Lazy loading modules """

import sys
import importlib.util


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
        if spec is None:   #pragma: no cover
            print("Can't find module %s" % modulename)
            return None
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

HAS_TABLES = tables is not None
HAS_PQ = pq is not None
HAS_FITS = fits is not None
HAS_ASTROPY = apTable is not None
HAS_HDF5 = h5py is not None
HAS_PANDAS = pd is not None
