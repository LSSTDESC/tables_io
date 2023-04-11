""" Lazy loading modules """

import importlib

class LazyModule:
    """
    A partial implementation of a lazily imported module.

    This isn't a general solution, since it doesn't actually
    create a module-like object, but the LazyLoader solution
    in importlib seems not to cope with submodules like astropy.table,
    or pyarrow.parquet, and always imports the full module.
    """
    def __init__(self, name):
        self.name = name
        self._module = None

    @property
    def module(self):
        if self._module is not None:
            return self._module
        try:
            self._module = importlib.import_module(self.name)
        except ImportError as err:
            raise ImportError(f"Cannot use selected data format, {self.name} not available") from err

        return self._module


    def __dir__(self):
        """
        Get the attributes of the module, to support tab-autocomplete.
        """
        return dir(self.module)

    def __getattr__(self, item):
        """Get something from the module"""
        return getattr(self.module, item)


def lazyImport(modulename):
    """
    Lazily load a module
    """
    return LazyModule(modulename)



tables = lazyImport('tables')
apTable = lazyImport('astropy.table')
apDiffUtils = lazyImport('astropy.utils.diff')
fits = lazyImport('astropy.io.fits')
h5py = lazyImport('h5py')
pd = lazyImport('pandas')
pq = lazyImport('pyarrow.parquet')
jnp = lazyImport('jax.numpy')
