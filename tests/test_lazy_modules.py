import pytest

from tests.helpers.utilities import compare_table_dicts, check_deps
from tables_io.lazy_modules import (
    tables,
    apTable,
    apDiffUtils,
    fits,
    h5py,
    pd,
    pq,
    jnp,
    lazyImport,
)

# TODO: Docstrings for all of these functions


@pytest.mark.parametrize(
    "mod",
    [tables, apTable, apDiffUtils, fits, h5py, pd, pq],
)
def test_deps(mod):
    assert check_deps([mod])


@pytest.mark.skipif(not check_deps([jnp]), reason="Failed to load jax.numpy")
def test_deps_jnp():
    assert check_deps([jnp])


def test_bad_deps():
    dummy = 0
    assert not check_deps([dummy])


def test_check_deps():
    """Testing the Lazy Import functionality: fails on importing module that doesn't exist"""
    bad_module = lazyImport("this_does_not_exist")
    assert not check_deps([bad_module])
