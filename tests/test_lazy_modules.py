import pytest

from tests.testUtils import compare_table_dicts, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp

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
