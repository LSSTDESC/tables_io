""" Multi-proccesor utils """

from .lazy_modules import mpi4py


def getMPICom(mpiCom=None):
    """ Get an MPI comuunicator

    Parameters
    ----------
    mpiCom : `mpi4py.Communicator` or `bool` or `None`

    Returns
    -------
    theCom : `mpi4py.Communicator` or `None`

    Notes
    -----
    if mpiCom is 'False' or `None` this will return `None`
    if mpicom is `True`, this will return `mpi4py.MPI.COMM_WORLD`
    in any other case it will return mpiCom
    """
    if mpiCom is None:
        return None
    if isinstance(mpiCom, bool):
        if mpiCom:
            from mpi4py import MPI
            return MPI.COMM_WORLD
        return None
    return mpiCom


def getSizeAndRank(mpiCom=None):
    """ Get the size and rank of the current process

    Parameters
    ----------
    mpiCom : `mpi4py.Communicator` or `bool` or `None`

    Returns
    -------
    (size, rank) : `tuple`, (`int`, `int`)

    Notes
    -----
    if mpiCom is 'False' or `None` this will return (1, 0)
    in any other case it will return mpiCom.Get_size(), mpiCom.Get_rank()
    """
    theComm = getMPICom(mpiCom)
    if theComm is None:
        return (1, 0)
    return (theComm.Get_size(), theComm.Get_rank())


def getMPIActive(i, mpiCom=None):
    """ Test to see if the current process is 'active' for a loop interation index

    Parameters
    ----------
    i : `int`
        A loop interation index

    mpiCom : `mpi4py.Communicator` or `bool` or `None`

    Returns
    -------
    active : `bool`

    Notes
    -----
    if mpiCom is 'False' or `None` this will return `True`
    in any other case it will return `i % size == rank`
    """
    if mpiCom is None:
        return True
    size, rank = getSizeAndRank(mpiCom)
    return i % size == rank


def getH5FileMPIKwargs(mpiCom=None):
    """ Return the keyword arguments to open a h5py.File under MPIO

    Parameters
    ----------
    mpiCom : `mpi4py.Communicator` or `bool` or `None`

    Returns
    -------
    kwds : `dict`
        The keywords to pass to h5py.File()

    Notes
    -----
    if mpiCom is 'False' or `None` this will return {}
    """
    theCom = getMPICom(mpiCom)
    if theCom is None:
        return {}
    return dict(driver="mpio", comm=theCom)
