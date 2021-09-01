#! /usr/bin/env python

import numpy as np

from tables_io import io

MPI_COM = True

test_outfile = './test_mpi.hdf5'

def check_mpi(mpiCom=MPI_COM):

    npdf = 40
    nbins = 21
    chunk_size = 5

    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    nChunk = npdf // chunk_size    

    group, outf = io.initializeHdf5Write(test_outfile, 'data', mpiCom=mpiCom, photoz_mode=((npdf,), 'f4'), photoz_pdf=((npdf, nbins), 'f4'))
    for i in range(nChunk):
        if not getMPIActive(i, True):
            continue
        start = i*chunk_size
        end = min(start+chunk_size, npdf)
        print("Writing %i %i %i" % (i, start, end))
        io.writeDictToHdf5Chunk(group, data_dict, start, end, zmode='photoz_mode', pz_pdf='photoz_pdf')
    io.finalizeHdf5Write(outf, 'md', mpiCom=mpiCom, zgrid=zgrid)


if __name__ == '__main__':
    check_mpi()
