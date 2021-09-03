========================================
Tablular data reading/writing interfaces
========================================

`tables_io` is intended to provide interfaces between a few types of
tabular data supported in astronomy, in particular in the LSST Dark
Energy Science Collaboration (DESC).

It is designed to be extendable, so that additional interfaces can
be supported down the road if there is a strong desire to do so.


Tabular data interfaces
-----------------------

The four most common tabular data interfaces used in the LSST DESC
are:

1. 'astropy tables', specifically `astropy.table.Table`

2. 'dictionaries of arrays', specifically `Mapping`, (`str`,
   `numpy.array`)

3. 'numpy record arrays`, specifically `numpy.recarray`
 
4. 'pandas dataframes', specifcially `pandas.DataFrame`



File formats
------------

The three most common file formats for used to store tables in the
LSST DESC are, 'FITS', 'HDF5' and 'parquet'.  However, because of the
flexbility provided by 'HDF5', there are different 'HDF5' structures
that are used for storing tables.   In all this leaves us with
effectively 5 and a half different file formats

1. 'FITS', used to store astropy tables, e.g., as produced by:

.. code-block:: python

    out_list = [fits.PrimaryHDU()]
    out_list.append(fits.table_to_hdu(table))
    fits.HDUList(out_list).writeto('out.fits')


    or as used to store numpy record arrays, e.g., as produced by:

.. code-block:: python

    out_list = [fits.PrimaryHDU()]
    out_list.append(fits.BinTableHDU.from_columns(rec.columns))
    fits.HDUList(out_list).writeto('out.fits')
   
    
2. 'HDF5', as used to store astopy tables, e.g., as produced by:

.. code-block:: python

    hdulist.writeto('out.hd5', format='hdf5')

3. 'HDF5', as used to store numpy arrays e.g., as produced by:

.. code-block:: python
		
    fout = h5py.File(filepath, 'a')		
    group = fout.create_group(groupname)
    for key, val in odict.items():
        group.create_dataset(key, dtype=val.dtype, data=val.data)

	
4. 'HDF5', as used to store pandas dataframes, e.g., as produced by: 

.. code-block:: python
		
    df.to_hdf(filepath, key)

    
5. 'parquet', as used to store pandas datafames, e.g., as produced by:

   
.. code-block:: python
		
    df.to_parquet(filepath)



Quick examples
==============

Here are some very quick high-level examples.


Table conversion
----------------

Conversion between different types of tables can be cone with the
`tables_io.convert` function.

.. code-block:: python

    # table is an astropy table

    # To convert it to a pandas dataframe
    df = tables_io.convert(table, tables_io.types.PD_DATAFRAME)

    # To convert it to a dictionary of numpy arrays
    adict = tables_io.convert(table, tables_io.types.NUMPY_DICT)

    
Table Writing
-------------

The `tables_io.write` function provides a unified interface for
writing files.

.. code-block:: python

    # table is an astropy table

    # To write it in its 'native' format, i.e., the default for
    # astropy tables, which is 'hf5', i.e., 'hdf5' for astropy
    tables_io.write(table, 'out')

    # To write it in a different format, e.g., parquet
    tables_io.write(table, 'out', 'pq')



Table Reading
-------------

The `tables_io.read` function provides a unified interface for
reading files.

.. code-block:: python

    # filepath is a file with tabular data

    # To read it to it's native format e.g., astropy tables for 'hf5' files
    table = tables_io.read(filepath)

    # To read it to a different format
    df = tables_io.read(filepath, tables_io.types.PD_DATAFRAME)

    # If the file suffix doesn't match the expectation
    table = tables_io.read('data.hdf5', fmt='hd5')



Iterating on Tables
-------------------

The `tables_io.iterate` function provides a unified interface for
iterating on table-like objects in files.

.. code-block:: python

    # filepath is a file with tabular data

    # To read it to it's native format e.g., numpy dicts for hdf5 files
    for start, stop, data in tables_io.iterate(filepath):
        ...

    # To read it to a different format
    for start, stop, data in tables_io.read(filepath, tables_io.types.PD_DATAFRAME):
        ...
	


Multiple Tables
---------------

The `tables_io.TableDict` class provides an interface for dealing with
multiple different shaped tables as a single object.

.. code-block:: python

    # data and metadata are two differently shaped astropy tables

    # Building a `TableDict` object
    td = tables_io.TableDict(dict(data=data, meta=metadata))

    # Converting all the tables
    td_np = td.convert(td, tables_io.types.NUMPY_DICT)

    # Writing the tables
    filepath = td_pd.write()

    # Reading the tables
    td_read_np = td.read(filepath)


Documentation Contents
----------------------

.. toctree::
   :includehidden:
   :maxdepth: 3

   install
   tables_io
   
