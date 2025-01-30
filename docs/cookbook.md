# Cookbook

## Basic table operations

### Read in a file to a specific format

### Write out a file to a specific file format

### Get information about a file without reading it all into memory (io_open)

### Converting tables to different formats

## Iteration

### Iterating through data in an HDF5 file

To iterate through an `HDF5` file, yielding only a section of data at a time, we can use the `iterator` or `iterator_native` functions as shown below. You can provide the size of the data section you would like as an
`int` to `chunk_size`, size here meaning the number of rows (or length of the `numpy.arrays` in the case of `numpyDict` tables). The default `chunk_size` is 100,000.

To output the data to a specific tabular type, then use `iterator` as shown below:

```{doctest}

>>> for start, stop, data in tables_io.iterator('datafile.hdf5','pandasDataFrame',chunk_size=3):
>>>    print(start,stop,data)
0 3    col1  col2
0     1     5
1     2    10
2     3    15
3 6    col1  col2
0     4    20
1     5    25
2     6    30
6 7    col1  col2
0     7    35

```

## Handling HDF5 files

### Chunked read of an HDF5 file

### Writing an hdf5 file with MPI

### slicing tables

### concatenate tables
