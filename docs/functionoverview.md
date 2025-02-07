# Function Overview

## Main functions

### Functions that read in data

| Function                                                                                | Description                                                                |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`read` {octicon}`link;0.9em`](#tables_io.io_utils.read.read)                           | Reads in a file to a `Table-like` or `TableDict-like` object               |
| [`read_native` {octicon}`link;0.9em`](#tables_io.io_utils.read.read_native)             | Reads in a file to a `TableDict-like` object of the native tabular type    |
| [`io_open` {octicon}`link;0.9em`](#tables_io.io_utils.read.io_open)                     | Reads a file as a file object                                              |
| [`check_columns` {octicon}`link;0.9em`](#tables_io.io_utils.read.check_columns)         | Checks that the a file has the same or fewer columns than a given list     |
| [`iterator` {octicon}`link;0.9em`](#tables_io.io_utils.iterator.iterator)               | Reads in data from a file one chunk at a time to a given format            |
| [`iterator_native` {octicon}`link;0.9em`](#tables_io.io_utils.iterator.iterator_native) | Reads in data from a file one chunk at a time to its native tabular format |

### Functions that write data

| Function                                                                       | Description                                                              |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| [`write` {octicon}`link;0.9em`](#tables_io.io_utils.write.write)               | Writes a `Table-like` or `TableDict-like` object to a file               |
| [`write_native` {octicon}`link;0.9em`](#tables_io.io_utils.write.write_native) | Writes a `Table-like` or `TableDict-like` object to its native file type |

### Other functions

| Function                                                                                   | Description                                                                     |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| [`convert` {octicon}`link;0.9em`](#tables_io.conv.conv_tabledict.convert)                  | Convert a `Table-like` or `TableDict-like` object to a specified tabular format |
| [`convert_table` {octicon}`link;0.9em`](#tables_io.conv.conv_table.convert_table)          | Convert a `Table-like` object to a specified tabular format                     |
| [`concat_tabledict` {octicon}`link;0.9em`](#tables_io.utils.concat_utils.concat_tabledict) | Concatenate multiple `TableDict-like` objects                                   |
| [`concat_table` {octicon}`link;0.9em`](#tables_io.utils.concat_utils.concat_table)         | Concatenate multiple `Table-like` objects                                       |
| [`slice_tabledict` {octicon}`link;0.9em`](#tables_io.utils.slice_utils.slice_tabledict)    | Slice a `TableDict-like` object                                                 |
| [`slice_table` {octicon}`link;0.9em`](#tables_io.utils.slice_utils.slice_table)            | Slice a `Table-like` object                                                     |
| [`get_table_type` {octicon}`link;0.9em`](#tables_io.types.get_table_type)                  | Returns the tabular type of a `Table-like` or `TableDict-like` object           |

## HDF5 module

The HDF5 module exists to allow users easier access to manually do chunked operations with `HDF5` files. While [`iterator`](#tables_io.io_utils.iterator.iterator) provides a way to do chunked reads of `HDF5` files, this is the only way to do chunked writes of `HDF5` files. To use the `hdf5` module run `from tables_io import hdf5`.

### Functions that read in data

| Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [`read_HDF5_group` {octicon}`link;0.9em`](#tables_io.io_utils.read.read_HDF5_group)                                                           | Read and return the requested group and file object from an `HDF5` file.       |
| [`read_HDF5_group_names` {octicon}`link;0.9em`](#tables_io.io_utils.read.read_HDF5_group_names)                                               | Read and return the list of group names from one level of an `HDF5` file       |
| [`read_HDF5_to_dict` {octicon}`link;0.9em`](#tables_io.io_utils.read.read_HDF5_to_dict)                                                       | Reads in data from an `HDF5` file to a dictionary                              |
| [`read_HDF5_group_to_dict` {octicon}`link;0.9em`](#tables_io.io_utils.read.read_HDF5_group_to_dict)                                           | Reads in data from an open `HDF5` file object to a dictionary or `numpy.array` |
| [`read_HDF5_dataset_to_array` {octicon}`link;0.9em`](#tables_io.io_utils.read.read_HDF5_dataset_to_array)                                     | Reads in all or part of an `HDF5` dataset                                      |

### Functions that write data

| Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`initialize_HDF5_write` {octicon}`link;0.9em`](#tables_io.io_utils.write.initialize_HDF5_write)                                | Prepares an `HDF5` file for output.                                                      |
| [`write_dict_to_HDF5_chunk` {octicon}`link;0.9em`](#tables_io.io_utils.write.write_dict_to_HDF5_chunk)                          | Writes a chunk of data from a `TableDict-like` object to an `HDF5` group or file object. |
| [`write_dicts_to_HDF5` {octicon}`link;0.9em`](#tables_io.io_utils.write.write_dicts_to_HDF5)                                    | Writes a `Table-like` object to an `HDF5` file given a filepath.                         |
| [`finalize_HDF5_write` {octicon}`link;0.9em`](#tables_io.io_utils.write.finalize_HDF5_write)                                    | Writes any last data and closes an `HDF5` file object.                                   |

### Utility functions

| Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`get_group_input_data_length` {octicon}`link;0.9em`](#tables_io.utils.array_utils.get_group_input_data_length)                               | Returns the length of an `HDF5` group object                                  |
| [`get_input_data_length` {octicon}`link;0.9em`](#tables_io.io_utils.iterator.get_input_data_length)                                           | Opens a file and gets the length of the first axis of the data in that file.  |
| [`data_ranges_by_rank` {octicon}`link;0.9em`](#tables_io.io_utils.iterator.data_ranges_by_rank)                                               | Given a number of rows and chunk size, yields the data range for each process |
