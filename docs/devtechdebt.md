# Technical Debt

On this page, current known issues and technical debt will be recorded. These issues would make
good first contributions for new contributors to the project. You can also take a look at the [issues](https://github.com/LSSTDESC/tables_io/issues) page for more.

## Code Debt

- [ ] [`iter_H5_to_dataframe`](#tables_io.io_utils.iterator.iter_H5_to_dataframe) is not implemented, iterator cannot iterate over `h5` files
- [ ] [`pa_table_to_recarray`](#tables_io.conv.conv_table.pa_table_to_recarray) is not implemented, but the functionality is being accomplished via converting to astropy Table first
- [ ] Migrate `tests/io_utils/test_io_classic.py` to `pytest` fully, and remove `unittest` import.

### Deprecation

The following functions and classes are deprecated, and should be removed along _some timeline_.

- [ ] [`TableDict`](#tables_io.table_dict) - not currently in use in the code
- [ ] [`writeNative`](#tables_io.writeNative) - replaced by [`write_native`](#tables_io.io_utils.write.write_native)
- [ ] [`readNative`](#tables_io.readNative) - replaced by [`read_native`](#tables_io.io_utils.read.read_native)
- [ ] [`iteratorNative`](#tables_io.iteratorNative) - replaced by [`iterator_native`](#tables_io.io_utils.iterator.iterator_native)
- [ ] [`convertObj`](#tables_io.convertObj) - replaced by [`convert_table`](#tables_io.conv.conv_table.convert_table)
- [ ] [`concatObjs`](#tables_io.concatObjs) - replaced by [`concat_table`](#tables_io.utils.concat_utils.concat_table)
- [ ] [`concat`](#tables_io.concat) - replaced by [`concat_tabledict`](#tables_io.utils.concat_utils.concat_tabledict)
- [ ] [`sliceObj`](#tables_io.sliceObj) - replaced by [`slice_table`](#tables_io.utils.slice_utils.slice_table)
- [ ] [`sliceObjs`](#tables_io.sliceObjs) - replaced by [`slice_tabledict`](#tables_io.utils.slice_utils.slice_tabledict)
