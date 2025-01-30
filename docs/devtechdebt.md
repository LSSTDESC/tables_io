# Technical Debt

On this page, current known issues and technical debt will be recorded. These issues would make
good first contributions for new contributors to the project.

## Code Debt

- `iter_H5_to_dataframe` is not implemented, iterator cannot iterate over `h5` files
- `pa_table_to_recarray` is not implemented, but the functionality is being accomplished via converting to astropy Table first

## Organizational Debt

### Deprecation

The following functions and classes are deprecated, and should be removed along _some timeline_.

- `TableDict` - not currently in use in the code
- `writeNative` - replaced by `write_native`
- `readNative` - replaced by `read_native`
- `iteratorNative` - replaced by `iterator_native`
- `convertObj` - replaced by `convert_table`
- `concatObj` - replaced by `concat_table`
- `concat` - replaced by `concat_tabledict`
- `sliceObj` - replaced by `slice_table`
- `sliceObjs` - replaced by `slice_tabledict`

## Requested Features
