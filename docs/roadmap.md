# Roadmap

## This version

### Breaking changes

- code reorganization means that some io functions that were previously accessed are no longer available in that place, i.e. `initialize_HDF5_write`. Some of these functions have been made available in the created `hdf5` module
