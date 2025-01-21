from .. import io_utils
from ..utils.arrayUtils import getGroupInputDataLength

# Exposing HDF5 Interfaces


initialize_HDF5_write = io_utils.write.initializeHdf5Write

write_dict_to_HDF5_chunk = io_utils.write.writeDictToHdf5Chunk

finalize_HDF5_write = io_utils.write.finalizeHdf5Write

read_HDF5_group = io_utils.read.readHdf5Group

get_group_data_input_length = getGroupInputDataLength

# Convenience Functions for MPI

data_ranges_by_rank = io_utils.iterator.data_ranges_by_rank
