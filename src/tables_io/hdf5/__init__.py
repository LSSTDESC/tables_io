from .. import io_utils
from ..utils.array_utils import getGroupInputDataLength

# Exposing HDF5 Interfaces


initialize_HDF5_write = io_utils.write.initializeHdf5Write

write_dict_to_HDF5_chunk = io_utils.write.writeDictToHdf5Chunk

write_dicts_to_HDF5 = io_utils.write.writeDictsToHdf5

finalize_HDF5_write = io_utils.write.finalizeHdf5Write

read_HDF5_group = io_utils.read.readHdf5Group

read_HDF5_group_names = io_utils.read.readHdf5GroupNames

read_HDF5_to_dict = io_utils.read.readHdf5ToDict

read_HDF5_group_to_dict = io_utils.read.readHdf5GroupToDict

read_HDF5_dataset_to_array = io_utils.read.readHdf5DatasetToArray

get_group_input_data_length = getGroupInputDataLength

# Convenience Functions for MPI

data_ranges_by_rank = io_utils.iterator.data_ranges_by_rank
