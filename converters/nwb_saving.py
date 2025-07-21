import os
from datetime import datetime
import gc
gc.collect()

from pynwb import NWBHDF5IO


def save_nwb_file(nwb_file, output_folder):
    """
    Save nwb file to output folder.
    Args:
        nwb_file: NWB file object
        output_folder: output folder path
    Returns:
        nwb_path: Path to the saved NWB file
    """
    nwb_name = nwb_file.identifier + ".nwb"

    with NWBHDF5IO(os.path.join(output_folder, nwb_name), 'w') as io:
        io.write(nwb_file)

    #print("NWB file created at :\n " + str(os.path.join(output_folder, nwb_name)))

    gc.collect()
    return str(os.path.join(output_folder, nwb_name))
