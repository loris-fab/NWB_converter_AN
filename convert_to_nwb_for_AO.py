"""_summary_
"""
# Imports converters
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.Initiation_nwb
import converters.acquisition_to_nwb
import converters.units_to_nwb
import converters.intervals_to_nwb


# Imports libraries
from contextlib import redirect_stdout
from pynwb import NWBHDF5IO, validate
from pathlib import Path
from tqdm import tqdm
import numpy as np
import importlib
import argparse
import platform
import h5py
import os
import gc


############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################


def convert_data_to_nwb_an_mat(mat_file, output_folder):
    """
    Convert a single .mat file (AN session) into an NWB file.

    Parameters
    ----------
    mat_file : str
        Path to the .mat file to convert.
    output_folder : str
        Output directory where the NWB file will be saved.

    """

    # --- Load the .csv file ---------------------------------------------------
    csv_file = "Subject_Session_Selection.csv"

    # --- Load the .mat file ---------------------------------------------------
    # If "Data" group exists, use it; otherwise, use root.
    with h5py.File(mat_file, 'r') as f:
        data_group = f['Data'] if 'Data' in f else f
        data = {key: data_group[key][()] for key in data_group.keys()}
    
        if "VideoOnsets" in data and "TrialOnsets_All" in data: # Small modification for video onsets
            if abs(np.max(data["TrialOnsets_All"] - data["VideoOnsets"])) < 1e-3:
                data["VideoOnsets"] = data["TrialOnsets_All"]

    # ---  NWB conversion for this session -------------------------------
    # Creating config file for NWB conversion
    Rewarded = converters.Initiation_nwb.Rewarded_or_not(mat_file=data, csv_file=csv_file)

    if Rewarded:
        output_path, config_file = converters.Initiation_nwb.files_to_config_Rewarded(data,csv_file=csv_file, output_folder=output_folder)
    else:
        output_path, config_file = converters.Initiation_nwb.files_to_config_NonRewarded(data,csv_file=csv_file, output_folder=output_folder)

    # Created NWB file 
    nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path) # same for rewarded and non-rewarded sessions                                      
    
    # Add general metadata
    signal, regions = converters.acquisition_to_nwb.extract_lfp_signal(data, mat_file)
    electrode_table_region, unique_values = converters.general_to_nwb.add_general_container(nwb_file=nwb_file, data=data, mat_file=mat_file, regions=regions) # same for rewarded and non-rewarded sessions

    # Add acquisition container
    converters.acquisition_to_nwb.add_lfp_acquisition(nwb_file=nwb_file, signal_array=signal, electrode_region=electrode_table_region) # same for rewarded and non-rewarded sessions  

    # Add intervall container
    if Rewarded:
        converters.intervals_to_nwb.add_intervals_container_Rewarded(nwb_file=nwb_file, data=data)
    else:
        converters.intervals_to_nwb.add_intervals_container_NonRewarded(nwb_file=nwb_file, data=data)

    # Add units container
    converters.units_to_nwb.add_units_container(nwb_file=nwb_file, data=data, unique_values=unique_values, mat_file=mat_file , regions=regions) # same for rewarded and non-rewarded sessions

    # Add processing container
    if Rewarded:
        converters.behavior_to_nwb.add_behavior_container_Rewarded(nwb_file=nwb_file, data=data, config=config_file)
    else:
        converters.behavior_to_nwb.add_behavior_container_NonRewarded(nwb_file=nwb_file, data=data, config_file=config_file)

    # Save NWB file
    output_folder = os.path.join(output_folder, "WR+") if Rewarded else os.path.join(output_folder, "WR-")
    os.makedirs(output_folder, exist_ok=True)
    nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)

    # Validating NWB file before saving...
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwb_errors = validate(io=io)
                # If validation errors occur, delete the invalid NWB file
    if nwb_errors:
        os.remove(nwb_path)
        raise RuntimeError("NWB validation failed: " + "; ".join(map(str, nwb_errors)))

    # Delete .yaml config file 
    if os.path.exists(output_path) and output_path.endswith('.yaml'):
        os.remove(output_path)



def convert_data_to_nwb_an(input_folder, output_folder):
    """
    Convert all .mat files in a folder into NWB format (AN sessions).

    Parameters
    ----------
    input_folder : str
        Folder containing the .mat files.
    output_folder : str
        Folder where NWB files will be saved.
    """
    # reload converters
    importlib.reload(converters.behavior_to_nwb)
    importlib.reload(converters.nwb_saving)
    importlib.reload(converters.general_to_nwb)
    importlib.reload(converters.Initiation_nwb)
    importlib.reload(converters.acquisition_to_nwb)
    importlib.reload(converters.units_to_nwb)
    importlib.reload(converters.intervals_to_nwb)

    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-NWB conversion_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

    assert os.path.exists(input_folder), "file input_folder does not exist"
    files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]
    # Iterate over each .mat file in the folder
    list_errors_files = []
    list_errors = []
    print("Converting data to NWB format for mouse:", list(set([f.split("_")[0] for f in files])))
    with tqdm(files, desc="Conversion .mat files") as pbar:
        for i, file in enumerate(pbar, start=1):
            full_path = os.path.join(input_folder, file)
            pbar.set_description(f"Conversion to NWB: 🔁 {file}")
            try:
                # Silence
                with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
                    # Run the conversion
                    convert_data_to_nwb_an_mat(mat_file=full_path,output_folder=output_folder)    
            except Exception as e:
                list_errors_files.append(file)
                list_errors.append(str(e))

            if i == len(files):
                pbar.set_description(f"Conversion to NWB is finished")
            gc.collect()


        if len(list_errors_files) > 0:
            print(f"⚠️ Conversion completed with errors for {len(list_errors_files)} files")
            for i, file in enumerate(list_errors_files):
                print(f"    - {file}: {list_errors[i]}")
            
        # Clean up any leftover config files (.yaml)
        for f in Path(output_folder).glob("*.yaml"):  
            f.unlink()

        print("**************************************************************************")


#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ MAIN _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to NWB format for AN sessions")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing .mat files")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the NWB files will be saved")


    args = parser.parse_args()
    convert_data_to_nwb_an(input_folder=args.input_folder, output_folder=args.output_folder)
