"""_summary_
"""
import os
import h5py
import gc
import sys
import importlib
import argparse
import platform
from tqdm import tqdm
from pynwb import NWBHDF5IO, validate
from contextlib import redirect_stdout
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.Initiation_nwb
import converters.acquisition_to_nwb
import converters.units_to_nwb
import converters.analysis_to_nwb
import converters.intervals_to_nwb


############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################


def convert_data_to_nwb_an_mat(mat_file, output_folder):
    """
    Converts data from a config file to an NWB file.
    :param config_file: Path to the yaml config file containing mouse ID and metadata for the session to convert
    :param output_folder: Path to the folder to save NWB files
    :param psth_window: Tuple of two floats representing the start and end of the PSTH and LFP window in seconds
    :param psth_bin: Float representing the bin size for PSTH and LFP in seconds
    """
    csv_file = "Subject_Session_Selection.csv"

    # Load the .mat file 
    with h5py.File(mat_file, 'r') as f:
        data_group = f['Data'] if 'Data' in f else f
        data = {key: data_group[key][()] for key in data_group.keys()}

    print("**************************************************************************")
    print(f" NWB conversion {mat_file}")
    print(" ")
    print(f"📃 Creating config file for NWB conversion :")
    importlib.reload(converters.Initiation_nwb)
    Rewarded = converters.Initiation_nwb.Rewarded_or_not(mat_file=data, csv_file=csv_file)
    if Rewarded:
        output_path, config_file = converters.Initiation_nwb.files_to_config_Rewarded(data,csv_file=csv_file, output_folder=output_folder)
    else:
        output_path, config_file = converters.Initiation_nwb.files_to_config_NonRewarded(data,csv_file=csv_file, output_folder=output_folder)

    print("📑 Created NWB file :")
    importlib.reload(converters.general_to_nwb)
    print(config_file['session_metadata']["session_description"])
    nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path) # same for rewarded and non-rewarded sessions                                      
    
    print("     o 📌 Add general metadata")
    importlib.reload(converters.acquisition_to_nwb)
    signal, regions = converters.acquisition_to_nwb.extract_lfp_signal(data, mat_file)
    electrode_table_region, unique_values = converters.general_to_nwb.add_general_container(nwb_file=nwb_file, data=data, mat_file=mat_file, regions=regions) # same for rewarded and non-rewarded sessions
    print("         - Subject metadata")
    print("         - Session metadata")
    print("         - Device metadata")
    print("         - Extracellular electrophysiology metadata")
    
    print("     o 📶 Add acquisition container")
    converters.acquisition_to_nwb.add_lfp_acquisition(nwb_file=nwb_file, signal_array=signal, electrode_region=electrode_table_region) # same for rewarded and non-rewarded sessions  

    print("     o ⏸️ Add intervall container")
    importlib.reload(converters.intervals_to_nwb)
    if Rewarded:
        converters.intervals_to_nwb.add_intervals_container_Rewarded(nwb_file=nwb_file, data=data, mat_file=mat_file)
    else:
        converters.intervals_to_nwb.add_intervals_container_NonRewarded(nwb_file=nwb_file, data=data, mat_file=mat_file)

    print("     o 🧠 Add units container")
    importlib.reload(converters.units_to_nwb)
    sampling_rate =  30000
    converters.units_to_nwb.add_units_container(nwb_file=nwb_file, data=data, unique_values=unique_values, mat_file=mat_file , sampling_rate = sampling_rate , regions=regions) # same for rewarded and non-rewarded sessions

    print("     o ⚙️ Add processing container")
    importlib.reload(converters.behavior_to_nwb)
    if Rewarded:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_Rewarded(nwb_file=nwb_file, data=data, config=config_file)
    else:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_NonRewarded(nwb_file=nwb_file, data=data, config_file=config_file)

    print("         - No ephys data for AN sessions")

    importlib.reload(converters.nwb_saving)
    if Rewarded:
        output_folder = os.path.join(output_folder, "WR+")
        os.makedirs(output_folder, exist_ok=True)
        nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)
    else:
        output_folder = os.path.join(output_folder, "WR-")
        os.makedirs(output_folder, exist_ok=True)
        nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)
        
    print(" ")
    print("🔎 Validating NWB file before saving...")
    with NWBHDF5IO(nwb_path, 'r') as io:
        errors = validate(io=io)

    if not errors:
        print("     o ✅ File is valid, no errors detected.")
    else:
        print("     o ❌ Errors detected:")
        for err in errors:
            print("         -", err)
    print(" ")
    print("💾 Saving NWB file")
    if not errors:
        print("     o 📂 NWB file saved at:")
        print("         -", nwb_path)
    else:
        print("     o ❌ NWB file is invalid, deleting file...")
        os.remove(nwb_path)
    print("**************************************************************************")

    # Delete .yaml config file 
    if os.path.exists(output_path) and output_path.endswith('.yaml'):
        os.remove(output_path)


def convert_data_to_nwb_an(input_folder, output_folder,print_progress=False):
    """
    Converts all .mat files in a folder to NWB format for AN sessions.
    :param input_folder: Path to the folder containing .mat files
    :param output_folder: Path to the folder where NWB files will be saved
    """
    assert os.path.exists(input_folder), "file input_folder does not exist"
    files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]
    # Iterate over each .mat file in the folder
    list_errors_files = []
    list_errors = []
    i = 0
    with tqdm(files, desc="Conversion .mat files") as pbar:
        for file in pbar:
            i += 1
            full_path = os.path.join(input_folder, file)
            pbar.set_description(f"Conversion to NWB: 🔁 {file}")
            try:
                if print_progress:
                    convert_data_to_nwb_an_mat(mat_file=full_path,output_folder=output_folder)
                    def clear_console():
                        if platform.system() == "Windows":
                            os.system("cls")
                        else:
                            os.system("clear")
                    clear_console()
                else:
                    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
                        convert_data_to_nwb_an_mat(mat_file=full_path,output_folder=output_folder)
            except Exception as e:
                #print(f"⚠️ Error in {file} : {e}")
                list_errors_files.append(file)
                list_errors.append(str(e))
            if i == len(files):
                pbar.set_description(f"Conversion to NWB is finished")
            gc.collect() 
        if len(list_errors_files) > 0:
            print(f"⚠️ Conversion completed with errors for {len(list_errors_files)} files")
            for i, file in enumerate(list_errors_files):
                print(f"    - {file}: {list_errors[i]}")
        gc.collect() 
    return None

#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ MAIN _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to NWB format for AN sessions")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing .mat files")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the NWB files will be saved")
    parser.add_argument("--print_progress", action="store_true", help="Print progress of conversion")


    args = parser.parse_args()
    convert_data_to_nwb_an(input_folder=args.input_folder, output_folder=args.output_folder , print_progress=args.print_progress)
