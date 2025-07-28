"""_summary_
"""
import importlib
import os
import h5py
from pynwb import NWBHDF5IO, validate
import argparse
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


def convert_data_to_nwb_an(mat_file, output_folder,psth_window=(-0.2, 0.5), psth_bin=0.010 ):
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
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
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
    #else:
        #converters.intervals_to_nwb.add_intervals_container_NonRewarded(nwb_file=nwb_file, data=data, mat_file=mat_file)

    print("     o 🧠 Add units container")
    importlib.reload(converters.units_to_nwb)
    sampling_rate =  30000
    converters.units_to_nwb.add_units_container(nwb_file=nwb_file, data=data, unique_values=unique_values, mat_file=mat_file , sampling_rate = sampling_rate , regions=regions) # same for rewarded and non-rewarded sessions

    print("     o ⚙️ Add processing container")
    importlib.reload(converters.behavior_to_nwb)
    importlib.reload(converters.analysis_to_nwb)
    if Rewarded:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_Rewarded(nwb_file=nwb_file, data=data, config=config_file)
    else:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_NonRewarded(nwb_file=nwb_file, data=data, config_file=config_file)

    print("         - No ephys data for AN sessions")
    print("         - Analysis complementary information")
    converters.analysis_to_nwb.add_analysis_container(nwb_file=nwb_file, Rewarded=Rewarded, psth_window=psth_window, psth_bin=psth_bin) # almost same for rewarded and non-rewarded sessions

    importlib.reload(converters.nwb_saving)
    nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder) # same for rewarded and non-rewarded sessions

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
    if os.path.exists(output_path):
        os.remove(output_path)



#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ MAIN _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to NWB format for AN sessions")
    parser.add_argument("mat_file", type=str, help="Path to the .mat file containing the data")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the NWB file will be saved")
    parser.add_argument("--psth_window", type=float, nargs=2, default=(-0.2, 0.5), help="PSTH and LFP window in seconds for analysis")
    parser.add_argument("--psth_bin", type=float, default=0.010, help="PSTH and LFP bin size in seconds for analysis")

    args = parser.parse_args()

    convert_data_to_nwb_an(
        mat_file=args.mat_file,
        output_folder=args.output_folder,
        psth_window=tuple(args.psth_window),
        psth_bin=args.psth_bin
    )