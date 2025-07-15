"""_summary_
"""
import importlib
import datetime
import os
import platform
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import json
import h5py
import shutil
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.subject_to_nwb
import converters.acquisition_to_nwb
import converters.units_to_nwb
import converters.ephys_to_nwb
import converters.analysis_to_nwb


import utils.utils_gf as utils_gf
from continuous_log_analysis import analyze_continuous_log
#from converters.behavior_to_nwb import convert_behavior_data
import converters.behavior_to_nwb
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.ephys_to_nwb import convert_ephys_recording
from converters.nwb_saving import save_nwb_file
import converters.nwb_saving
import converters.general_to_nwb
from converters.subject_to_nwb import create_nwb_file_an
import converters.subject_to_nwb
from converters.suite2p_to_nwb import convert_suite2p_data
from converters.widefield_to_nwb import convert_widefield_recording
#from converters.DLC_to_nwb import convert_dlc_data
from converters.facemap_to_nwb import convert_facemap_data
from utils.behavior_converter_misc import find_training_days
from utils.server_paths import (get_nwb_folder, get_subject_analysis_folder, get_experimenter_analysis_folder,
                                get_subject_data_folder, get_dlc_file_path, get_facemap_file_path, EXPERIMENTER_MAP)
from pynwb import NWBHDF5IO
from pynwb import validate

############### GLOBAL VARIABLE ########################

related_publications = 'Oryshchuk A, Sourmpis C, Weverbergh J, Asri R, Esmaeili V, Modirshanechi A, Gerstner W, Petersen CCH, Crochet S. Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during goal-directed behavior. Cell Rep. 2024 Jan 23;43(1):113618. doi: 10.1016/j.celrep.2023.113618. Epub 2023 Dec 26. PMID: 38150365.'
#related_publications = 'doi: 10.1016/j.celrep.2023.113618'
csv_file = "data/Subject_Session_Selection.csv"



############################################################
# Function that creates the config file for the NWB conversion
############################################################


def files_to_config(mat_file, output_folder="data"):
    """
    Converts a .mat file and csv_file into a .yaml configuration file for the NWB pipeline.

    :param mat_file: Path to the .mat file
    :return: Configuration dictionary + path to the yaml file
    """
    data = mat_file
    mouse = ''.join(chr(c) for c in data['mouse'].flatten())
    date = ''.join(chr(c) for c in data['date'].flatten())
    session_name = f"{mouse}_{date}"  # e.g., "AO039_20190626"

    # Load the CSV file 
    csv_data = pd.read_csv(csv_file, sep=";")
    csv_data.columns = csv_data.columns.str.strip() 

    try:
        subject_info = csv_data[csv_data['Session'].astype(str).str.strip() == session_name].iloc[0]
    except IndexError:
        raise ValueError(f"Session {session_name} not found in the CSV file.")

    ###  Session metadata extraction  ###

    ### Experiment_description
    date = ''.join(chr(c) for c in data['date'].flatten())
    date_experience = pd.to_datetime(date, format='%Y%m%d')


    ref_weight = subject_info.get("Weight of Reference", "")
    if pd.isna(ref_weight) or str(ref_weight).strip().lower() in ["", "nan"]:
        ref_weight = "Unknown"
    else:
        try:
            ref_weight = float(ref_weight)
        except Exception:
            ref_weight = "Unknown"  

    video_sr = int(data["Video_sr"])
    if pd.isna(video_sr) or str(video_sr).strip().lower() in ["", "nan"]:
        video_sr = 200
    else:
        video_sr = int(data["Video_sr"])

    # Check if all traces have the same number of frames and compute camera start delay and exposure time
    Frames_per_Video = data["JawTrace"].shape[1]
    if data["JawTrace"].shape[1] == Frames_per_Video and data["NoseSideTrace"].shape[1] == Frames_per_Video and data["NoseTopTrace"].shape[1] == Frames_per_Video and data["WhiskerAngle"].shape[1] == Frames_per_Video and data["TongueTrace"].shape[1] == Frames_per_Video :
        pass
    else:
        error_message = "Inconsistent number of frames across traces."
        raise ValueError(error_message)

    if  np.array_equal(data["VideoOnsets"], data["TrialOnsets_All"]):
        camera_start_delay = 0.0
    elif np.all(data["VideoOnsets"] < data["TrialOnsets_All"]):
        camera_start_delay = float(np.mean(data["TrialOnsets_All"] - data["VideoOnsets"]))
    else:
        error_message = "Problem with VideoOnsets and TrialOnsets_All timing."
        camera_start_delay = "Unknown"
        raise ValueError(error_message)

    video_duration = Frames_per_Video / video_sr

    camera_exposure_time = 3
    experiment_description = {
    'reference_weight': ref_weight,
    #'wh_reward': 1,
    #'aud_reward': 1,
    #'reward_proba': 1,
    #'lick_threshold': 0.08,
    #'no_stim_weight': 8,
    #'wh_stim_weight': 10,
    #'aud_stim_weight': 2,
    'camera_flag': 1,
    'camera_freq': video_sr,
    #'camera_exposure_time': camera_exposure_time,
    'each_video_duration': video_duration,
    'camera_start_delay': camera_start_delay,
    #'artifact_window': 100,
    'licence': str(subject_info.get("licence", "")).strip(),
    'ear tag': str(subject_info.get("Ear tag", "")).strip(),
}
    ### Experimenter
    experimenter = EXPERIMENTER_MAP.get(mouse[:2], 'Inconnu')
   
    ### Session_id, identifier, institution, keywords
    session_id = subject_info["Session"].strip() 
    identifier = session_id + "_" + str(subject_info["Start Time (hhmmss)"])
    keywords = ["neurophysiology", "behaviour", "mouse", "electrophysiology"] #DEMANDER SI BESOIN DE CA

    ### Session start time
    session_start_time = str(subject_info["Session Date (yyymmdd)"])+" " + str(subject_info["Start Time (hhmmss)"])

    ###  Subject metadata extraction  ###

    ### Birth date and age calculation
    birth_date = pd.to_datetime(subject_info["Birth date"], dayfirst=True)
    age = subject_info["Mouse Age (d)"]
    age = f"P{age}D"


    ### Genotype 
    genotype = subject_info.get("mutations", "")
    if pd.isna(genotype) or str(genotype).strip().lower() in ["", "nan"]:
        genotype = "WT"
    genotype = str(genotype).strip()


    ### weight
    weight = subject_info.get("Weight Session", "")
    if pd.isna(weight) or str(weight).strip().lower() in ["", "nan"]:
        weight = "Unknown"
    else:
        try:
            weight = float(weight)
        except Exception:
            weight = "Unknown" 

    ### Behavioral metadata extraction 
    camera_flag = 1

    # Construct the output YAML path
    config = {
        'session_metadata': {
            'experiment_description' : experiment_description,
            'experimenter': experimenter,
            'identifier': identifier,
            'institution': "Ecole Polytechnique Federale de Lausanne",
            'keywords': keywords,
            'lab' : "Laboratory of Sensory Processing",
            'notes': 'na',
            'pharmacology': 'na',
            'protocol': 'na',
            'related_publications': related_publications,
            'session_description': "ephys" +" " + str(subject_info.get("Session Type", "Unknown").strip()) + ":" + " Acute extracellular recordings using NeuroNexus single-shank 32-channel probes. Bandpass filtered (0.3 Hz – 7.5 kHz), amplified and digitized at 30 kHz (CerePlex M32, Blackrock). Data recorded via CerePlex Direct system. DiI coating used for post hoc localization. Initial 5–10 strong-whisker stimulation trials excluded from analysis.",
            'session_id': session_id,
            'session_start_time': session_start_time,
            'slices': "na", 
            'source_script': 'na',
            'source_script_file_name': 'na',
            'stimulus_notes': 'Whisker stimulation was applied unilaterally to the C2 region to evoke sensory responses.',
            'surgery': 'na',
            'virus': 'na',

        },
        'subject_metadata': {
            'age': age,
            'age__reference': 'birth',
            'date_of_birth': birth_date.strftime('%m/%d/%Y') if birth_date else None,
            'description': mouse,
            'genotype': genotype,
            'sex': subject_info.get("Sex_bin", "").upper().strip(),
            'species': "Mus musculus",
            'strain': subject_info.get("strain", "").strip(),
            'subject_id': mouse,
            'weight': weight,

        },
    }

    # save config
    output_path = os.path.join(output_folder, f"{session_name}_config.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    if "Non" in subject_info.get("Session Type", "Unknown").strip():
        Rewarded = False
    else:
        Rewarded = True

    return output_path, config, Rewarded



############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################


def convert_data_to_nwb_an(mat_file, output_folder, with_time_string=True, output_folder_config="data", psth_window=(-0.2, 0.5), psth_bin=0.010 ):
    """
    :param config_file: Path to the yaml config file containing mouse ID and metadata for the session to convert
    :param output_folder: Path to the folder to save NWB files
    :param with_time_string: If True, append the current time to the output file name
    :return: None 
    Converts data from a config file to an NWB file.
    """
    
    # Load the .mat file 
    with h5py.File(mat_file, 'r') as f:
        data_group = f['Data'] if 'Data' in f else f
        data = {key: data_group[key][()] for key in data_group.keys()}

    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print(" ")
    print(f"📃 Creating config file for NWB conversion :")
    output_path, config_file, Rewarded = files_to_config(data, output_folder=output_folder_config)
    print("   -", output_path)
    print(" ")
    print("📑 Created NWB file")
    print("     o 📌 Add general metadata")
    print("         - Subject metadata")
    print("         - Session metadata")
    importlib.reload(converters.subject_to_nwb)
    nwb_file = converters.subject_to_nwb.create_nwb_file_an(config_file=output_path)
    print("         - Device metadata")
    print("         - Extracellular electrophysiology metadata")
    importlib.reload(converters.general_to_nwb)
    if Rewarded:
        electrode_table_region = converters.general_to_nwb.add_general_container_Rewarded(nwb_file=nwb_file, data=data, mat_file=mat_file)
        pass
    print("     o 📶 Add acquisition container")
    importlib.reload(converters.acquisition_to_nwb)
    if Rewarded:
        converters.acquisition_to_nwb.add_lfp_acquisition(nwb_file=nwb_file, signal_array=converters.acquisition_to_nwb.extract_lfp_signal(data, mat_file), electrode_region=electrode_table_region)
        pass
    print("     o 🧠 Add units container")
    importlib.reload(converters.units_to_nwb)
    if Rewarded:
        sampling_rate =  30000
        converters.units_to_nwb.add_units_container_Rewarded(nwb_file=nwb_file, data=data, electrode_table_region=electrode_table_region, mat_file=mat_file , sampling_rate = sampling_rate)
        pass
    print("     o ⚙️ Add processing container")
    importlib.reload(converters.behavior_to_nwb)
    importlib.reload(converters.analysis_to_nwb)
    #convert_behavior_data(nwb_file=nwb_file, timestamps_dict=timestamps_dict, config_file=config_file)
    if Rewarded:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_Rewarded(nwb_file=nwb_file, data=data, config=config_file)
        print("         - No ephys data for AN sessions")
        print("         - Analysis complementary information")
        converters.analysis_to_nwb.add_analysis_container_Rewarded(nwb_file=nwb_file,psth_window=psth_window,psth_bin=psth_bin)
        pass

    
    """
    if config_dict.get("two_photon_metadata") is not None:
        print(" ")
        print("Convert CI movie")
        convert_ci_movie(nwb_file=nwb_file, config_file=config_file, movie_format='link',
                         add_movie_data_or_link=True, ci_frame_timestamps=timestamps_dict['galvo_position'])

        print(" ")
        print("Convert Suite2p data")
        convert_suite2p_data(nwb_file=nwb_file,
                             config_file=config_file,
                             ci_frame_timestamps=timestamps_dict['galvo_position'])

    if config_dict.get("ephys_metadata") is not None:
        if config_dict.get("ephys_metadata").get("processed") == 1:
             print(" ")
             print("Convert extracellular electrophysiology data")
             convert_ephys_recording(nwb_file=nwb_file,
                                     config_file=config_file)

    # Check we are on WF computer
    platform_info = platform.uname()
    computer_name = platform_info.node
    wf_computers = ['SV-07-082', 'SV-07-097']  # Add name of WF preprocessing computers here
    if computer_name in wf_computers and config_dict.get("widefield_metadata") is not None:
        print(" ")
        print("Convert widefield data")
        convert_widefield_recording(nwb_file=nwb_file,
                                    config_file=config_file,
                                    wf_frame_timestamps=timestamps_dict["widefield"])

    if config_dict.get('behaviour_metadata')['camera_flag'] == 1:
        dlc_file = get_dlc_file_path(config_file)
        if dlc_file is not None:
            print(" ")
            print("Convert DeepLabCut data")
            convert_dlc_data(nwb_file=nwb_file,
                             config_file=config_file,
                             video_timestamps={k: timestamps_dict[k] for k in ("cam1", "cam2")})

        facemap_file = get_facemap_file_path(config_file)
        if facemap_file is not None:
            print(" ")
            print("Convert Facemap data")
            convert_facemap_data(nwb_file=nwb_file,
                                 config_file=config_file,
                                 video_timestamps={k: timestamps_dict[k] for k in ("cam1", "cam2")})

    """


    # Validate the NWB & saving
    importlib.reload(converters.nwb_saving)
    nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder, with_time_string=with_time_string)

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

    return nwb_path
