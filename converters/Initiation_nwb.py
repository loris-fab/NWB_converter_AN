from datetime import datetime
import pandas as pd
import numpy as np
import os
import yaml
from dateutil.tz import tzlocal
from pynwb import NWBFile
from pynwb.file import Subject

#############################################################################
# Function that creates the nwb file object using all metadata
#############################################################################

def create_nwb_file_an(config_file):
    """
    Create an NWB file object using all metadata containing in YAML file

    Args:
        config_file (str): Absolute path to YAML file containing the subject experimental metadata

    """

    try:
        with open(config_file, 'r', encoding='utf8') as stream:
            config = yaml.safe_load(stream)
    except:
        print("Issue while reading the config file.")
        return

    subject_data_yaml = config['subject_metadata']
    session_data_yaml = config['session_metadata']

    # Subject info
    keys_kwargs_subject = ['age', 'age__reference', 'description', 'genotype', 'sex', 'species', 'subject_id',
                           'weight', 'date_of_birth', 'strain']
    kwargs_subject = dict()
    for key in keys_kwargs_subject:
        kwargs_subject[key] = subject_data_yaml.get(key)
        if kwargs_subject[key] is not None:
            kwargs_subject[key] = str(kwargs_subject[key])
    if 'date_of_birth' in kwargs_subject:
        date_of_birth = datetime.strptime(kwargs_subject['date_of_birth'], '%m/%d/%Y')
        date_of_birth = date_of_birth.replace(tzinfo=tzlocal())
        kwargs_subject['date_of_birth'] = date_of_birth
    subject = Subject(**kwargs_subject)

    # Session info
    keys_kwargs_nwb_file = ['session_description', 'identifier', 'session_id', 'session_start_time',
                            'experimenter', 'experiment_description', 'institution', 'keywords',
                            'notes', 'pharmacology', 'protocol', 'related_publications',
                            'source_script', 'source_script_file_name',  'surgery', 'virus',
                            'stimulus_notes', 'slices', 'lab']

    kwargs_nwb_file = dict()
    for key in keys_kwargs_nwb_file:
        kwargs_nwb_file[key] = session_data_yaml.get(key)
        if kwargs_nwb_file[key] is not None:
            if not isinstance(kwargs_nwb_file[key], list):
                kwargs_nwb_file[key] = str(kwargs_nwb_file[key])
    if 'session_description' not in kwargs_nwb_file:
        print('session_description is needed in the config file.')
        return
    if 'identifier' not in kwargs_nwb_file:
        print('identifier is needed in the config file.')
        return
    if 'session_start_time' not in kwargs_nwb_file:
        print('session_start_time is needed in the config file.')
        return
    else:

        if kwargs_nwb_file['session_start_time'][-2:] == '60': # handle non leap seconds
            kwargs_nwb_file['session_start_time'] = kwargs_nwb_file['session_start_time'][0:-2] + '59'
        session_start_time = datetime.strptime(kwargs_nwb_file['session_start_time'], '%Y%m%d %H%M%S')
        session_start_time = session_start_time.replace(tzinfo=tzlocal())
        kwargs_nwb_file['session_start_time'] = session_start_time
    if 'session_id' not in kwargs_nwb_file:
        kwargs_nwb_file['session_id'] = kwargs_nwb_file['identifier']


    # Create NWB file object
    kwargs_nwb_file['subject'] = subject
    kwargs_nwb_file['file_create_date'] = datetime.now(tzlocal())

    nwb_file = NWBFile(**kwargs_nwb_file)
    if nwb_file is None:
        raise ValueError("❌ NWB file creation failed. Please check the provided metadata in the config file.")

    return nwb_file


#############################################################################
# Function that creates the config file (rewarded) for the NWB conversion
#############################################################################


def files_to_config_Rewarded(mat_file, csv_file,output_folder="data"):
    """
    Converts a .mat file and csv_file into a .yaml configuration file for the NWB pipeline.
    :param mat_file: dictionary containing the data from the .mat file
    :param output_folder: Path to the folder to save the config file
    :return: Configuration dictionary + path to the yaml file
    """
    related_publications = 'Oryshchuk A, Sourmpis C, Weverbergh J, Asri R, Esmaeili V, Modirshanechi A, Gerstner W, Petersen CCH, Crochet S. Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during goal-directed behavior. Cell Rep. 2024 Jan 23;43(1):113618. doi: 10.1016/j.celrep.2023.113618. Epub 2023 Dec 26. PMID: 38150365.'
    
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

    experiment_description = {
    'reference_weight': ref_weight,
    #'wh_reward': ?,
    #'aud_reward': ?,
    #'reward_proba': ?,
    #'lick_threshold': ?,
    #'no_stim_weight': ?,
    #'wh_stim_weight': ?,
    #'aud_stim_weight': ?,
    #'camera_flag': ?,
    'behavior type': 'Psychometric Whisker',
    'camera_freq': video_sr,
    #'camera_exposure_time': camera_exposure_time,
    'each_video_duration': video_duration,
    'camera_start_delay': camera_start_delay,
    #'artifact_window': ?,
    'licence': str(subject_info.get("licence", "")).strip()+ " (All procedures were approved by the Swiss Federal Veterinary Office)",
    'ear tag': str(subject_info.get("Ear tag", "")).strip(),
    'Software and algorithms' : "MATLAB R2021a, Kilosort2, Allen CCF tools , DeepLabCut 2.2b7 ",
    'Ambient noise' : "80 dB",
}
    ### Experimenter
    experimenter = "Anastasiia Oryshchuk"
   
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
            'notes': "Combining high-density extracellular electrophysiological recordings (from wS1, tjM1, mPFC) with high-speed videography of orofacial movements of mice performing a psychometric whisker sensory detection task reported by licking, Oryshchuk et al.",
            'pharmacology': 'na',
            'protocol': 'na',
            'related_publications': related_publications,
            'session_description': "ephys" +" " + str(subject_info.get("Session Type", "Unknown").strip()) + ":" " Whisker-rewarded (WR+) mice were trained to lick within 1 s following the whisker stimulus (go trials) but not in the absence of the stimulus (no-go trials). The neuronal representation of sensory, motor, and decision information was studied in a sensory, a motor, and a higher-order cortical area in these mice trained to lick for a water reward in response to a brief whisker stimulus.",
            'session_id': session_id,
            'session_start_time': session_start_time,
            'slices': "Allen CCF tools was used to register brain slices and probe locations to the Allen mouse brain atlas.", 
            'source_script': 'na',
            'source_script_file_name': 'na',
            'stimulus_notes': 'Whisker stimulation (a brief magnetic pulse of 1-ms acting upon a small metal particle) was applied unilaterally to the C2 region to evoke sensory responses. Stimulus trials included four whisker stimulus amplitudes of 1 , 1.8 , 2.5 , and 3.3 deflection of the right C2 whisker, also delivered with equal probabilities.',
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
    return output_path, config

def Rewarded_or_not(mat_file, csv_file):
    """
    Check if the session is rewarded or not based on the CSV file.
    :param mat_file: dictionary containing the data from the .mat file
    :param csv_file: Path to the CSV file
    :return: True if rewarded, False otherwise
    """
    # Load the .mat file
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

    if "Non" in subject_info.get("Session Type", "Unknown").strip():
        Rewarded = False
    else:
        Rewarded = True

    return Rewarded

#############################################################################
# Function that creates the config file (non rewarded) for the NWB conversion
#############################################################################

def files_to_config_NonRewarded(mat_file, csv_file,output_folder="data"):
    """
    Converts a .mat file and csv_file into a .yaml configuration file for the NWB pipeline.

    :param mat_file: dictionary containing the data from the .mat file
    :param output_folder: Path to the folder to save the config file
    :return: Configuration dictionary + path to the yaml file
    """
    related_publications = 'Oryshchuk A, Sourmpis C, Weverbergh J, Asri R, Esmaeili V, Modirshanechi A, Gerstner W, Petersen CCH, Crochet S. Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during goal-directed behavior. Cell Rep. 2024 Jan 23;43(1):113618. doi: 10.1016/j.celrep.2023.113618. Epub 2023 Dec 26. PMID: 38150365.'
    
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
    Frames_tot = data["JawTrace"].shape[1]
    if data["JawTrace"].shape[1] == Frames_tot and data["NoseSideTrace"].shape[1] == Frames_tot and data["NoseTopTrace"].shape[1] == Frames_tot and data["WhiskerAngle"].shape[1] == Frames_tot and data["TongueTrace"].shape[1] == Frames_tot:
        pass
    else:
        error_message = "Inconsistent number of frames across traces."
        raise ValueError(error_message)


    video_duration_total = Frames_tot / video_sr

    experiment_description = {
    'reference_weight': ref_weight,
    #'wh_reward': ?,
    #'aud_reward': ?,
    #'reward_proba': ?,
    #'lick_threshold': ?,
    #'no_stim_weight': ?,
    #'wh_stim_weight': ?,
    #'aud_stim_weight': ?,
    #'camera_flag': ?,
    'behavior type': 'Psychometric Whisker',
    'camera_freq': video_sr,
    #'camera_exposure_time': camera_exposure_time,
    'total_video_duration': video_duration_total,
    #'artifact_window': ?,
    'licence': str(subject_info.get("licence", "")).strip() + " (All procedures were approved by the Swiss Federal Veterinary Office)",
    'ear tag': str(subject_info.get("Ear tag", "")).strip(),
    'Software and algorithms' : "MATLAB R2021a, Kilosort2, Allen CCF tools , DeepLabCut 2.2b7 ",
    'Ambient noise' : "80 dB",
}
    ### Experimenter
    experimenter = "Anastasiia Oryshchuk"
   
    ### Session_id, identifier, institution, keywords
    session_id = subject_info["Session"].strip() 
    identifier = session_id + "_" + str(subject_info["Start Time (hhmmss)"])
    keywords = ["neurophysiology", "behaviour", "mouse", "electrophysiology"]

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
            'notes': "Combining high-density extracellular electrophysiological recordings (from wS1, tjM1, mPFC) with high-speed videography of orofacial movements of mice performing a psychometric whisker sensory detection task reported by licking, Oryshchuk et al.",
            'pharmacology': 'na',
            'protocol': 'na',
            'related_publications': related_publications,
            'session_description': "ephys" +" " + str(subject_info.get("Session Type", "Unknown").strip()) + ":" + " Whisker non-rewarded (WR ) mice, the whisker stimulus was decorrelated to reward delivery. The neuronal representation of sensory, motor, and decision information was examined in mice exposed to brief whisker stimuli that did not predict reward availability, in sensory, motor, and higher-order cortical areas.",
            'session_id': session_id,
            'session_start_time': session_start_time,
            'slices': "Allen CCF tools was used to register brain slices and probe locations to the Allen mouse brain atlas.", 
            'source_script': 'na',
            'source_script_file_name': 'na',
            'stimulus_notes': 'C2 whisker stimulation (a brief magnetic pulse of 1-ms acting upon a small metal particle) occurred independently of trial timing and could happen at any moment. Stimulus trials included four whisker stimulus amplitudes of 1 , 1.8 , 2.5 , and 3.3 deflection of the right C2 whisker, also delivered with equal probabilities.',
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
    return output_path, config


