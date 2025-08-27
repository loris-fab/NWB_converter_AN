

import numpy as np
import h5py
from pynwb.epoch import TimeIntervals   

###############################################################
# Functions for converting intervals to NWB format for AN sessions
###############################################################

def add_intervals_container_Rewarded(nwb_file, data: dict) -> None:
    """
    Add trial information for a rewarded whisker detection task to the NWBFile.

    This function creates or updates the NWB trial table with detailed metadata 
    about each trial, including:
      - trial type (stimulus vs no stimulus),
      - whisker stimulation parameters,
      - response window times,
      - trial outcome (hit, miss, correct rejection, false alarm),
      - licking and jaw movement information.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to which trials will be added.
    data : dict
        Dictionary containing trial-level data extracted from the .mat file.

    Returns
    -------
    None
        Updates the NWB file in place.
    """

    duration = 2.0
    # --- Extract trial data ---
    trial_onsets = np.asarray(data['TrialOnsets_All']).flatten()
    stim_indices = np.asarray(data['StimIndices']).flatten().astype(int)
    stim_amps = np.asarray(data['StimAmps']).flatten()
    reaction_lat = np.asarray(data['ReactionTimes']).flatten()

        # Absolute reaction times (trial onset + latency); keep NaN when latency equals onset logic
    reaction_abs = trial_onsets + reaction_lat
    reaction_abs = np.array([float(el) if el != trial_onsets[index] else np.nan for index, el in enumerate(reaction_abs)], dtype=float)
    n_trials = len(trial_onsets)
    jaw_onsets_raw = np.asarray(data['JawOnsetsTms']).flatten()

    # --- Response classification and licking behavior ---
    hit = np.asarray(data['HitIndices']).flatten().astype(bool)
    miss = np.asarray(data['MissIndices']).flatten().astype(bool)
    cr = np.asarray(data['CRIndices']).flatten().astype(bool)
    fa = np.asarray(data['FAIndices']).flatten().astype(bool)

    response_data = np.full(n_trials, np.nan, dtype=float)
    lick_flag = response_data.copy()
        # perf codes: 0=miss, 1=hit, 2=CR, 3=FA
    response_data[miss] = 0.0  # Miss
    response_data[hit] = 1.0   # Hit
    response_data[cr] = 2.0    # CR
    response_data[fa] = 3.0    # FA

    lick_flag[hit] = 1.0
    lick_flag[miss] = 0.0
    lick_flag[cr] = 0.0
    lick_flag[fa] = 1.0

    # --- Define new trial columns ---
    new_columns = {
        'trial_type': 'Stimulus Whisker vs no stimulation trial',
        'whisker_stim': '1 if whisker stimulus delivered, else 0',
        'whisker_stim_amplitude': 'Amplitude of whisker stimulus',
        'whisker_stim_time': 'Whisker stimulus onset time',
        'whisker_stim_duration': 'Duration of whisker stimulus (ms)',
        'no_stim': '1 if no whisker stimulus delivered, else 0',
        'no_stim_time': 'No whisker stimulus onset time',
        'reward_available': 'Whether reward could be earned (1 = yes)',
        'response_window_start_time': 'Start of response window',
        'response_window_stop_time': 'Stop of response window',
        'perf': 'Trial outcome label (0= whisker miss; 1= whisker hit ; 2= correct rejection ; 3= false alarm)',
        'lick_time': 'Whitin response window lick time. Absolute time (s) relative to session start time.',
        'jaw_dlc_licks':  'Jaw movements for each trial observed with DLC',
        'lick_flag': '1 if lick occurred within response window, else 0'
    }

    # --- Add columns before inserting trials ---
    if nwb_file.trials is None:
        # This creates an empty trial table
        for col, desc in new_columns.items():
            nwb_file.add_trial_column(name=col, description=desc)

    else:
        # Add only missing columns if table already exists
        for col, desc in new_columns.items():
            if col not in nwb_file.trials.colnames:
                nwb_file.add_trial_column(name=col, description=desc)

    # --- Add trials ---
    for i in range(n_trials):
        nwb_file.add_trial(
            start_time=float(trial_onsets[i]),
            stop_time=float(trial_onsets[i]) + duration,
            trial_type='whisker_trial' if stim_indices[i] else 'no_stim_trial',
            whisker_stim=int(stim_indices[i]),
            whisker_stim_amplitude=float(stim_amps[i]),
            whisker_stim_time = float(trial_onsets[i]) if stim_indices[i] else np.nan,
            whisker_stim_duration = "1 (ms)",
            no_stim = 0 if stim_indices[i] else 1,
            no_stim_time = np.nan if stim_indices[i] else float(trial_onsets[i]),
            reward_available = 1,
            response_window_start_time=float(trial_onsets[i]) + 0.05,
            response_window_stop_time =float(trial_onsets[i]) + 1,
            perf=response_data[i],
            lick_time=reaction_abs[i],
            jaw_dlc_licks=1 if jaw_onsets_raw[i] > 0 else 0,
            lick_flag=lick_flag[i]
        )



def add_intervals_container_NonRewarded(nwb_file, data: dict) -> None:
    """
    Add trial information for a non-rewarded whisker detection task to the NWBFile.

    This function creates or updates the NWB trial table with trial metadata, 
    including:
      - whisker stimulus timing and amplitude,
      - response window times,
      - performance label (hit or miss),
      - lick presence and lick times.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to which trials will be added.
    data : dict
        Dictionary containing trial-level data extracted from the .mat file.

    Returns
    -------
    None
        Updates the NWB file in place.
    """


    # --- Extract trial data ---
    trial_onsets = np.asarray(data['Perf'][0]).flatten()
    duration = 2.0
    stim_amps = np.asarray(data['Perf'][2]).flatten()
    lick_flag = np.asarray(data['Perf'][3]).flatten()
    lick_times = np.asarray(data['Perf'][4]).flatten()
    perf = ["whisker miss" if el == 0 else "whisker hit" for el in lick_flag]
    n_trials = len(trial_onsets)

    # --- Define new trial columns ---
    new_columns = {
        'trial_type': 'whisker_trial is when the whisker stimulus is presented',
        'perf': 'Performance of the trial (whisker hit or miss)',
        'whisker_stim': '1 because the whisker stimulus is presented',
        'whisker_stim_amplitude': 'Amplitude of whisker stimulus',
        'whisker_stim_duration' : 'Duration of whisker stimulus',
        'whisker_stim_time' : 'Time of whisker stimulus presentation',
        'reward_available': '0 because no reward is available',
        "response_window_start_time": 'Start time of the response window',
        "response_window_stop_time": 'Stop time of the response window',
        'lick_flag': '1 if lick occurred within response window, else 0',
        'lick_time': 'Lick time within response window. Absolute time (s) relative to session start time.',
    }

    # --- Add columns before inserting trials ---
    if nwb_file.trials is None:
        # This creates an empty trial table
        for col, desc in new_columns.items():
            nwb_file.add_trial_column(name=col, description=desc)

    else:
        # Add only missing columns if table already exists
        for col, desc in new_columns.items():
            if col not in nwb_file.trials.colnames:
                nwb_file.add_trial_column(name=col, description=desc)

    # --- Add trials ---
    for i in range(n_trials):
        nwb_file.add_trial(
            start_time=float(trial_onsets[i]),
            stop_time=float(trial_onsets[i]) + duration,
            trial_type= "whisker_trial",
            perf= perf[i],
            whisker_stim= 1,
            whisker_stim_amplitude=float(stim_amps[i]),
            whisker_stim_duration= str("1 (ms)"),
            whisker_stim_time=float(trial_onsets[i]),
            reward_available=0,
            response_window_start_time=float(trial_onsets[i])+ 0.05,
            response_window_stop_time=float(trial_onsets[i]) + 1.0,
            lick_flag=float(lick_flag[i]),
            lick_time=float(lick_times[i]),
        )
