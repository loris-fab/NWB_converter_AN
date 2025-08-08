

import numpy as np
import h5py
from pynwb.epoch import TimeIntervals   

###############################################################
# Functions for converting intervals to NWB format for AN sessions
###############################################################

def add_intervals_container_Rewarded(nwb_file, data: dict, mat_file) -> None:
    """
    Add detailed trial information to the NWBFile for a rewarded whisker detection task.
    """

    duration = 2.0
    # --- Extract trial data ---
    trial_onsets = np.asarray(data['TrialOnsets_All']).flatten()
    stim_indices = np.asarray(data['StimIndices']).flatten().astype(int)
    stim_amps = np.asarray(data['StimAmps']).flatten()
    reaction_lat = np.asarray(data['ReactionTimes']).flatten()
    reaction_abs = trial_onsets + reaction_lat
    n_trials = len(trial_onsets)
    jaw_onsets_raw = np.asarray(data['JawOnsetsTms']).flatten()

    # --- Response classification ---
    hit = np.asarray(data['HitIndices']).flatten().astype(bool)
    miss = np.asarray(data['MissIndices']).flatten().astype(bool)
    cr = np.asarray(data['CRIndices']).flatten().astype(bool)
    fa = np.asarray(data['FAIndices']).flatten().astype(bool)

    response_data = np.full(n_trials, np.nan, dtype=float)
    lick_flag = response_data.copy()
    response_data[miss] = 0.0  # Miss
    response_data[hit] = 1.0   # Hit
    response_data[cr] = 2.0    # CR
    response_data[fa] = 3.0    # FA

    lick_flag[hit] = 1.0
    lick_flag[miss] = 0.0
    lick_flag[cr] = 0.0
    lick_flag[fa] = 0.0

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



def add_intervals_container_NonRewarded(nwb_file, data: dict, mat_file) -> None:
    """
    Add detailed trial information to the NWBFile for a rewarded whisker detection task.
    """


    # --- Extract trial data ---
    trial_onsets = np.asarray(data['TrialOnsets_All']).flatten()
    stim_amps = np.asarray(data['CoilAmps']).flatten()
    n_trials = len(trial_onsets)


    # --- Per-trial CoilOnsets ---
    CoilOnsets = np.asarray(data['CoilOnsets']).flatten()
    CoilOnsets_per_trial = []
    CoilOnsets_per_trial_tms = []
    CoilOnsets_amplitude = []
    for i, t0 in enumerate(trial_onsets):
        t1 = t0 + 2.0
        indices = np.where((CoilOnsets >= t0) & (CoilOnsets < t1))[0]
        if len(indices) > 0:
            CoilOnsets_per_trial.append("whisker_trial")
            CoilOnsets_per_trial_tms.append(CoilOnsets[indices[0]])
            CoilOnsets_amplitude.append(stim_amps[indices[0]])
        else:
            CoilOnsets_per_trial.append("no_whisker_trial")
            CoilOnsets_per_trial_tms.append(np.nan)
            CoilOnsets_amplitude.append(0)

    # --- Per-trial CoilOnsets ---
    jaw_onsets_raw = np.asarray(data['JawOnsets_Tms']).flatten()
    jaw_onsets_raw_per_trial = []
    jaw_onsets_raw_per_trial_tms = []
    for i, t0 in enumerate(trial_onsets):
        t1 = t0 + 2.0
        indices = np.where((jaw_onsets_raw >= t0) & (jaw_onsets_raw < t1))[0]
        if len(indices) > 0:
            jaw_onsets_raw_per_trial.append(1)
            jaw_onsets_raw_per_trial_tms.append(jaw_onsets_raw[indices[0]])
        else:
            jaw_onsets_raw_per_trial.append(0)
            jaw_onsets_raw_per_trial_tms.append(np.nan)

    # --- Per-trial ValveOnsets_Tms ---
    ValveOnsets_Tms = np.asarray(data['ValveOnsets_Tms']).flatten()
    ValveOnsets_per_trial = []
    ValveOnsets_per_trial_tms = []
    for i, t0 in enumerate(trial_onsets):
        t1 = t0 + 2.0
        indices = np.where((ValveOnsets_Tms >= t0) & (ValveOnsets_Tms < t1))[0]
        if len(indices) > 0:
            ValveOnsets_per_trial.append(1)
            ValveOnsets_per_trial_tms.append(ValveOnsets_Tms[indices[0]])
        else:
            ValveOnsets_per_trial.append(0)
            ValveOnsets_per_trial_tms.append(np.nan)


    # --- Define new trial columns ---
    new_columns = {
        'trial_type': 'Whisker vs auditory trial',
        'whisker_stim': '1 if whisker stimulus delivered, else 0 but not delivered necessarily at the trial start time',
        'whisker_stim_onset': 'Whisker stimulus onset times',
        'whisker_stim_amplitude': 'Amplitude of whisker stimulus',
        'reward_available': 'Whether reward could be earned (1 = yes)',
        'jaw_dlc_licks':  'Jaw movements for each trial observed with DLC',
        'reward_available': 'Whether reward could be earned (1 = yes) but not delivered necessarily at the trial start time',
        'reward_available_onset': 'Valve onset times to deliver reward',
        'jaw_dlc_licks_onset': 'Jaw movements onset times observed with DLC'
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
            stop_time=float(trial_onsets[i]) + 2.0,
            trial_type= CoilOnsets_per_trial[i],
            whisker_stim= 1 if CoilOnsets_per_trial[i] == "whisker_trial" else 0,
            whisker_stim_onset=float(CoilOnsets_per_trial_tms[i]),
            whisker_stim_amplitude=float(CoilOnsets_amplitude[i]),
            jaw_dlc_licks= jaw_onsets_raw_per_trial[i],
            reward_available=ValveOnsets_per_trial[i],
            reward_available_onset=float(ValveOnsets_per_trial_tms[i]),
            jaw_dlc_licks_onset=float(jaw_onsets_raw_per_trial_tms[i]),
        )
