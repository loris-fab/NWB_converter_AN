import os
import re

import numpy as np
import yaml
from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEpochs, BehavioralEvents, BehavioralTimeSeries
from pynwb.image import ImageSeries
from pynwb.file import NWBFile
from pynwb import TimeSeries

from utils import continuous_processing, server_paths
from utils.behavior_converter_misc import (add_trials_standard_to_nwb,
                                           add_trials_to_nwb,
                                           build_simplified_trial_table,
                                           build_standard_trial_table,
                                           get_context_timestamps_dict,
                                           get_piezo_licks_timestamps_dict,
                                           get_trial_timestamps_dict,
                                           get_motivated_epoch_ts)



def convert_behavior_data(nwb_file, timestamps_dict, config_file):
    """
    Convert behavior data to NWB format and add to NWB file.
    Args:
        nwb_file:
        timestamps_dict:
        config_file:

    Returns:

    """

    # Get session behaviour results file
    behavior_results_file = server_paths.get_behavior_results_file(config_file)

    # Make trial table
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    if 'behaviour_metadata' in config_dict:
        if config_dict.get('behaviour_metadata').get('trial_table') == 'standard':
            trial_table = build_standard_trial_table(
                config_file=config_file,
                behavior_results_file=behavior_results_file,
                timestamps_dict=timestamps_dict
            )

            ## Fix for mice with no block info in sessions until 06/08/2024
            subject_id = config_dict.get('subject_metadata').get('subject_id')
            if subject_id in ['PB185', 'PB186', 'PB187', 'PB188', 'PB189', 'PB190']:
                if int(config_dict.get('session_metadata').get('session_id').split('_')[1]) <= 20240806:
                    import pandas as pd
                    context_rewarded = pd.read_csv(r"M:\analysis\Pol_Bech\behaviour_context_files\rewarded_context.csv")
                    trial_table['context_background'] = trial_table['context'].map(
                        {1: context_rewarded.loc[context_rewarded['MouseName'] == subject_id, 'RewardedContext'].item(),
                         0: 'pink' if context_rewarded.loc[context_rewarded[
                                                               'MouseName'] == subject_id, 'RewardedContext'].item() == 'brown' else 'brown'})

        elif config_dict.get('behaviour_metadata').get('trial_table') == 'simple':
            trial_table = build_simplified_trial_table(behavior_results_file=behavior_results_file,
                                                       timestamps_dict=timestamps_dict)
    else:
        print('NWB config file lacks a behaviour_metadata section. Fix config file first.')

    print("Adding trials to NWB file")
    if config_dict.get('behaviour_metadata').get('trial_table') == 'standard':
        add_trials_standard_to_nwb(nwb_file=nwb_file, trial_table=trial_table)
    else:
        add_trials_to_nwb(nwb_file=nwb_file, trial_table=trial_table)

    # Create NWB behaviour module (and module interfaces)

    if timestamps_dict is None:
        return

    print("Creating behaviour processing module")
    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    try:
        behavior_events = bhv_module.get(name='BehavioralEvents')
    except KeyError:
        behavior_events = BehavioralEvents(name='BehavioralEvents')
        bhv_module.add_data_interface(behavior_events)

    # Get trial timestamps and indexes
    trial_timestamps_dict, trial_indexes_dict = get_trial_timestamps_dict(timestamps_dict,
                                                                          behavior_results_file, config_file)

    # Add a time series of trial timestamps, for each trial type
    trial_types = list(trial_timestamps_dict.keys())
    for trial_type in trial_types:
        data_to_store = np.transpose(np.array(trial_indexes_dict.get(trial_type)))
        timestamps_on_off = trial_timestamps_dict.get(trial_type)
        timestamps_to_store = timestamps_on_off[0]

        trial_timeseries = TimeSeries(name=f'{trial_type}_trial',
                                      data=data_to_store,
                                      unit='seconds',
                                      resolution=-1.0,
                                      conversion=1.0,
                                      offset=0.0,
                                      timestamps=timestamps_to_store,
                                      starting_time=None,
                                      rate=None,
                                      comments='no comments',
                                      description=f'index (data) and timestamps of {trial_type} trials',
                                      control=None,
                                      control_description=None,
                                      continuity='instantaneous')

        behavior_events.add_timeseries(trial_timeseries)
        print(f"Adding {len(data_to_store)} {trial_type} to BehavioralEvents")

    # Get piezo lick timestamps
    piezo_licks_timestamps_dict = get_piezo_licks_timestamps_dict(timestamps_dict)

    if piezo_licks_timestamps_dict is not None:
        timestamps_to_store = np.array(piezo_licks_timestamps_dict)
        if timestamps_to_store.any():
            timestamps_to_store = timestamps_to_store[:, 0]

        data_to_store = np.transpose(np.array(timestamps_to_store))
        lick_timeseries = TimeSeries(name='piezo_lick_times',
                                     data=data_to_store,
                                     unit='seconds',
                                     resolution=-1.0,
                                     conversion=1.0,
                                     offset=0.0,
                                     timestamps=timestamps_to_store,
                                     starting_time=None,
                                     rate=None,
                                     comments='no comments',
                                     description='piezo lick timestamps',
                                     control=None,
                                     control_description=None,
                                     continuity='instantaneous')
        behavior_events.add_timeseries(lick_timeseries)
        print(f"Adding {len(data_to_store)} piezo lick times to BehavioralEvents")

    # Get context timestamps if they exist #TODO: potentially what follows this with passive/active
    context_timestamps_dict, context_sound_dict = get_context_timestamps_dict(timestamps_dict=timestamps_dict,
                                                                              nwb_trial_table=trial_table)
    # If context, add context timestamps to NWB file
    if context_timestamps_dict is not None:
        print("Adding context epochs to NWB file")
        try:
            behavior_epochs = bhv_module.get(name='BehavioralEpochs')
        except KeyError:
            behavior_epochs = BehavioralEpochs(name='BehavioralEpochs')
            bhv_module.add_data_interface(behavior_epochs)

        for epoch, intervals_list in context_timestamps_dict.items():
            print(f"Add {len(intervals_list)} {epoch} epochs to NWB ")
            time_stamps_to_store = []
            data_to_store = []
            description = context_sound_dict.get(epoch)
            for interval in intervals_list:
                start_time = interval[0]
                stop_time = interval[1]
                time_stamps_to_store.extend([start_time, stop_time])
                data_to_store.extend([1, -1])
            behavior_epochs.create_interval_series(name=epoch, data=data_to_store, timestamps=time_stamps_to_store,
                                                   comments='no comments',
                                                   description=description,
                                                   control=None, control_description=None)

    if config_dict.get("two_photon_metadata") is not None:
        # Get motivated/unmotivated timestamps
        motivated_timestamps_dict = get_motivated_epoch_ts(timestamps_dict=timestamps_dict, nwb_trial_table=trial_table)

        # Add motivated epochs
        if motivated_timestamps_dict is not None:
            print("Adding motivated epochs to NWB file")
            try:
                behavior_epochs = bhv_module.get(name='BehavioralEpochs')
            except KeyError:
                behavior_epochs = BehavioralEpochs(name='BehavioralEpochs')
                bhv_module.add_data_interface(behavior_epochs)

            for epoch, intervals_list in motivated_timestamps_dict.items():
                print(f"Add {len(intervals_list)} {epoch} epochs to NWB ")
                time_stamps_to_store = []
                data_to_store = []
                description = epoch + '_epoch'
                for interval in intervals_list:
                    start_time = interval[0]
                    stop_time = interval[1]
                    time_stamps_to_store.extend([start_time, stop_time])
                    data_to_store.extend([1, -1])
                behavior_epochs.create_interval_series(name=epoch, data=data_to_store, timestamps=time_stamps_to_store,
                                                       comments='no comments',
                                                       description=description,
                                                       control=None, control_description=None)

    # Check if behaviour video filming
    if config_dict.get('session_metadata').get('experimenter') == 'AB':
        if config_dict.get('behaviour_metadata').get('behaviour_type') in ['auditory', 'free_licking']:
            print('Ignoring videos for auditory sessions')
            movie_files = None
        else:
            if config_dict.get('behaviour_metadata').get('camera_flag') == 1:

                movie_files = server_paths.get_session_movie_files(config_file)
                movie_files = [f for f in movie_files if 'short' not in f]
            else:
                movie_files = None
    else:
        movie_files = server_paths.get_movie_files(config_file)

    # If there is a behaviour video, add camera frame timestamps to NWB file
    if config_dict.get('behaviour_metadata').get('camera_flag') == 0:
        print('Camera flag is set to 0, ignoring any video files data')
        movie_files = None

    if movie_files is not None:
        print("Adding behavior movies as external file to NWB file")
        for movie_index, movie in enumerate(movie_files):

            # If movie file does not exist, skip
            if not os.path.exists(movie):
                print(f"File not found, do next video")
                continue

            # Get information about video
            print("Check length and frame rate")
            video_length, video_frame_rate = continuous_processing.read_behavior_avi_movie(movie_file=movie)
            print(f"Video length: {video_length}, frame rate: {video_frame_rate}")

            #  Check number of frames in video vs. number of timestamps
            if config_dict.get('session_metadata').get('experimenter') == 'AB':
                key_view_mapper = {'top': 'cam1', 'side': 'cam2', 'lateral': 'cam2'}
                movie_file_suffix = next(
                    (part for part in os.path.basename(movie).replace('-', '_').replace(' ', '_').split('_')
                     if part in key_view_mapper), None) # replaces spaces with underscores
                cam_key = key_view_mapper.get(movie_file_suffix)
                # Replace suffix with camera key
                movie_nwb_file_name = movie.replace(movie_file_suffix, f'{movie_file_suffix}_{cam_key}')

            else:
                movie_nwb_file_name = f"{ os.path.split(movie)[1].split('.')[0]}_camera_{movie_index + 1}"
                if 'side' in movie_nwb_file_name:
                    cam_key = 'cam1'
                if 'top' in movie_nwb_file_name:
                    cam_key = 'cam2'
                if ('top' not in movie_nwb_file_name) and ('side' not in movie_nwb_file_name):
                    cam_key = 'cam1'

            # Get frame timestamps
            on_off_timestamps = timestamps_dict[cam_key]
            if np.abs(len(on_off_timestamps) - video_length) > 2:
                print(f"Difference in number of frames ({video_length}) vs detected frames ({len(on_off_timestamps)}) "
                      f"is {len(on_off_timestamps) - video_length} (larger than 2), do next video")
                continue
            elif len(on_off_timestamps) == video_length-1:
                movie_timestamps = [on_off_timestamps[i][0] for i in range(video_length-1)]

            else:
                movie_timestamps = [on_off_timestamps[i][0] for i in range(video_length)]

            behavior_external_file = ImageSeries(
                name=movie_nwb_file_name,
                description="Behavior video of animal in the task",
                unit="n.a.",
                external_file=[movie],
                format="external",
                starting_frame=[0],
                timestamps=movie_timestamps
            )

            nwb_file.add_acquisition(behavior_external_file)





def add_behavior_container(nwb_file, data: dict, config: dict):
    """
    Adds a 'behavior' container to the NWB file from the loaded .mat data.

    :param nwb_file: existing NWB file
    :param data: dictionary from the .mat file (already pre-loaded with h5py)
    :param config: YAML configuration dictionary already loaded
    :return: None
    """

    # 1. Created behavior processing module
    bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    ###############################################
    ### Add behavioral events (e.g., JawOnsets) ###
    ###############################################


    behavior_events = BehavioralEvents(name='BehavioralEvents')
    bhv_module.add_data_interface(behavior_events)


    # --- TRIAL ONSETS ---
    trial_onsets = np.asarray(data['TrialOnsets_All']).flatten()
    ts_trial = TimeSeries(
        name='TrialOnsets',
        data=np.ones_like(trial_onsets),
        unit='n.a.',
        timestamps=trial_onsets,
        description='Timestamps marking the onset of each trial (value=1 at each onset).',
        comments='One trial is 1 second long, starting at the onset timestamp.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_trial)


    # --- STIMULATION FLAGS ---
    stim_flags = np.asarray(data['StimIndices']).flatten()      # 0 or 1 for each trial
    stim_amps = np.asarray(data['StimAmps']).flatten()  # Amplitude of stimulation for each trial
    ts_stim_flags = TimeSeries(
        name='StimFlags',
        data=stim_amps,
        timestamps=trial_onsets,
        unit='uA',
        description='Amplitude of stimulation for each trial, where 0 indicates no stimulation.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_stim_flags)
    
    
    # --- REACTION TIMES ---
    reaction_times = np.asarray(data['ReactionTimes']).flatten()
    reaction_timestamps = trial_onsets[reaction_times > 0] + reaction_times[reaction_times > 0]

    ts_reaction = TimeSeries(
        name='ReactionEvents',
        data=np.ones_like(reaction_timestamps),
        timestamps=reaction_timestamps,
        unit='n.a.',
        description = "Timestamps of reaction events, marked by a value of 1 at each reaction time, defined as a lick occurring after trial onset.",
    )
    behavior_events.add_timeseries(ts_reaction)

    # --- ENGAGEMENT FLAGS ---
    engaged_trials = np.asarray(data['EngagedTrials']).flatten()
    reaction_times = np.asarray(data['ReactionTimes']).flatten()
    engaged_trials = engaged_trials[reaction_times > 0]

    ts_engagement = TimeSeries(
        name='EngagementEvents',
        data=engaged_trials,
        timestamps=reaction_timestamps,
        unit='n.a.',
        description = "Engagement events indicated by a value of 1 at timestamps when the mouse was behaviorally engaged during a reaction event.",
    )
    behavior_events.add_timeseries(ts_engagement)

    # --- VIDEO ONSETS ---
    video_onsets = np.asarray(data['VideoOnsets']).flatten()
    ts_video = TimeSeries(
        name='VideoOnsets',
        data=np.ones_like(video_onsets),
        unit='n.a.',
        timestamps=video_onsets,
        description='Timestamps marking the onset of each video recording (value=1 at each onset).',
        comments='One video is 3 second long, starting at the onset timestamp.',
    )
    behavior_events.add_timeseries(ts_video)

    # ---- "JawOnsetsTms" ------
    jaw_onsets = np.array(data['JawOnsetsTms']).flatten()
    jaw_onsets = np.asarray(jaw_onsets, dtype=float)
    jaw_onsets = jaw_onsets[~np.isnan(jaw_onsets)]

    jaw_series = TimeSeries(
        name='JawOnsets',
        data=np.ones_like(jaw_onsets), 
        unit='n.a.',
        timestamps=jaw_onsets,
        description='Timestamps marking the onset of jaw movements (value=1 at each onset).',
    )
    behavior_events.add_timeseries(jaw_series)


    # ---- "ResponseType" ------

    hit = np.asarray(data['HitIndices']).flatten().astype(bool)
    miss = np.asarray(data['MissIndices']).flatten().astype(bool)
    cr = np.asarray(data['CRIndices']).flatten().astype(bool)
    fa = np.asarray(data['FAIndices']).flatten().astype(bool)

    n_trials = len(hit)
    response_labels = np.full(n_trials, 'Unlabeled', dtype=object)  # valeur par défaut

    # Attribution avec priorité : FA < CR < MISS < HIT
    response_labels[fa] = 'FA'
    response_labels[cr] = 'CR'
    response_labels[miss] = 'MISS'
    response_labels[hit] = 'HIT'

    labels = ['HIT', 'MISS', 'CR', 'FA', 'Unlabeled']
    label_to_int = {label: i for i, label in enumerate(labels)}

    response_data = np.array([label_to_int[label] for label in response_labels])
    
    response_labels_ts = TimeSeries(
        name='ResponseType',
        data=response_data,
        unit='code',
        timestamps=trial_onsets,
        description = "Response type for each trial",
        comments='Integer-encoded trial responses: 0 = HIT, 1 = MISS, 2 = CR (Correct Rejection), 3 = FA (False Alarm), 4 = Unlabeled (no assigned response).',

    )

    behavior_events.add_timeseries(response_labels_ts)

    """
    # --- LICK TIME ---
    lick_times = np.asarray(data['LickTime']).flatten()
    ts_licks = TimeSeries(
        name='LickTimes',
        data=np.ones_like(lick_times),
        unit='n.a.',
        timestamps=lick_times,
    )
    behavior_events.add_timeseries(ts_licks)
    """

    #########################################################
    ### Add continuous traces (e.g., JawTrace, NoseTrace) ###
    #########################################################
    
    behavior_ts = BehavioralTimeSeries(name='BehavioralTimeSeries')
    bhv_module.add_data_interface(behavior_ts)

    def add_continuous_trace(name, data_key):
        if data_key in data:
            trace = np.array(data[data_key])                  # (n_trials, n_frames)
            n_trials, n_frames = trace.shape
            trial_duration = 2.0                              # All trials are 2 seconds long
            sampling_rate = n_frames / trial_duration         

            full_trace = trace.flatten()                      # concatenate all trials
            timestamps = np.arange(len(full_trace)) / sampling_rate

            ts = TimeSeries(
                name=name,
                data=full_trace,
                unit='a.u.',
                timestamps=timestamps,
                description=f'{name} concatenated across all trials',
                comments=f'Trace at {sampling_rate:.1f} Hz, {n_trials} trials concatenated'
            )

            behavior_ts.add_timeseries(ts)
    

    add_continuous_trace("JawTrace", "JawTrace")
    #add_continuous_trace("TongueTrace", "TongueTrace")
    #add_continuous_trace("NoseTopTrace", "NoseTopTrace")
    #add_continuous_trace("NoseSideTrace", "NoseSideTrace")
    #add_continuous_trace("WhiskerAngle", "WhiskerAngle")
