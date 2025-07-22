import numpy as np
from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries
from pynwb import TimeSeries


################################################################
# Functions for adding behavior container to NWB file
################################################################

def add_behavior_container_Rewarded(nwb_file, data: dict,config: dict):
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
        description='Timestamps marking the onset of each trial.',
        comments='Encoded as 1 at each trial onset timestamp & the trial duration is 1 seconds.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_trial)


    # --- STIMULATION FLAGS ---    
    stim_amps = np.asarray(data['StimAmps']).flatten()  # Amplitude of stimulation for each trial
    ts_stim_flags = TimeSeries(
        name='StimFlags',
        data=stim_amps,
        timestamps=trial_onsets,
        unit='code',
        description='Timestamps marking the amplitude of whisker stimulation for each trial',
        comments='Whisker stimulation amplitudes are encoded as integers: 0 = no stimulus (Catch trial), 1 = 1.0°, 2 = 1.8°, 3 = 2.5°, 4 = 3.3° deflection of the C2 whisker.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_stim_flags)
    
    
    # --- REACTION TIMES ---
    reaction_times = np.asarray(data['ReactionTimes']).flatten()
    reaction_timestamps = trial_onsets + reaction_times
    binary_vector = (reaction_times > 0).astype(int)

    ts_reaction = TimeSeries(
        name='ReactionTimes',
        data=binary_vector,
        timestamps=reaction_timestamps,
        unit='n.a.',
        description = "Timestamps of reaction events defined as a lick occurring after trial onset.",
        comments = "Encoded as 1 at time of reaction, 0 if no reaction occurred with the corresponding trial timestamp.",
    )
    behavior_events.add_timeseries(ts_reaction)

    # --- ENGAGEMENT FLAGS ---
    engaged_trials = np.asarray(data['EngagedTrials']).flatten()

    ts_engagement = TimeSeries(
        name='EngagementEvents',
        data=engaged_trials,
        timestamps=reaction_timestamps,
        unit='n.a.',
        description = "Engagement events indicated when the mouse was behaviorally engaged during a reaction event.",
        comments = "Encoded as 1 at each engagement event timestamp. If no engagement occurred, the value is 0 at the corresponding trial timestamp.",
    )
    behavior_events.add_timeseries(ts_engagement)

    each_video_duration = config["session_metadata"]['experiment_description']['each_video_duration']
    # --- VIDEO ONSETS ---
    video_onsets = np.asarray(data['VideoOnsets']).flatten()
    ts_video = TimeSeries(
        name='VideoOnsets',
        data=np.ones_like(video_onsets),
        unit='n.a.',
        timestamps=video_onsets,
        description='Timestamps marking the onset of each video recording.',
        comments=f'Encoded as 1 at each video onset timestamp & the video duration is {each_video_duration} seconds.',
    )
    behavior_events.add_timeseries(ts_video)

    # ---- "JawOnsetsTms" ------
    trial_onsets = np.asarray(data['TrialOnsets_All']).flatten()
    jaw_onsets_raw = np.asarray(data['JawOnsetsTms']).flatten()
     
    mask_valid = ~np.isnan(jaw_onsets_raw) & (jaw_onsets_raw != 0)
    jaw_onsets_filled = trial_onsets.copy()
    jaw_onsets_filled[mask_valid] = jaw_onsets_raw[mask_valid]

    jaw_series = TimeSeries(
        name='jaw_dlc_licks',
        data=(jaw_onsets_raw > 0).astype(int), 
        unit='n.a.',
        timestamps=jaw_onsets_filled,
        description='Timestamps marking the onset of jaw movements for each trial observed with DLC.',
        comments='Encoded as 1 at each jaw onset timestamp. If no jaw movement occurred, the value is 0 at the corresponding trial timestamp.',
        rate=None,
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

    labels = ['MISS', 'HIT', 'CR', 'FA', 'Unlabeled']
    label_to_int = {label: i for i, label in enumerate(labels)}

    response_data = np.array([label_to_int[label] for label in response_labels])
    
    response_labels_ts = TimeSeries(
        name='ResponseType',
        data=response_data,
        unit='code',
        timestamps=reaction_timestamps,
        description = "Response type for each trial",
        comments='Integer-encoded trial responses: 0 = MISS, 1 = HIT, 2 = CR (Correct Rejection), 3 = FA (False Alarm), 4 = Unlabeled (no assigned response).',

    )

    behavior_events.add_timeseries(response_labels_ts)

    def add_event(name, mask):
            ts = TimeSeries(
                name=name,
                data=mask.astype(int),
                unit='n.a.',
                timestamps=reaction_timestamps,
                description=f"Timestamps for {name}",
                comments=f"Encoded as 1 at each {name} event timestamp.",
            )
            behavior_events.add_timeseries(ts)

    add_event('auditory_hit_trial', hit)
    add_event('auditory_miss_trial', miss)
    add_event('correct_rejection_trial', cr)
    add_event('false_alarm_trial', fa)

    #########################################################
    ### Add continuous traces (e.g., JawTrace, NoseTrace) ###
    #########################################################
    
    behavior_ts = BehavioralTimeSeries(name='BehavioralTimeSeries')
    bhv_module.add_data_interface(behavior_ts)
    # --- JawTrace, TongueTrace, NoseTopTrace, NoseSideTrace, WhiskerAngle ---
    video_onsets = data["VideoOnsets"]
    video_sr = float(data["Video_sr"].flatten()[0])
    
    def add_behavioral_traces_to_nwb(data, video_onsets, video_sr, behavior_ts):
        """
        Add continuous behavioral traces to an NWB BehavioralTimeSeries object.

        Args:
            data (dict): Dictionary containing the traces (e.g., from .mat file)
            video_onsets (ndarray): Start times of each video trial
            video_sr (float): Video sampling rate in Hz
            behavior_ts (BehavioralTimeSeries): NWB container to receive TimeSeries
        """
        def flatten_trace_with_timestamps(trace, video_onsets, video_sr):
            """
            Flatten a (n_trials, n_frames) trace and generate aligned timestamps.

            Args:
                trace (ndarray): Trace array with shape (n_trials, n_frames)
                video_onsets (ndarray): Start times of each video trial (shape: n_trials,)
                video_sr (float): Video sampling rate in Hz

            Returns:
                vecteur_trace (ndarray): Flattened trace
                vecteur_timestamps (ndarray): Aligned timestamps
            """
            trace = np.asarray(trace)
            video_onsets = np.asarray(video_onsets).flatten()
            n_trials, n_frames = trace.shape
            dt = 1 / video_sr

            # Build aligned timestamps for each frame within trials
            vecteur_timestamps = np.zeros(n_trials * n_frames)
            for i, onset in enumerate(video_onsets):
                start = i * n_frames
                stop = start + n_frames
                vecteur_timestamps[start:stop] = onset + np.arange(n_frames) * dt

            vecteur_trace = trace.flatten()
            return vecteur_trace, vecteur_timestamps

        # List of trace keys to add
        trace_keys = ["JawTrace", "TongueTrace", "NoseTopTrace", "NoseSideTrace", "WhiskerAngle"]

        for key in trace_keys:
            if key in data:
                trace = data[key]
                if key == "JawTrace" or key == "TongueTrace" or key == "NoseTopTrace" or key == "NoseSideTrace":
                    trace = trace/ 1000  
                values, times = flatten_trace_with_timestamps(trace, video_onsets, video_sr)

                if key == "WhiskerAngle":
                    description = "Whisker angle trace across aligned video_onsets."
                    comments = "The whisker angle is defined as the angle between the whisker shaft and the midline of the brain (at rest), which separates the two cerebral hemispheres."
                elif key == "JawTrace":
                    description = "Jaw trace across aligned video_onsets."
                    comments = "The jaw trace is defined as the vertical position of the jaw relative to the rest position."
                elif key == "TongueTrace":
                    description = "Tongue trace across aligned video_onsets."
                    comments = "The tongue trace is defined as the vertical position of the tongue relative to the rest position. There are some nan because the tongue is not always visible."
                elif key == "NoseTopTrace":
                    description = "Nose top trace across aligned video_onsets."
                    comments = "The nose top trace is defined as the vertical position of the nose top relative to the rest position."
                elif key == "NoseSideTrace":
                    description = "Nose side trace across aligned video_onsets."
                    comments = "The nose side trace is defined as the horizontal position of the nose side relative to the rest position."
                

                ts = TimeSeries(
                    name=key,
                    data=values,
                    unit='a.u.',
                    timestamps=times,
                    description=description,
                    comments=comments,
                )
                behavior_ts.add_timeseries(ts)

    add_behavioral_traces_to_nwb(data, video_onsets, video_sr, behavior_ts)

    #---- LickData ------
    lick_data = np.asarray(data["LickData"]).flatten()
    lick_time = np.asarray(data["LickTime"]).flatten()

    lick_ts = TimeSeries(
        name="LickTrace",
        data=lick_data,
        unit='a.u.',
        timestamps=lick_time,
        description="Lick signal over time ",
        comments="Lick data is a binary signal where over 0 indicates a lick event.",
    )
    behavior_ts.add_timeseries(lick_ts)



    return None


def add_behavior_container_NonRewarded(nwb_file, data: dict, config_file: dict):
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
        description='Timestamps marking the onset of each trial.',
        comments='Encoded as 1 at each trial onset timestamp & the trial duration is 2 seconds.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_trial)

    # --- Valve ONSETS ---
    ValveOnsets_Tms = np.asarray(data['ValveOnsets_Tms']).flatten()
    ts_valve = TimeSeries(
        name='ValveOnsets',
        data=np.ones_like(ValveOnsets_Tms),
        unit='n.a.',
        timestamps=ValveOnsets_Tms,
        description='Timestamps marking the onset of each valve activation.',
        comments='Encoded as 1 at each valve activation timestamp',
        rate = None,
    )
    behavior_events.add_timeseries(ts_valve)

    # --- Valve Associations ---
    Valve_Ind_Assosiation = np.asarray(data['Valve_Ind_Assosiation']).flatten()
    ts_valve = TimeSeries(
        name='Valve_Ind_Assosiation',
        data=Valve_Ind_Assosiation,
        unit='n.a.',
        timestamps=ValveOnsets_Tms,
        description='Timestamps marking if the valve was activated manually or automatically.',
        comments='Encoded as 1 if the valve was activated manually, 0 if it was activated automatically.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_valve)

    # --- MOUSE TRIGGERED ---
    Valve_Ind_MouseTriggered = np.asarray(data['Valve_Ind_MouseTriggered']).flatten()
    ts_mouse_triggered = TimeSeries(
        name='Valve_Ind_MouseTriggered',
        data=Valve_Ind_MouseTriggered,
        unit='n.a.',
        timestamps=ValveOnsets_Tms,
        description='Timestamps marking if the valve was activated by the mouse.',
        comments='Encoded as 1 if the valve was activated by the mouse, 0 if it was not.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_mouse_triggered)

    # --- STIMULATION FLAGS ---    
    stim_amps = np.asarray(data['StimAmps']).flatten()  # Amplitude of stimulation for each trial
    stim_timestamps = np.asarray(data['CoilOnsets']).flatten()
    ts_stim_flags = TimeSeries(
        name='StimFlags',
        data=stim_amps,
        timestamps=stim_timestamps,
        unit='code',
        description='Timestamps marking the amplitude of whisker stimulation',
        comments='Whisker stimulation amplitudes are encoded as integers: 0 = no stimulus (Catch trial), 1 = 1.0°, 2 = 1.8°, 3 = 2.5°, 4 = 3.3° deflection of the C2 whisker.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_stim_flags)
    
    # ---- "JawOnsetsTms" ------
    jaw_onsets_raw = np.asarray(data['JawOnsets_Tms']).flatten()

    jaw_series = TimeSeries(
        name='jaw_dlc_licks',
        data=np.ones_like(jaw_onsets_raw), 
        unit='n.a.',
        timestamps=jaw_onsets_raw,
        description='Timestamps marking the onset of jaw movements observed with DLC.',
        comments='Encoded as 1 at each jaw onset timestamp.',
        rate=None,
    )
    behavior_events.add_timeseries(jaw_series)

    # ---- "PiezoLickOnsets" ------
    PiezoLickOnset_Tms_CompleteLicks = np.asarray(data['PiezoLickOnset_Tms_CompleteLicks']).flatten()
    piezo_lick_series = TimeSeries(
        name='PiezoLickOnsets',
        data=np.ones_like(PiezoLickOnset_Tms_CompleteLicks), 
        unit='n.a.',
        timestamps=PiezoLickOnset_Tms_CompleteLicks,
        description='Timestamps marking the onset of piezoelectric sensor detected licks.',
        comments='Encoded as 1 at each piezoelectric lick onset timestamp.',
        rate=None,
    )
    behavior_events.add_timeseries(piezo_lick_series)

    #########################################################
    ### Add continuous traces (e.g., JawTrace, NoseTrace) ###
    #########################################################
    
    behavior_ts = BehavioralTimeSeries(name='BehavioralTimeSeries')
    bhv_module.add_data_interface(behavior_ts)
    # --- JawTrace, TongueTrace, NoseTopTrace, NoseSideTrace, WhiskerAngle ---
    VideoFrames_Tms = np.asarray(data["VideoFrames_Tms"]).flatten()


    def add_behavioral_traces_to_nwb(data, VideoFrames_Tms, behavior_ts):
        """
        Add continuous behavioral traces to an NWB BehavioralTimeSeries object.

        Args:
            data (dict): Dictionary containing the traces (e.g., from .mat file)
            VideoFrames_Tms (ndarray): times of each frames
            video_sr (float): Video sampling rate in Hz
            behavior_ts (BehavioralTimeSeries): NWB container to receive TimeSeries
        """
        # List of trace keys to add
        trace_keys = ["JawTrace", "TongueTrace", "NoseTopTrace", "NoseSideTrace", "WhiskerAngle"]

        for key in trace_keys:
            if key in data:
                trace = data[key]
                if key == "JawTrace" or key == "TongueTrace" or key == "NoseTopTrace" or key == "NoseSideTrace":
                    trace = trace/ 1000  
                values, times = np.asarray(trace)[0], VideoFrames_Tms
                if len(values) != len(times):
                    raise ValueError(f"Length mismatch: {key} has {len(values)} values but VideoFrames_Tms has {len(times)} timestamps.")
                if key == "WhiskerAngle":
                    description = "Whisker angle trace for each video frame."
                    comments = "The whisker angle is defined as the angle between the whisker shaft and the midline of the brain (at rest), which separates the two cerebral hemispheres."
                elif key == "JawTrace":
                    description = "Jaw trace for each video frame."
                    comments = "The jaw trace is defined as the vertical position of the jaw relative to the rest position."
                elif key == "TongueTrace":
                    description = "Tongue trace for each video frame."
                    comments = "The tongue trace is defined as the vertical position of the tongue relative to the rest position. There are some nan because the tongue is not always visible."
                elif key == "NoseTopTrace":
                    description = "Nose top trace for each video frame."
                    comments = "The nose top trace is defined as the vertical position of the nose top relative to the rest position."
                elif key == "NoseSideTrace":
                    description = "Nose side trace for each video frame."
                    comments = "The nose side trace is defined as the horizontal position of the nose side relative to the rest position."
                

                ts = TimeSeries(
                    name=key,
                    data=values,
                    unit='a.u.',
                    timestamps=times,
                    description=description,
                    comments=comments,
                )
                behavior_ts.add_timeseries(ts)

    add_behavioral_traces_to_nwb(data, VideoFrames_Tms, behavior_ts)


    return None