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
        comments='time start of each trial',
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
    reaction_timestamps2 = np.asarray([float(el) if el != trial_onsets[index] else 0 for index, el in enumerate(reaction_timestamps)], dtype=float)
    reaction_timestamps2 = reaction_timestamps2[reaction_timestamps2 != 0]  

    ts_reaction = TimeSeries(
        name='ReactionTimes',
        data=np.ones_like(reaction_timestamps2),
        timestamps=reaction_timestamps2,
        unit='n.a.',
        description = "Timestamps of response-time defined as lick-onset occurring after trial onset.",
        comments = "Reaction time from PiezoLickSignal.",
    )
    behavior_events.add_timeseries(ts_reaction)

    # --- ENGAGEMENT FLAGS ---
    engaged_trials = np.asarray(data['EngagedTrials']).flatten()

    ts_engagement = TimeSeries(
        name='EngagedTrials',
        data=engaged_trials,
        timestamps=reaction_timestamps,
        unit='n.a.',
        description = "Engagement trials indicate trials when the mouse was behaviorally engaged in the task.",
        comments = "1 for engaged, 0 for disengaged trials.",
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
        comments=f'time start of each video. Video duration is  {each_video_duration} seconds.',
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
        comments='reaction time from the jaw opening.',
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
        comments='trial responses: 0 = MISS, 1 = HIT, 2 = CR (Correct Rejection), 3 = FA (False Alarm), 4 = Unlabeled (no assigned response).',

    )

    behavior_events.add_timeseries(response_labels_ts)

    def add_event(name, mask):
            ts = TimeSeries(
                name=name,
                data=mask.astype(int),
                unit='n.a.',
                timestamps=reaction_timestamps,
                description=f"Timestamps for {name}",
                comments=f"time of each {name} event.",
            )
            behavior_events.add_timeseries(ts)

    add_event('whisker_hit_trial', hit)
    add_event('whisker_miss_trial', miss)
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
                    comments = "the whisker angle is extracted from video filming using DeepLabCut 2.2b7 and is defined as the angle between the whisker shaft and the midline of the head."
                elif key == "JawTrace":
                    description = "Jaw traces aligned to video_onsets."
                    comments = "the jaw trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the distance between the tip of the jaw and the resting (closed) position of the jaw (mm)."
                elif key == "TongueTrace":
                    description = "Tongue traces aligned to video_onsets."
                    comments = "the tongue trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the distance between the tip of the tongue and the resting (closed) position of the jaw (mm). NB: tongue trace is only defined when the tongue is visible (protruded) otherwise = NaN"
                elif key == "NoseTopTrace":
                    description = "Nose top traces aligned to video_onsets."
                    comments = "the nose top trace trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the position of the nose relative to the resting position from the top view video."
                elif key == "NoseSideTrace":
                    description = "Nose side traces aligned to video_onsets."
                    comments = "the nose side trace trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the position of the nose relative to the resting position from the side view video."
                

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
        name="PiezoLickSignal",
        data=lick_data,
        unit='a.u.',
        timestamps=lick_time,
        description="Lick signal over time (V, Sampling rate = 100 Hz)",
        comments="PiezoLickSignal is the continuous electrical signal recorded from the piezo film attached to the water spout to detect when the mouse contacts the water spout with its tongue.",
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


    # --- Reward_Window_onset ---
    reward_window_onsets = np.asarray(data['TrialOnsets_All']).flatten()
    ts_reward_window = TimeSeries(
        name='Reward_Window_onset',
        data=np.ones_like(reward_window_onsets),
        unit='n.a.',
        timestamps=reward_window_onsets,
        description='Timestamps marking the onset of the reward window.',
        comments='time start of each reward window ',
        rate=None,
    )
    behavior_events.add_timeseries(ts_reward_window)


    # --- Valve ONSETS ---
    ValveOnsets_Tms = np.asarray(data['ValveOnsets_Tms']).flatten()
    ts_valve = TimeSeries(
        name='Reward_time',
        data=np.ones_like(ValveOnsets_Tms),
        unit='n.a.',
        timestamps=ValveOnsets_Tms,
        description='Timestamps marking the delivery of the water reward.',
        comments='time of the reward delivery',
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
    stim_amps = np.asarray(data['Perf'][2]).flatten()  # Amplitude of stimulation for each trial
    trial_onsets = np.asarray(data['Perf'][0]).flatten()
    ts_stim_flags = TimeSeries(
        name='StimFlags',
        data=stim_amps,
        timestamps=trial_onsets,
        unit='code',
        description='Timestamps marking the amplitude of whisker stimulation',
        comments='Whisker stimulation amplitudes are encoded as integers: 0 = no stimulus (Catch trial), 1 = 1.0°, 2 = 1.8°, 3 = 2.5°, 4 = 3.3° deflection of the C2 whisker.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_stim_flags)
    
    # ---- "JawOnsetsTms" ------
    jaw_onsets_raw = np.asarray(data['JawOnsets_Tms_All']).flatten()

    jaw_series = TimeSeries(
        name='jaw_dlc_licks',
        data=np.ones_like(jaw_onsets_raw), 
        unit='n.a.',
        timestamps=jaw_onsets_raw,
        description='Timestamps marking the onset of all jaw movements observed with DLC.',
        comments='reaction time from the jaw opening.',
        rate=None,
    )
    behavior_events.add_timeseries(jaw_series)

    #---- "ResponseType" -----
    lick_flag = np.asarray(data['Perf'][3]).flatten()
    lick_series = TimeSeries(
        name='ResponseType',
        data=lick_flag,
        unit='n.a.',
        timestamps=trial_onsets,
        description='Response type for each trial',
        comments='trial responses: 0 = MISS ; 1=HIT',
    )
    behavior_events.add_timeseries(lick_series)

    #--- "whisker_hit_trial" ---
    timestamps_hit = [el for index , el in enumerate(trial_onsets) if lick_flag[index] == 1]
    ts_whisker_hit = TimeSeries(
        name='whisker_hit_trial',
        data=np.ones_like(timestamps_hit),
        unit='n.a.',
        timestamps=timestamps_hit,
        description='Timestamps for whisker_hit_trial',
        comments='time of each whisker_hit_trial event.',
    )
    behavior_events.add_timeseries(ts_whisker_hit)

    #--- "whisker_miss_trial" ---
    timestamps_miss = [el for index , el in enumerate(trial_onsets) if lick_flag[index] == 0]
    ts_whisker_miss = TimeSeries(
        name='whisker_miss_trial',
        data=np.ones_like(timestamps_miss),
        unit='n.a.',
        timestamps=timestamps_miss,
        description='Timestamps for whisker_miss_trial',
        comments='time of each whisker_miss_trial event.',
    )
    behavior_events.add_timeseries(ts_whisker_miss)

    """
    # ---- "PiezoLickOnsets" ------
    PiezoLickOnset_Tms_CompleteLicks = np.asarray(data['PiezoLickOnset_Tms_CompleteLicks']).flatten()
    piezo_lick_series = TimeSeries(
        name='Lick_onset',
        data=np.ones_like(PiezoLickOnset_Tms_CompleteLicks), 
        unit='n.a.',
        timestamps=PiezoLickOnset_Tms_CompleteLicks,
        description='Timestamps marking the onset of all licking.',
        comments='time of detected licks from the piezo sensor.',
        rate=None,
    )
    behavior_events.add_timeseries(piezo_lick_series)

    """

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
        trace_keys = ["JawTrace", "TongueTrace", "WhiskerAngle"]

        for key in trace_keys:
            if key in data:
                trace = data[key]
                if key == "JawTrace" or key == "TongueTrace" or key == "NoseTopTrace" or key == "NoseSideTrace":
                    trace = trace/ 1000  
                values, times = np.asarray(trace)[0], VideoFrames_Tms
                if len(values) != len(times):
                    raise ValueError(f"Length mismatch: {key} has {len(values)} values but VideoFrames_Tms has {len(times)} timestamps.")
                if key == "WhiskerAngle":
                    description = "Whisker angle trace across aligned video_onsets."
                    comments = "the whisker angle is extracted from video filming using DeepLabCut 2.2b7 and is defined as the angle between the whisker shaft and the midline of the head."
                elif key == "JawTrace":
                    description = "Jaw traces aligned to video_onsets."
                    comments = "the jaw trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the distance between the tip of the jaw and the resting (closed) position of the jaw (mm)."
                elif key == "TongueTrace":
                    description = "Tongue traces aligned to video_onsets."
                    comments = "the tongue trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the distance between the tip of the tongue and the resting (closed) position of the jaw (mm). NB: tongue trace is only defined when the tongue is visible (protruded) otherwise = NaN"
                elif key == "NoseTopTrace":
                    description = "Nose top traces aligned to video_onsets."
                    comments = "the nose top trace trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the position of the nose relative to the resting position from the top view video."
                elif key == "NoseSideTrace":
                    description = "Nose side traces aligned to video_onsets."
                    comments = "the nose side trace trace is extracted from video filming using DeepLabCut 2.2b7 and is defined as the position of the nose relative to the resting position from the side view video."
                

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

