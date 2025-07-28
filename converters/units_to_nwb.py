
import h5py
import numpy as np
from pynwb.misc import Units

def add_units_container(nwb_file, data, unique_values, mat_file , sampling_rate ,regions):
    """
    Add or complete the 'units' container in the NWB file using neuronal spike and metadata.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to update.
    data : dict
        Dictionary loaded from the .mat file (already pre-loaded using h5py).
    electrode_table_region : pynwb.core.DynamicTableRegion
        A region referencing all electrodes (64).
    mat_file : str
        Path to the .mat file for dereferencing MATLAB cell arrays.
    """

        
    units_table = Units(
    name="units",
    description=(
    "Spike sorting was performed using Kilosort2.5 (https://github.com/MouseLand/Kilosort). "
    "Manual curation was applied to retain only well-isolated units based on the sortingQuality matrix "
    "(https:// github.com/cortex-lab/sortingQuality)."
    )
    )

    nwb_file.units = units_table

    #Other info we don't have: Electrode Groupe NO, max-Channel , bc_cluster_id , useTheseTimesStart, useTheseTimesStop, percentageSpikesMissing_gaussian, percentageSpikesMissing_symmetric, presenceRatio, ks labels NO,nPeaks, nTroughs,spatialDecaySlope, waveformBaselineFlatness, rawAmplitude, signalToNoiseRatio,waveform_mean,pt_ratio,
    with h5py.File(mat_file, 'r') as f:
        n_units = data["spikets"].shape[0]
        nwb_file.add_unit_column("cluster_id", "cluster index, from KS(probe-wise)")
        nwb_file.add_unit_column("main_channel", "Most responsive electrode index")
        nwb_file.add_unit_column("depth Histo", "Depth of the unit (um)")
        nwb_file.add_unit_column("firing_rate", "Mean firing-rate over trials (Hz)")
        nwb_file.add_unit_column("nSpikes", "number of spikes")
        nwb_file.add_unit_column("waveformDuration_peakTrough", "peak-to-trough template waveform duration, in ms")
        nwb_file.add_unit_column("fractionRPVs_estimatedTauR", "Percent of refractory period violations")
        nwb_file.add_unit_column("sampling_rate", "Sampling rate used for that probe, in Hz")
        nwb_file.add_unit_column("ccf_ml", "Allen CCF ML coordinate (um)")
        nwb_file.add_unit_column("ccf_ap", "Allen CCF AP coordinate (um)")
        nwb_file.add_unit_column("ccf_dv", "Allen CCF DV coordinate (um)")
        nwb_file.add_unit_column("ccf_id", "ccf region ID")
        nwb_file.add_unit_column("Target_area", "Localization target: mPFC (the medial prefrontal cortex : a higher-order area potentially involved in decision-making) or WS1 (whisker somatosensory area : the first cortical area involved in the processing of the whisker stimulus) or tjm1 (the primary tongue and jaw motor area : a motor area involved in the control of directional licking.)")
        nwb_file.add_unit_column("ccf_name (full)", "Full brain area name")
        nwb_file.add_unit_column("ccf_name (acronym)", "Acronym of the brain area")
        nwb_file.add_unit_column("Type of neuron", "Neuron type: RSU (Regular Spiking Unit > 0.34 ms), FSU (Fast Spiking Unit < 0.26 ms), NoA (No assignment)")        
        #nwb_file.add_unit_column("Baseline_fr_session", "Session-level baseline firing rate")
        nwb_file.add_unit_column("isi_violation", "ISI violation rate")
        nwb_file.add_unit_column("iso_distance", "Isolation distance")

        def get_str(ref):  # Convert HDF5 reference to string
            return ''.join(chr(c[0]) for c in f[ref][:])


        for i in range(n_units):
            # Spike times
            spike_ref = data["spikets"][i][0]
            spike_times = np.array(f[spike_ref]).flatten()
            nspikes = spike_times.shape[0]
            raw_coords_ref = data["ML_DV_AP"][i][0]
            coords = np.array(f[raw_coords_ref]).flatten()  # <- toujours 1D
            ml, dv, ap = coords.tolist()

            if coords.size != 3:
                print(f"Warning: Unit {i} has invalid CCF coords, skipping")
                continue
            
            if np.sum(regions) == 2:
                area_one, area_two = unique_values
                if get_str(data["Area"][i][0]) == area_one:
                    spike_main_channel = int(data["Spike_MainChannel"][i][0])
                elif get_str(data["Area"][i][0]) == area_two:
                    spike_main_channel = int(data["Spike_MainChannel"][i][0]) + 32
                spike_main_channel = spike_main_channel - 1
            elif np.sum(regions) == 1:
                area_one = unique_values
                spike_main_channel = int(data["Spike_MainChannel"][i][0]) - 1
            else:
                raise ValueError("Expected one or two unique areas, found: {}".format(np.sum(regions)))
            
            # Other metadata
            baseline_vec = np.array(data["BaselineFR_Mean"][i][0])
            unit_info = {
                "cluster_id": int(data["clusterID"][i][0]),
                "main_channel": spike_main_channel,
                "depth Histo": float(data["depthHisto"][i][0]/1000),
                "firing_rate": float(np.mean(baseline_vec)),
                'nSpikes': nspikes,
                "waveformDuration_peakTrough": float(data["width"][i][0]),
                "fractionRPVs_estimatedTauR": float(data["RP_Violation"][i][0]),
                'sampling_rate' : sampling_rate,
                "ccf_ml": float(ml),
                "ccf_ap": float(ap),
                "ccf_dv": float(dv),
                "ccf_id": int(data["ARAindex"][i][0]),
                "Target_area": get_str(data["Area"][i][0]),
                "ccf_name (full)": get_str(data["struct"][i][0]),
                "ccf_name (acronym)": get_str(data["struct_acr"][i][0]),
                "Type of neuron": get_str(data["type"][i][0]),
                #"baseline_fr_session": get_str(data["BaselineFR_Session"][i][0]),
                "isi_violation": float(data["ISI_Violation"][i][0]),
                "iso_distance": float(data["ISO_Distance"][i][0]),
            }

            nwb_file.add_unit(
                id=int(i),
                spike_times=spike_times,
                electrodes= [spike_main_channel],
                **unit_info
            )


