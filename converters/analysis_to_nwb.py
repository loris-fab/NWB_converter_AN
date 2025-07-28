# converters/analysis_to_nwb.py
import numpy as np
from pynwb import ProcessingModule
from pynwb.base import TimeSeries
import matplotlib.pyplot as plt

#######################################################
# Functions for converting analysis data to NWB format
#######################################################


def add_analysis_container(
    *,
    nwb_file, Rewarded,
    psth_window,         # seconds around stimulus
    psth_bin                  # 10-ms bins
):
    """
    Populate ``nwb_file.analysis`` with:
      • a PSTH (all units, stimulus-aligned)
      • a mean stimulus-locked LFP trace per channel

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
    data : dict
        Already-loaded MATLAB 'Data' struct (keys → ndarrays / HDF refs)
    Rewarded : bool
        True if the session is rewarded, False otherwise
    psth_window : tuple
        (start, stop) window around each stimulus for the PSTH (seconds)
    psth_bin : float
        Bin width for PSTH (seconds)
    """

    ####################################################
    ###  Make / get the "analysis" processing module ###
    ####################################################
    
    ana_mod = nwb_file.create_processing_module(
            name="analysis",
            description="Secondary analyses: mean LFP and global LFP")

    #################
    ###   PSTH    ###
    #################

    start_w, stop_w = psth_window
    n_bins = int(np.ceil((stop_w - start_w) / psth_bin))
    bin_edges = np.linspace(start_w, stop_w, n_bins + 1, dtype=np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # prepare output (n_units × n_bins)

    n_units = len(nwb_file.units['spike_times'].data[:])
    psth_matrix = np.zeros((n_units, n_bins), dtype=np.float32)

    if Rewarded:
        TrialOnsets_All = nwb_file.processing['behavior'].data_interfaces['BehavioralEvents'].time_series["TrialOnsets"].timestamps[:]
        stim_indice = nwb_file.processing['behavior'].data_interfaces['BehavioralEvents'].time_series["StimFlags"].data[:]
        stim_times = TrialOnsets_All[stim_indice >= 1]
    else:
        stim_times = nwb_file.processing['behavior'].data_interfaces['BehavioralEvents'].time_series["StimFlags"].timestamps[:]

    for u in range(n_units): # loop over units
        # spike times for unit u
        spk_times = nwb_file.units['spike_times'][u]

        rel_times = np.concatenate([spk_times - s for s in stim_times])

        # histogram
        counts, _ = np.histogram(rel_times, bins=bin_edges)
        # convert to rate (Hz): counts / (n_trials * bin_width)
        psth_matrix[u, :] = counts / (len(stim_times) * psth_bin)

    """
    psth_ts = TimeSeries(
        name="PSTH_all_units",
        data=psth_matrix.T,
        unit="Hz",
        timestamps=bin_centers,
        description=f"PSTH (bin={psth_bin*1e3:.0f} ms) averaged over {len(stim_times)} stimulation; "
                    f"window {start_w}s → {stop_w}s around whisker stimulus.",
        comments="Rows = units, columns = time-bins"
    )
    ana_mod.add_data_interface(psth_ts)
    """

    timestamps = bin_centers
    data_all_units = psth_matrix.T  # shape = (n_timepoints, n_units)
    mean_psth = np.mean(data_all_units, axis=1)

    # Create a TimeSeries for the mean PSTH
    psth_mean_ts = TimeSeries(
        name="PSTH_mean",
        data=mean_psth,
        unit='Hz',
        timestamps=timestamps,
        description=f"Mean PSTH across all units (bin={psth_bin*1e3:.0f} ms) averaged over {len(stim_times)} stimulation; "
                    f"window {start_w}s → {stop_w}s around whisker stimulus."
    )

    ana_mod.add_data_interface(psth_mean_ts)

    # plot an PSTHmean
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, mean_psth, label="Mean PSTH")
    plt.axvline(x=0, color='red', linestyle='--', label="Stimulus")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("PSTH - Mean Across Units")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    #plt.savefig("data/analysis/psth_mean.png")
    #plt.close()
    print("             > Added PSTH_mean_across_all_units to analysis module")


    ###################
    ###  Mean LFP   ###
    ###################

    # fetch raw LFP acquisition
    lfp_acq = nwb_file.acquisition["ElectricalSeries_LFP"]
    lfp_rate = lfp_acq.rate
    lfp_data = lfp_acq.data 

    # determine sample indices for the window
    idx_pre  = int(np.floor(start_w * lfp_rate))
    idx_post = int(np.ceil(stop_w  * lfp_rate))
    span = idx_post - idx_pre

    mean_lfp = np.zeros((lfp_data.shape[1], span), dtype=np.float32)

    for s in stim_times:
        i_start = int(np.round((s - lfp_acq.starting_time) * lfp_rate)) + idx_pre
        i_end   = i_start + span
        # skip if window exceeds bounds
        if i_start < 0 or i_end > lfp_data.shape[0]:
            continue
        mean_lfp += lfp_data[i_start:i_end, :].T   # channels × time

    mean_lfp /= len(stim_times)

    lfp_times = np.linspace(start_w, stop_w, span, endpoint=False)

    mean_lfp_ts = TimeSeries(
        name="MeanLFP_all_electrodes",
        data=mean_lfp.T,
        unit="volts",
        timestamps=lfp_times,
        description="Stimulus-aligned average LFP (channels × time)",
        comments="Averaged across all trials; same window and alignment as PSTH."
    )
    ana_mod.add_data_interface(mean_lfp_ts)

    # Compute mean across electrodes (1D signal : time only)
    mean_lfp_all_channels = mean_lfp.mean(axis=0)  # shape: (time,)

    # Create new TimeSeries for the global mean LFP
    mean_lfp_global_ts = TimeSeries(
        name="MeanLFP_global",
        data=mean_lfp_all_channels,
        unit="volts",
        timestamps=lfp_times,
        description="Average LFP across all electrodes, aligned to stimulus",
        comments="Mean of MeanLFP_all_electrodes across channels and averaged across trials; same window and alignment as PSTH."
    )

    # Add it to the analysis module
    ana_mod.add_data_interface(mean_lfp_global_ts)


    # plot the mean LFP
    plt.figure(figsize=(8, 4))
    for ch in range(min(3, mean_lfp.shape[0])):
        plt.plot(lfp_times, mean_lfp[ch, :], label=f"Channel {ch+1}")
    plt.axvline(x=0, color='red', linestyle='--', label="Stimulus")
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (V)")
    plt.title("Mean LFP - First 3 Channels")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    #plt.savefig("data/analysis/mean_lfp_example.png")
    #plt.close()
    print("             > Added LFP_mean_across_all_units to analysis module")

    # plot the global mean LFP
    plt.figure(figsize=(8, 4))
    plt.plot(lfp_times, mean_lfp_all_channels, label="Global Mean LFP")
    plt.axvline(x=0, color='red', linestyle='--', label="Stimulus")
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (V)")
    plt.title("Global Mean LFP")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    #plt.savefig("data/analysis/mean_lfp_global.png")
    #plt.close()
    print("             > Added global_LFP to analysis module")

    return None