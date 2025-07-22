from pynwb.ecephys import ElectricalSeries
import h5py
import numpy as np

#####################################################
# Functions that handle LFP acquisition in NWB files
#####################################################

def add_lfp_acquisition(nwb_file, signal_array, electrode_region):
    """
    Add LFP signal to NWB file acquisition as an ElectricalSeries.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
    signal_array : np.ndarray, shape (n_timepoints, n_channels)
        The LFP signal data.
    electrode_region : DynamicTableRegion
    """
    sampling_rate = float(2000)

    e_series = ElectricalSeries(
        name="ElectricalSeries_LFP",                             
        data=signal_array,
        electrodes=electrode_region,
        starting_time=0.0,
        rate=sampling_rate,
        description=f"Raw acquisition traces: Local Field Potential from {signal_array.shape[1]} electrodes"
    )
    nwb_file.add_acquisition(e_series)


def extract_lfp_signal(data, mat_file):
    """
    Extract and merge LFP signals from two shanks into one array of shape (T, n_channels).

    Parameters
    ----------
    data : dict
        Dictionary loaded from .mat (with h5py).
    mat_file : str
        Path to the original .mat file.

    Returns
    -------
    np.ndarray
        Array of shape (n_timepoints, n_channels)
    """
    with h5py.File(mat_file, 'r') as f:
        lfp_refs = data["LFPs"]  # shape (2, 1)
        blocks = []
        WS1, mPFC, tjM1 = True, True, True
        for i in range(3):  # 3 shanks
            ref = lfp_refs[i][0] if hasattr(lfp_refs[i], '__getitem__') else lfp_refs[i]
            mat = np.array(f[ref])
            if mat.ndim != 2:
                if i == 0:
                    WS1 = False
                elif i == 1:
                    mPFC = False
                elif i == 2:
                    tjM1 = False
            else:
                mat = mat.T                   
                blocks.append(mat)
        if not blocks:
            raise ValueError("All blocks are empty. Cannot extract LFP signal.")
        full_array = np.concatenate(blocks, axis=0)

    return full_array.T , [WS1, mPFC, tjM1]  # Transpose to shape (T, 32 or 64 or 96)
