from pynwb.ecephys import ElectricalSeries
import h5py
import numpy as np

def add_lfp_acquisition(nwb_file, signal_array, electrode_region):
    """
    Add LFP signal to NWB file acquisition as an ElectricalSeries.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
    signal_array : np.ndarray, shape (n_timepoints, 64)
    electrode_region : DynamicTableRegion
    """
    sampling_rate = float(2000)

    e_series = ElectricalSeries(
        name="ElectricalSeries : LFP",                             
        data=signal_array,
        electrodes=electrode_region,
        starting_time=0.0,
        rate=sampling_rate,
        description="Raw acquisition traces: Local Field Potential from 64 electrodes"
    )
    nwb_file.add_acquisition(e_series)


def extract_lfp_signal(data, mat_file):
    """
    Extract and merge LFP signals from two shanks into one array of shape (T, 64).

    Parameters
    ----------
    data : dict
        Dictionary loaded from .mat (with h5py).
    mat_file : str
        Path to the original .mat file.

    Returns
    -------
    np.ndarray
        Array of shape (n_timepoints, 64)
    """
    with h5py.File(mat_file, 'r') as f:
        lfp_refs = data["LFPs"]  # shape (2, 1)
        blocks = []

        for i in range(2):  # 2 shanks
            ref = lfp_refs[i][0] if hasattr(lfp_refs[i], '__getitem__') else lfp_refs[i]
            mat = np.array(f[ref])      
            mat = mat.T                   
            blocks.append(mat)

        full_array = np.concatenate(blocks, axis=0)

    return full_array.T # Transpose to shape (T, 64)
