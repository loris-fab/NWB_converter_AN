
from uuid import uuid4
import numpy as np
import h5py


def add_general_container(nwb_file, data, mat_file):
    """
    Add general metadata including devices and extracellular electrophysiology to the NWB file.
    
    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The existing NWB file object to update.
    data : dict
        Dictionary from the .mat file (already loaded, e.g., via h5py).
    mat_file : str
        Path to the original .mat file.

    Returns
    -------
    electrode_table_region : pynwb.core.DynamicTableRegion
        A region referencing all electrodes. Useful when creating ElectricalSeries.
    """

    # ##############################################################
    # 1. Add Device (e.g., Neuropixels probe)
    # ##############################################################

    # ── Probe (NeuroNexus) ──────────────────────────────────────────────
    probe = nwb_file.create_device(
        name="NeuroNexus A1x32",
        description=(
            "Single-shank silicon probe A1x32-Poly2-10 mm-50 s-177 "
            "(32 recording sites covering ~775 µm of cortical depth)"
        ),
        manufacturer="NeuroNexus"
    )

    # ── Headstage (Blackrock) ───────────────────────────────────────────
    headstage = nwb_file.create_device(
        name="CerePlex M32",
        description="Digital headstage used for 0.3 Hz–7.5 kHz amplification and 30 kHz digitization",
        manufacturer="Blackrock Microsystems"
    )

    # ##############################################################
    # 2. Create Electrode Group and Add electrodes to the NWB file
    # ##############################################################

    ml_dv_ap  = np.asarray(data.get("ML_DV_AP_32"))   
    with h5py.File(mat_file, 'r') as f:
        ref = ml_dv_ap[1][0] if hasattr(ml_dv_ap[1], '__getitem__') else ml_dv_ap[1]
        obj = f[ref]
        shank1 = np.array(obj)
        ref = ml_dv_ap[-1][0] if hasattr(ml_dv_ap[-1], '__getitem__') else ml_dv_ap[-1]
        obj = f[ref]
        shank2 = np.array(obj)

        area_array = np.array([
        f[ref[0]][()].tobytes().decode('utf-16le').strip()
        for ref in data['Area']])

        if len(np.unique(area_array)) == 2:
            area_one, area_two = np.unique(area_array)
        elif len(np.unique(area_array)) == 1:
            area_one = area_two = np.unique(area_array)[0]
        else:
            raise ValueError("Expected one or two unique areas, found: {}".format(np.unique(area_array)))

    shank_total = np.concatenate((shank1, shank2), axis=1)

    # Sanity check
    assert shank_total.shape == (3, 64), "Expected shape of shank_total is (3, 64), got {}".format(shank_total.shape)

    # Create group for each shank
    shank_1 = nwb_file.create_electrode_group(
        name="Shank one",
        description="NeuroNexus A1x32 probe, Shank 1",
        location="In the {} according to the Allen Brain Atlas".format(area_one),
        device=probe
    )
    shank_2 = nwb_file.create_electrode_group(
        name="Shank two",
        description="NeuroNexus A1x32 probe, Shank 2",
        location="In the {} according to the Allen Brain Atlas".format(area_two),
        device=probe
    )

    with h5py.File(mat_file, 'r') as f:
        nwb_file.add_electrode_column(name="ccf_ml", description="ccf coordinate in ml axis")
        nwb_file.add_electrode_column(name="ccf_ap", description="ccf coordinate in ap axis")
        nwb_file.add_electrode_column(name="ccf_dv", description="ccf coordinate in dv axis")

        # Add electrodes from Shank1
        for i in range(64):
            ml, dv, ap = shank_total[:, i]
            if i < 33:
                shank = shank_1
                loca = area_one
            else:
                shank = shank_2
                loca = area_two
            nwb_file.add_electrode(
                id=i,
                location=loca,
                # Group, group_name,index_on_probe,
                ccf_ml=ml,
                ccf_dv=dv,
                ccf_ap=ap,
                # shank_col,shank_row,ccf_id,ccf_acronym,cff_name,cff_parent_id,cff_parent_acronym,cff_parent_name,
                #rel_x = np.nan,  # x coordinate in the probe space
                #rel_y = np.nan,  # y coordinate in the probe space
                #rel_z = np.nan,  # z coordinate in the probe space
                #imp=np.nan,
                #filtering="none",
                group = shank,
            )

    #nwb_file.electrodes.to_dataframe()
    #display(nwb_file.electrodes.to_dataframe())

    # ##############################################################
    # 3. Return region (useful for linking to ElectricalSeries)
    # ##############################################################

    electrode_table_region = nwb_file.create_electrode_table_region(
        region=list(range(64)), 
        description="All electrodes from Shank1 and Shank2"
    )
    return electrode_table_region
