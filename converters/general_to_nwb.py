
from uuid import uuid4
import numpy as np
import h5py


def add_general_container(nwb_file, data, mat_file, regions):
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
        manufacturer="NeuroNexus, MI, USA"
    )

    # ── Headstage (Blackrock) ───────────────────────────────────────────
    headstage = nwb_file.create_device(
        name="CerePlex M32",
        description="Digital headstage used for 0.3 Hz–7.5 kHz amplification and 30 kHz digitization",
        manufacturer="Blackrock Microsystems, UT, USA"
    )

    # -- High speed camera ────────────────────────────────────────────
    camera = nwb_file.create_device(
        name="High speed camera",
        description="CL 600 X 2/M",
        manufacturer="Optronis, Germany"
    )

    # ── Data acquisition system (Blackrock) ───────────────────────────
    Data_acquisition_system = nwb_file.create_device(
        name="CerePlex Direct",
        description= "Data acquisition system for recording",
        manufacturer="Blackrock Microsystems, UT, USA"
    )

    # ##############################################################
    # 2. Create Electrode Group and Add electrodes to the NWB file
    # ##############################################################

    if np.sum(regions) == 2:
        ml_dv_ap  = np.asarray(data.get("ML_DV_AP_32"))   
        with h5py.File(mat_file, 'r') as f:
            area_array = np.array([
            f[ref[0]][()].tobytes().decode('utf-16le').strip()
            for ref in data['Area']])

            unique_values, first_indices = np.unique(area_array, return_index=True)

            ref = ml_dv_ap[first_indices[0]][0] if hasattr(ml_dv_ap[1], '__getitem__') else ml_dv_ap[first_indices[0]]
            obj = f[ref]
            shank1 = np.array(obj)
            ref = ml_dv_ap[first_indices[1]][0] if hasattr(ml_dv_ap[-1], '__getitem__') else ml_dv_ap[first_indices[1]]
            obj = f[ref]
            shank2 = np.array(obj)
            shank_total = np.concatenate((shank1, shank2), axis=1)
            assert shank_total.shape == (3, 64), "Expected shape of shank_total is (3, 64), got {}".format(shank_total.shape)

        if unique_values[0] == "mPFC":
            info_reg = "(the medial prefrontal cortex : a higher-order area potentially involved in decision-making)"
        elif unique_values[0] == "tjM1":
            info_reg = "(the primary tongue and jaw motor area : a motor area involved in the control of directional licking)"
        elif unique_values[0] == "Ws1":
            info_reg = "(the primary whisker sensory cortex: the first cortical area involved in the processing of the whisker stimulus)"

        if unique_values[1] == "mPFC":
            info_reg = "(the medial prefrontal cortex : a higher-order area potentially involved in decision-making)"
        elif unique_values[1] == "tjM1":
            info_reg = "(the primary tongue and jaw motor area : a motor area involved in the control of directional licking)"
        elif unique_values[1] == "Ws1":
            info_reg = "(the primary whisker sensory cortex: the first cortical area involved in the processing of the whisker stimulus)" 

        # Create group for each shank
        shank_1 = nwb_file.create_electrode_group(
            name="Shank one",
            description="NeuroNexus A1x32 probe, Shank 1",
            location="In the {} according to the Allen Brain Atlas".format(unique_values[0] + info_reg),
            device=probe
        )
        shank_2 = nwb_file.create_electrode_group(
            name="Shank two",
            description="NeuroNexus A1x32 probe, Shank 2",
            location="In the {} according to the Allen Brain Atlas".format(unique_values[1] + info_reg),
            device=probe
        )

        with h5py.File(mat_file, 'r') as f:
            nwb_file.add_electrode_column(name="ccf_ml", description="ccf coordinate in ml axis")
            nwb_file.add_electrode_column(name="ccf_ap", description="ccf coordinate in ap axis")
            nwb_file.add_electrode_column(name="ccf_dv", description="ccf coordinate in dv axis")

            # Add electrodes from Shank1
            for i in range(64):
                ml, dv, ap = shank_total[:, i]
                if i < 32:
                    shank = shank_1
                    loca = unique_values[0]
                else:
                    shank = shank_2
                    loca = unique_values[1]

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

    if np.sum(regions) == 3:
        raise ValueError("This function currently supports only 2 regions (e.g., Shank1 and Shank2). Please check the regions provided in the data.")

    if np.sum(regions) == 1:
        ml_dv_ap  = np.asarray(data.get("ML_DV_AP_32"))   
        with h5py.File(mat_file, 'r') as f:
            ref = ml_dv_ap[1][0] if hasattr(ml_dv_ap[1], '__getitem__') else ml_dv_ap[1]
            obj = f[ref]
            shank_total = np.array(obj)
            assert shank_total.shape == (3, 32), "Expected shape of shank1 is (3, 32), got {}".format(shank_total.shape)

        if regions[0] == 1:
            unique_values = "Ws1"
            info_reg = "(the primary whisker sensory cortex: the first cortical area involved in the processing of the whisker stimulus)"
        elif regions[1] == 1:
            unique_values = "mPFC"
            info_reg = "(the medial prefrontal cortex : a higher-order area potentially involved in decision-making)"
        elif regions[2] == 1:
            unique_values = "tjM1"
            info_reg = "(the primary tongue and jaw motor area : a motor area involved in the control of directional licking)"

        # Create group for each shank
        shank_1 = nwb_file.create_electrode_group(
            name="Shank one",
            description="NeuroNexus A1x32 probe, Shank 1",
            location="In the {} according to the Allen Brain Atlas".format(unique_values + info_reg),
            device=probe
        )

        with h5py.File(mat_file, 'r') as f:
            nwb_file.add_electrode_column(name="ccf_ml", description="ccf coordinate in ml axis")
            nwb_file.add_electrode_column(name="ccf_ap", description="ccf coordinate in ap axis")
            nwb_file.add_electrode_column(name="ccf_dv", description="ccf coordinate in dv axis")

            # Add electrodes from Shank1
            for i in range(32):
                ml, dv, ap = shank_total[:, i]
                nwb_file.add_electrode(
                    id=i,
                    location=unique_values,
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
                    group = shank_1,
                )

    # ##############################################################
    # 3. Return region (useful for linking to ElectricalSeries)
    # ##############################################################

    electrode_table_region = nwb_file.create_electrode_table_region(
        region=list(range(shank_total.shape[1])), 
        description="All electrodes from Shank1 and Shank2"
    )
    return electrode_table_region, unique_values 
