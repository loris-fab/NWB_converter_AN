
# 🧠 NWB Conversion Pipeline for AN Sessions

This project provides a conversion pipeline for behavioral and electrophysiological data from the article: **Oryshchuk et al., 2024, Cell Reports** into the standard **Neurodata Without Borders (NWB)** format.

## 📚 Reference

Oryshchuk et al., *Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during goal-directed behavior*, Cell Reports, 2024.  
👉 [DOI](https://doi.org/10.1016/j.celrep.2023.113618)



## ⚙️ Features

- Reads `.mat` files containing raw data  
- Converts to NWB structure including:
  - General metadata (subject, session…)
  - Time intervals (e.g., trials)
  - Units (e.g., spikes)
  - Behavioral data (licks, rewards…)
  - Optional analysis containers (e.g., LFPmean)
- Validates the NWB file after conversion



## 📁 Project Structure

```

NWB\_converter\_AN/
│
├── converters/
│   ├── acquisition\_to\_nwb.py
│   ├── analysis\_to\_nwb.py
│   ├── behavior\_to\_nwb.py
│   ├── general\_to\_nwb.py
│   ├── intervals\_to\_nwb.py
│   ├── units\_to\_nwb.py
│   ├── Initiation\_nwb.py
│   └── nwb\_saving.py
│
├── Subject\_Session\_Selection.csv 
├── requirement.txt
├── convert\_to\_nwb\_for\_AO.py  ← Main conversion script

````

---

## 🚀 Usage

Create environment and Install dependencies with:
```bash
conda env create -f environment.yml
```

if it doesn't work try : 
```bash
conda create -n nwb_env python=3.9
conda activate nwb_env
pip install -r requirement.txt
```

## 🧩 How to use
Run the following command in the terminal, replacing `input_folder` with the path to the folder containing Anastasiia Oryshchuk’s `.mat` files, and `output_folder` with the destination directory where you want the resulting NWB files to be saved.

```bash
python convert_to_nwb_for_AO.py input_folder output_folder
```
*Options:*
* `--print_progress`: Print progress of conversion with more details

If everything runs correctly, you should see an output similar to this:

```bash
Conversion to NWB: 🔁 AO039_20190626.mat:  50%|███████████           | 1/2 [00:00<00:00,  7.01it/s]
**************************************************************************
 NWB conversion /Volumes/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/AO/mat_files/AO039_20190626.mat
 
📃 Creating config file for NWB conversion :
📑 Created NWB file :
ephys Whisker Rewarded: Whisker-rewarded (WR+) mice were trained to lick within 1 s following the whisker stimulus (go trials) but not in the absence of the stimulus (no-go trials). The neuronal representation of sensory, motor, and decision information was studied in a sensory, a motor, and a higher-order cortical area in these mice trained to lick for a water reward in response to a brief whisker stimulus.
     o 📌 Add general metadata
         - Subject metadata
         - Session metadata
         - Device metadata
         - Extracellular electrophysiology metadata
     o ⏸️ Add intervall container
     o 🧠 Add units container
     o ⚙️ Add processing container
         - Behavior data
         - No ephys data for AN sessions
 
🔎 Validating NWB file before saving...
     o ✅ File is valid, no errors detected.
 
💾 Saving NWB file
     o 📂 NWB file saved at:
         - /Volumes/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/AO/NWB_files/AO039_20190626_160524.nwb
**************************************************************************
Conversion to NWB is finished: 100%|████████████████████████████| 2/2 [00:59<00:00, 29.78s/it]
⚠️ Conversion completed with errors for 1 files
    - PL200_D1.mat: Unable to open file (file signature not found)
```




## ✍️ Author

Project developed as part of a student project focused on organizing and converting neuroscience data from the above-mentioned publication.
Main code by **@loris-fab**

For any questions related to the code, please contact: loris.fabbro@epfl.ch


---


