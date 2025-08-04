
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
Run the following command in the terminal, replacing `path.mat` with the path to the `.mat` file provided by Anastasiia Oryshchuk, and `output_folder` with the directory where you want the NWB file to be saved.

```bash
python convert_to_nwb_for_AO.py path.mat output_folder
```
*Options:*

* `--psth_window`: time window for PSTH (default: -0.2 0.5 seconds)
* `--psth_bin`: bin width for PSTH (default: 0.010 seconds)


If everything runs correctly, you should see an output similar to this:


```bash
**************************************************************************
-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_

📃 Creating config file for NWB conversion:

📑 Creating NWB file:
ephys Whisker Rewarded: Acute extracellular recordings using NeuroNexus single-shank 32-channel probes. Bandpass filtered (0.3 Hz – 7.5 kHz), amplified and digitized at 30 kHz (CerePlex M32, Blackrock). Data recorded via CerePlex Direct system.

     o 📌 Adding general metadata:
         - Subject metadata
         - Session metadata
         - Device metadata
         - Extracellular electrophysiology metadata
     o 📶 Add acquisition container
     o ⏸️ Adding interval container  
     o 🧠 Adding units container  
     o ⚙️ Adding processing container:
         - Behavioral data
         - No electrophysiology data for AN sessions
         - Complementary analysis information
            > Added PSTH_mean_across_all_units to analysis module
            > Added LFP_mean_across_all_units to analysis module
            > Added global_LFP to analysis module
🔎 Validating NWB file before saving...
     o ✅ File is valid, no errors detected.

💾 Saving NWB file:
     o 📂 NWB file saved at:
         - data/AO039_20190626_160524.nwb
**************************************************************************
```




## ✍️ Author

Project developed as part of a student project focused on organizing and converting neuroscience data from the above-mentioned publication.
Main code by **@loris-fab**

For any questions related to the code, please contact: loris.fabbro@epfl.ch


---


