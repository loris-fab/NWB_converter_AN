
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
├── Subject\_Session\_Selection.csv 
├── requirement.txt
├── convert\_to\_nwb\_for\_AO.py  ← Main conversion script

````

---

## 🚀 Usage
Follow the environment setup instructions provided in [LSENS-Lab-Immersion repository](https://github.com/loris-fab/LSENS-Lab-Immersion.git), and include the link to it.

## 🧩 How to use
Run the following command in the terminal, replacing `input_folder` with the path to the folder containing Anastasiia Oryshchuk’s `.mat` files, and `output_folder` with the destination directory where you want the resulting NWB files to be saved.

```bash
python convert_to_nwb_for_AO.py input_folder output_folder
```
*Options:*
* `--print_progress`: Print progress of conversion with more details

for exemple :
```bash
python convert_to_nwb_for_AO.py data/mouse_anastasia/WR data/output
```

If everything runs correctly, you should see an output similar to this:

```bash
**************************************************************************
-_-_-_-_-_-_-_-_-_-_-_-_-_-NWB conversion_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
Conversion to NWB is finished: 100%|████████| 1/1 [00:35<00:00, 35.98s/it]
**************************************************************************
```




## ✍️ Author

Project developed as part of a student project focused on organizing and converting neuroscience data from the above-mentioned publication.
Main code by **@loris-fab**

For any questions related to the code, please contact: loris.fabbro@epfl.ch


---
