
# 🧠 NWB Conversion Pipeline for AN Sessions

This project provides a conversion pipeline for behavioral and electrophysiological data from the article: **Oryshchuk et al., 2024, Cell Reports** into the standard **Neurodata Without Borders (NWB)** format.

## 📚 Reference

Oryshchuk et al., *Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during goal-directed behavior*, Cell Reports, 2024.  
👉 [DOI](https://doi.org/10.1016/j.celrep.2023.113618)



## ⚙️ Features

- Reads `.mat` files containing raw data and `.csv` files containing subject metadata
- Converts to NWB structure including:
  - General metadata (subject, session…)
  - Time intervals (e.g., trials)
  - Units (e.g., spikes)
  - Behavioral data (licks, rewards…)
- Validates the NWB file after conversion



## 📁 Project Structure

```

NWB\_converter\_AN/
│
├── converters/
│   ├── acquisition\_to\_nwb.py
│   ├── behavior\_to\_nwb.py
│   ├── general\_to\_nwb.py
│   ├── intervals\_to\_nwb.py
│   ├── units\_to\_nwb.py
│   ├── Initiation\_nwb.py
│   └── nwb\_saving.py
├── Subject\_Session\_Selection.csv 
├── convert\_to\_nwb\_for\_AO.py  ← Main conversion script

````



## 💻 Work Environment
Follow the environment setup instructions provided in [LSENS-Lab-Immersion repository](https://github.com/loris-fab/LSENS-Lab-Immersion.git), and include the link to it.

## 🧩 How to use

Please find below the key information

1. `input_folder` → path to the directory ontaining Anastasiia Oryshchuk’s `.mat` files.
2. `output_folder` → directory where you want the NWB file to be saved


### Commande in the terminal
Run the following command in the terminal, replacing the arguments :

```bash
python convert_to_nwb_for_AO.py input_folder output_folder
```

for exemple :
```bash
python convert_to_nwb_for_AO.py //sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Sylvain_Crochet/DATA_REPOSITORY/Oryshchuk_Spike&LFP_2024/WR- mice //sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/AO/NWB_files
```
BE CARFULE : The .mat files must be retrieved from either the **WR- mice** or **WR+ mice** folders. So add it to the input_folder_path

### Run inside a Jupyter Notebook

You can also call the conversion function directly in a Jupyter Notebook without using the command line.
Simply import the function `convert_data_to_nwb_pl` from your script and call it with the proper arguments:

*for exemple for window:* 
```python
from convert_to_nwb_for_AO import convert_data_to_nwb_an

convert_data_to_nwb_an(
    input_folder = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Sylvain_Crochet/DATA_REPOSITORY/Oryshchuk_Spike&LFP_2024/WR- mice",
    output_folder= "//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/AO/NWB_files",
)
```
### Outcome
If everything runs correctly, you should see an output similar to this:

```bash
**************************************************************************
-_-_-_-_-_-_-_-_-_-_-_-_-_-NWB conversion_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
Converting data to NWB format for mouse: ['AO028'...]
Conversion to NWB is finished: 100%|████████| 1/1 [00:35<00:00, 35.98s/it]
**************************************************************************
```




## ✍️ Author

Project developed as part of a student project focused on organizing and converting neuroscience data from the above-mentioned publication.
Main code by **@loris-fab**

For any questions related to the code, please contact: loris.fabbro@epfl.ch


---
