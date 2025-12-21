MRID - DICOM Classification Tool

This repository contains a GUI tool for quickly classifying DICOM series
into target MRI series types (T1/T2/FLAIR/T1c/Other).

Structure
- mrid_app/
  - \_\_init\_\_.py
  - config.py       # constants and model settings
  - utils.py        # small helper utilities
  - prep.py         # feature preprocessing and text encoding
  - header.py       # header-model training/prediction logic
  - cnn.py          # CNN model and image preprocessing helpers
  - gui.py          # Tkinter GUI and control flow
- run.py            # entrypoint to launch the GUI

Getting started
1. Create a virtual environment and install requirements:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Launch the GUI:

   python run.py

