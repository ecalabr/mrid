MRID - DICOM Classification Tool

This repository contains a GUI tool for quickly classifying DICOM series
into target MRI series types (T1/T2/FLAIR/T1c/Other). The original
single-file script has been refactored into a package for clarity.

Structure
- mrid.py (original single-file script - kept for reference)
- mrid_app/
  - __init__.py
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

Notes
- The refactor splits responsibilities across modules but keeps the
  user-facing behaviour the same.
- CNN/image training code is included but may require GPU-enabled PyTorch
  and a non-empty dataset to run successfully.
- The SentenceTransformer model will be downloaded on first run.

Next steps / Improvements
- Fully move the remaining classification logic (file scanning,
  hierarchy building, and per-study GUI) into modules for testability.
- Add unit tests and CI configuration.
- Provide CLI wrappers for non-GUI usage.
