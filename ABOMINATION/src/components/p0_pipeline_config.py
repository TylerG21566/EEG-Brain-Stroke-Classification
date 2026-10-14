# Pipeline options. One section per stage; each step reads its section as self.opts.
# Relative paths resolve against ABOMINATION/ (absolute paths pass through).

from enum import StrEnum

class Dataset(StrEnum): # members compare equal to their string, so they work as dict keys / in JSON
    ARCHIVE = "archive"  # Tamim's Kaggle .npy trials
    FIGSHARE = "figshare"  # Liu & Lv stroke EDFs

class View(StrEnum): # p1 debug popup
    NONE = "none"  # no popup
    TABLES = "tables"  # full p1 tables in browser
    NPY = "npy"  # first recording only

CONFIG = {
    "paths": {
        "data": "data",
        "out": "out",
    },
    "p1_data_formatting": {
        "dataset": Dataset.ARCHIVE,  # Dataset.ARCHIVE | Dataset.FIGSHARE
        "debug_view": View.NPY,  # View.NONE | View.TABLES | View.NPY
        "view_samples": 1024,  # samples shown by View.NPY
        "debug_dir": "out/p1_debug",  # where the debug CSVs/HTML go
        "roots": {
            Dataset.ARCHIVE: "data/Tamim-data",
            Dataset.FIGSHARE: "data/FIgshare",
        },
        "archive": {
            "channels": 16,
            "samples": 1024,
            "sfreq": 256.0,
            # Tamim's assumed 10-20 order (his archive_channel_mapping.json); FZ/CZ/PZ missing
            "channel_names": ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"],
            "name_pattern": r"(\w+)_subject_(\d+)_(\d+)\.npy",  # subtype, subject, file index
        },
    },
    "p2_sample_processing": {
        "out_dirs": {  # one output folder per dataset
            Dataset.ARCHIVE: "out/p2/Tamim",
            Dataset.FIGSHARE: "out/p2/FIgshare",
        },
        "sfreq": 250.0,  # resample target (Hz)
        "l_freq": 1.0,  # bandpass low edge (Hz)
        "h_freq": 30.0,  # bandpass high edge (Hz)
        "eye_factor": 0.2,  # scale on the FP1/FP2 regression
        "eye_channels": ["FP1", "FP2"],  # regress on their mean when all present
        "non_eeg": ["HEOL", "HEOR", "VEOL", "VEOR", "TRIGGER"],  # dropped before CAR
        "processes": 4,  # worker pool size
    },
}
