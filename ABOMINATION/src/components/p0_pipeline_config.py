# Pipeline options. One section per stage; each step reads its section as self.opts.
# Relative paths resolve against ABOMINATION/ (absolute paths pass through).

import os
from enum import StrEnum

class Dataset(StrEnum): # members compare equal to their string, so they work as dict keys / in JSON
    ARCHIVE = "archive"  # Kaggle .npy trials
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
        "debug_view": View.NONE,  # View.NONE | View.TABLES | View.NPY
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
            # assumed 10-20 order; FZ/CZ/PZ missing
            "channel_names": ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"],
            "name_pattern": r"(\w+)_subject_(\d+)_(\d+)\.npy",  # subtype, subject, file index
        },
    },
    "p2_sample_processing": {
        "out_dirs": {  # one output folder per dataset
            Dataset.ARCHIVE: "out/p2/Archive",
            Dataset.FIGSHARE: "out/p2/FIgshare",
        },
        "sfreq": 250.0,  # resample target (Hz)
        "l_freq": 1.0,  # bandpass low edge (Hz)
        "h_freq": 30.0,  # bandpass high edge (Hz)
        "eye_factor": 0.2,  # scale on the FP1/FP2 regression
        "eye_channels": ["FP1", "FP2"],  # regress on their mean when all present
        "non_eeg": ["HEOL", "HEOR", "VEOL", "VEOR", "TRIGGER"],  # dropped before CAR
        "processes": os.cpu_count(),  # worker pool size (all cores)
        "debug": True,  # True: print the first processed sample's signal
    },
    "p3_epoching": {
        # (duration s, overlap %) pairs; each p4 worker loads a subject once and runs every pair
        "combinations": [(d, o) for d in (4, 10) for o in (0, 50)],
        "clip": 20.0,  # clip z-scored epochs at ±clip SD
        "out_dirs": {  # debug only: <out_dir>/<duration>s/<overlap>pct/<subject>_e<idx>_t<start>s.npy
            Dataset.ARCHIVE: "out/p3/Archive",
            Dataset.FIGSHARE: "out/p3/FIgshare",
        },
        "debug": False,  # True: write every epoch to out_dirs as it is generated
    },
    "p4_features": {  # per-epoch features -> per-subject quantile summary
        "processes": os.cpu_count(),  # worker pool size (all cores; one subject per task)
        "quantiles": [10, 25, 50, 75, 90],  # percentiles of each feature across a subject's epochs
        "entropy_bins": 64,  # histogram bins over [-clip, clip] for Shannon entropy
        # Welch PSD per channel: 4 relative band powers + 6 pairwise band ratios (10 × 16 = 160); nperseg = min(win, max_nperseg_s × sfreq), floored at min_nperseg
        "welch": {"bands": {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}, "window": "hamming", "max_nperseg_s": 2.0, "min_nperseg": 64},
        # Morlet time-frequency PNG per subject: freqs (Hz), cycles per wavelet, time decimation, epochs per batch, image dpi
        "wavelet": {"enabled": True, "freqs": list(range(1, 31)), "n_cycles": 6, "decim": 5, "chunk": 16, "dpi": 90},
        # <out_dir>/<duration>s/<overlap>pct/: <subject>.csv (shape features), <subject>_welch.csv (Welch); rows = features, columns = quantiles; <subject>.png (wavelet TF)
        "out_dirs": {
            Dataset.ARCHIVE: "out/p4/Archive",
            Dataset.FIGSHARE: "out/p4/FIgshare",
        },
        "debug": False,  # True: print the first subject's summary
    },
}
