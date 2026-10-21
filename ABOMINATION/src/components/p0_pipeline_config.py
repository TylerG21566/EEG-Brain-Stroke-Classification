# Pipeline options. One section per stage; each step reads its section as self.opts.
# Relative paths resolve against ABOMINATION/ (absolute paths pass through).

import os
from enum import StrEnum

class Dataset(StrEnum): # members compare equal to their string, so they work as dict keys / in JSON
    ARCHIVE = "archive"  # Kaggle .npy trials
    FIGSHARE = "figshare"  # Liu & Lv stroke EDFs
    TUH = "tuh"  # TUH EEG corpus v2.0.2, stroke vs healthy sessions labelled in the headers db

class View(StrEnum): # p1 debug popup
    NONE = "none"  # no popup
    TABLES = "tables"  # full p1 tables in browser
    NPY = "npy"  # first recording only

class Selector(StrEnum): # p5 feature scoring
    MUTUAL_INFO = "mutual_info"  # sklearn mutual_info_classif (Tamim's SelectKBest choice)
    ANOVA = "anova"  # sklearn f_classif: fast, linear separation only

class ModelName(StrEnum): # p6 classifier; add RNN etc. here as new folders in components/models/
    XGBOOST = "xgboost"  # gradient-boosted trees (Tamim §4.7.1)

CONFIG = {
    "paths": {
        "data": "data",
        "out": "out",
    },
    "p1_data_formatting": {
        "dataset": Dataset.TUH,  # Dataset.ARCHIVE | Dataset.FIGSHARE | Dataset.TUH
        "debug_view": View.NONE,  # View.NONE | View.TABLES | View.NPY
        "view_samples": 1024,  # samples shown by View.NPY
        "debug_dir": "out/p1_debug",  # where the debug CSVs/HTML go
        "roots": {
            Dataset.ARCHIVE: "data/Tamim-data",
            Dataset.FIGSHARE: "data/FIgshare",
            Dataset.TUH: "data/GOAT/tuh_eeg/v2.0.2",  # EDFs at <root>/<edf_path without "../">
        },
        "archive": {
            "channels": 16,
            "samples": 1024,
            "sfreq": 256.0,
            # assumed 10-20 order; FZ/CZ/PZ missing
            "channel_names": ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"],
            "name_pattern": r"(\w+)_subject_(\d+)_(\d+)\.npy",  # subtype, subject, file index
        },
        "tuh": {  # one row per subject: their longest recording in a labelled session; signals loaded later (p2 workers, read_tuh)
            "db_path": "data/GOAT/DOCS/DataFeatures/tueg_headers.db",  # labels / recordings / edf_headers tables
            "electrodes": ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ", "C3", "C4", "CZ", "P3", "P4", "PZ", "O1", "O2", "T3", "T4", "T5", "T6"],  # picked in this order
            "references": ["REF", "LE"],  # EDF channel "EEG <E>-<ref>", first match wins
            "labels": {"stroke": "stroke", "healthy": "normal"},  # db label -> subtype ("normal" = p7 negative)
            "skip_missing": True,  # True: drop recordings whose EDF isn't on disk (prints counts); False: raise
        },
    },
    "p2_sample_processing": {
        "out_dirs": {  # one output folder per dataset
            Dataset.ARCHIVE: "out/p2/Archive",
            Dataset.FIGSHARE: "out/p2/FIgshare",
            Dataset.TUH: "out/p2/TUH",
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
            Dataset.TUH: "out/p3/TUH",
        },
        "debug": False,  # True: write every epoch to out_dirs as it is generated
    },
    "p4_features": {  # per-epoch features -> per-subject quantile summary
        "processes": os.cpu_count()//2,  # worker pool size (all cores; one subject per task)
        "quantiles": [10, 25, 50, 75, 90],  # percentiles of each feature across a subject's epochs
        
        
        # Every feature class has "enabled" (whole class / CSV on or off) and, where it has several, "features" (each one on or off)
        # skew, kurtosis, histogram Shannon entropy per channel (3 × channels: 48 archive, 57 TUH); entropy_bins over [-clip, clip]
        "shape": {"enabled": True, 
                  "features": {"skew": True, "kurtosis": True, "entropy": True}, 
                  "entropy_bins": 64
                  },
        
        
        # Welch PSD per channel: 4 relative band powers + 6 pairwise band ratios + DTABR, spectral mean, 1/f slope (13 × channels: 208 archive, 247 TUH); nperseg = min(win, max_nperseg_s × sfreq), floored at min_nperseg
        # slow / fast: bands summed for DTABR = (delta + theta) / (alpha + beta)
        "welch": {"enabled": False, 
                  "features": {"rel": True, "ratio": True, "dtabr": True, "spec_mean": True, "slope": True},
                  "bands": {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}, 
                  "window": "hamming", 
                  "max_nperseg_s": 2.0, 
                  "min_nperseg": 64,
                  "slow": ["delta", "theta"], 
                  "fast": ["alpha", "beta"]
                  },
        
        
        # BSI per left/right pair (skipped if a channel is missing) + their mean, from the same Welch PSD over the band range
        "asymmetry": {"enabled": True, 
                      "features": {"pairs": True, "mean": True}, 
                      "pairs": [("FP1", "FP2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]
                      },
        
        
        # regularity entropies per channel (3 × channels: 48 archive, 57 TUH): embedding m, tolerance r × SD, fuzzy exponent n; O(N²), so run on every decim-th sample (250 -> 62.5 Hz, still above the 30 Hz bandpass) in chunks of (epoch, channel) series
        "entropy": {"enabled": False, 
                    "features": {"fuzzy": True, "sample": True, "approx": True}, 
                    "m": 2, 
                    "r": 0.2, 
                    "n": 2, 
                    "decim": 4, 
                    "chunk": 32
                    },
        
        
        # Morlet time-frequency PNG per subject: freqs (Hz), cycles per wavelet, time decimation, epochs per batch, image dpi
        "wavelet": {"enabled": False, 
                    "freqs": list(range(1, 31)), 
                    "n_cycles": 6, 
                    "decim": 5, 
                    "chunk": 16, 
                    "dpi": 90
                    },
        
        
        # <out_dir>/<duration>s/<overlap>pct/: <subject>.csv (shape features), <subject>_welch.csv (Welch), <subject>_bsi.csv (asymmetry), <subject>_entropy.csv (regularity); rows = features, columns = quantiles; <subject>.png (wavelet TF)
        "out_dirs": {
            Dataset.ARCHIVE: "out/p4/Archive",
            Dataset.FIGSHARE: "out/p4/FIgshare",
            Dataset.TUH: "out/p4/TUH",
        },
        
        
        "debug": False,  # True: print the first subject's summary
    },
    "p5_selection": {  # runs inside p7, on each fold's training subjects only; considers only features switched on in p4_features
        "method": Selector.MUTUAL_INFO,  # Selector.MUTUAL_INFO | Selector.ANOVA
        "k": 30,  # keep the top k features (capped at the number available)
        "seed": 0,  # random_state for mutual information
        "debug": True,  # True: print each fold's chosen features
    },
    "p6_model": {  # runs inside p7 after p5, on each fold's chosen features; fitted on training subjects only, scores the test subjects
        "model": ModelName.XGBOOST,  # ModelName.XGBOOST; its hyperparameters are the sub-dict of the same name below
        "threshold": 0.5,  # P(stroke) >= threshold -> 1
        "debug": False,  # True: print each fold's train/test sizes and test scores
        "xgboost": {  # fixed hyperparameters (no tuning), Tamim §4.7.1 Table 4.3
            "n_estimators": 60,  # boosting rounds
            "max_depth": 2,  # depth of each tree
            "learning_rate": 0.04,  # shrinkage per round
            "reg_lambda": 20.0,  # L2 penalty on leaf weights
            "reg_alpha": 1.0,  # L1 penalty on leaf weights
            "gamma": 0.1,  # min loss reduction to split
            "subsample": 1.0,  # rows sampled per tree
            "colsample_bytree": 0.7,  # features sampled per tree
            "min_child_weight": 1,  # min hessian sum per leaf
            "scale_pos_weight": "auto",  # "auto" = neg/pos of each fold's training subjects; or a number
            "seed": 0,  # random_state
            "n_jobs": -1,  # threads (all cores)
        },
    },
    "p7_loader": {  # loads p4's CSVs from disk (so it can run without stages 1-4) and runs p5 (-> p6) in subject-level CV per (duration, overlap)
        "folds": 5,  # stratified folds over subjects (needs >= folds subjects per class)
        "repeats": 1,  # CV repeats with different shuffles (Tamim used 50)
        "seed": 0,  # shuffle seed for repeat 0 (repeat r uses seed + r)
        "negative": ["normal"],  # subtypes labelled 0; everything else is 1 (stroke). Subtype = subject_id before the last "_"
        "metrics": ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"],  # computed per fold on the test subjects
        "out_dirs": {  # p7 writes per-fold metrics and per-subject predictions here
            Dataset.ARCHIVE: "out/p7/Archive",
            Dataset.FIGSHARE: "out/p7/FIgshare",
            Dataset.TUH: "out/p7/TUH",
        },
        "debug": False,  # True: print each subject table's shape and label counts
    },
}
