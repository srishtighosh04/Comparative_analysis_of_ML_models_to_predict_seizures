"""
Configuration file for EEG Seizure Detection Project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MIT_CHB_DIR = os.path.join(PROCESSED_DATA_DIR, "mit_chb")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
PAPERS_DIR = os.path.join(DOCS_DIR, "papers")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Dataset files
SEIZURE_DATASETS = {
    "train": "eeg-seizure_train.npz",
    "test": "eeg-seizure_test.npz", 
    "val": "eeg-seizure_val.npz",
    "val_balanced": "eeg-seizure_val_balanced.npz"
}

PREDICTIVE_DATASETS = {
    "train": "eeg-predictive_train.npz",
    "val": "eeg-predictive_val.npz",
    "val_balanced": "eeg-predictive_val_balanced.npz"
}

# Legacy dataset configuration (for reference)
DATASET_FOLDERS = {
    "F": 0,  # Seizure-free (opposite hemisphere)
    "N": 1,  # Seizure-free (epileptic patients)
    "O": 2,  # Healthy eyes-closed (Set B)
    "Z": 3,  # Healthy eyes-open (Set A)
    "S": 4   # Seizure activity (Set E)
}

# Legacy classification groups (for reference)
CLASSIFICATION_GROUPS = {
    "A-E": ("Z", "S"),  # Healthy eyes-open vs. Epileptic
    "B-E": ("O", "S"),  # Healthy eyes-closed vs. Epileptic
    "C-E": ("N", "S"),  # Seizure-controlled vs. Epileptic
    "D-E": ("F", "S"),  # Seizure-controlled (opposite hemisphere) vs. Epileptic
    "ABCD-E": ("Z", "O", "N", "F", "S"),  # All Healthy and Seizure-free vs. Epileptic
    "AB-CD": ("Z", "O", "N", "F")  # Healthy vs. Seizure-free
}

# Signal processing parameters
SAMPLING_RATE = 256  # Hz
FILTER_PARAMS = {
    "lowcut": 1,
    "highcut": 70,
    "order": 5
}

# Frequency bands
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 70)
}

# Wavelet parameters
WAVELET_TYPE = 'db6'
WAVELET_LEVELS = 4

# Machine learning parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
KNN_NEIGHBORS = 3

# Visualization settings
FIGURE_SIZE = (8, 6)
HEATMAP_COLORS = 'Blues'
