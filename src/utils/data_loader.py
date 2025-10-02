"""
Data loading utilities for EEG datasets
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import (RAW_DATA_DIR, MIT_CHB_DIR, SEIZURE_DATASETS, 
                          PREDICTIVE_DATASETS, DATASET_FOLDERS)


def load_npz_dataset(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load EEG dataset from .npz file
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        Tuple of (data, labels) or (None, None) if error
    """
    try:
        dataset = np.load(file_path)
        
        # Try common key names for data
        data_keys = ['x', 'X', 'data', 'signals', 'eeg_data']
        label_keys = ['y', 'Y', 'labels', 'targets', 'classes']
        
        data, labels = None, None
        
        # Find data
        for key in data_keys:
            if key in dataset.keys():
                data = dataset[key]
                break
        
        # Find labels
        for key in label_keys:
            if key in dataset.keys():
                labels = dataset[key]
                break
        
        if data is None:
            # Use first key as data if no standard key found
            keys = list(dataset.keys())
            if keys:
                data = dataset[keys[0]]
                print(f"Using '{keys[0]}' as data key")
        
        print(f"Loaded dataset from {os.path.basename(file_path)}")
        if data is not None:
            print(f"  Data shape: {data.shape}")
        if labels is not None:
            print(f"  Labels shape: {labels.shape}")
            print(f"  Unique labels: {np.unique(labels)}")
        
        return data, labels
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def load_seizure_dataset(dataset_type: str = "train") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load seizure detection dataset
    
    Args:
        dataset_type: Type of dataset ("train", "test", "val", "val_balanced")
        
    Returns:
        Tuple of (data, labels)
    """
    if dataset_type not in SEIZURE_DATASETS:
        print(f"Unknown dataset type: {dataset_type}")
        print(f"Available types: {list(SEIZURE_DATASETS.keys())}")
        return None, None
    
    file_name = SEIZURE_DATASETS[dataset_type]
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    
    return load_npz_dataset(file_path)


def load_predictive_dataset(dataset_type: str = "train") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load seizure prediction dataset
    
    Args:
        dataset_type: Type of dataset ("train", "val", "val_balanced")
        
    Returns:
        Tuple of (data, labels)
    """
    if dataset_type not in PREDICTIVE_DATASETS:
        print(f"Unknown dataset type: {dataset_type}")
        print(f"Available types: {list(PREDICTIVE_DATASETS.keys())}")
        return None, None
    
    file_name = PREDICTIVE_DATASETS[dataset_type]
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    
    return load_npz_dataset(file_path)


def load_mit_chb_processed(file_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load processed MIT-CHB dataset
    
    Args:
        file_name: Name of the .npz file in mit_chb directory
        
    Returns:
        Tuple of (data, labels)
    """
    file_path = os.path.join(MIT_CHB_DIR, file_name)
    return load_npz_dataset(file_path)


def list_available_datasets() -> Dict[str, List[str]]:
    """
    List all available datasets
    
    Returns:
        Dictionary with dataset categories and their files
    """
    available = {
        "seizure_detection": [],
        "seizure_prediction": [],
        "mit_chb_processed": []
    }
    
    # Check seizure detection datasets
    for dataset_type, file_name in SEIZURE_DATASETS.items():
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if os.path.exists(file_path):
            available["seizure_detection"].append(f"{dataset_type} ({file_name})")
    
    # Check predictive datasets
    for dataset_type, file_name in PREDICTIVE_DATASETS.items():
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if os.path.exists(file_path):
            available["seizure_prediction"].append(f"{dataset_type} ({file_name})")
    
    # Check MIT-CHB processed datasets
    if os.path.exists(MIT_CHB_DIR):
        mit_files = [f for f in os.listdir(MIT_CHB_DIR) if f.endswith('.npz')]
        available["mit_chb_processed"] = mit_files
    
    return available


def get_dataset_info(file_path: str) -> Dict[str, Union[str, int, Tuple]]:
    """
    Get information about a dataset file
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Dictionary with dataset information
    """
    try:
        dataset = np.load(file_path)
        info = {
            "file_path": file_path,
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "keys": list(dataset.keys()),
        }
        
        # Get shape information for each key
        for key in dataset.keys():
            info[f"{key}_shape"] = dataset[key].shape
            info[f"{key}_dtype"] = str(dataset[key].dtype)
        
        return info
        
    except Exception as e:
        return {"error": str(e)}


# Legacy functions for backward compatibility
def load_eeg_signal(file_path: str) -> np.ndarray:
    """Legacy function - loads text files (for backward compatibility)"""
    try:
        signal = np.loadtxt(file_path)
        return signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])


def load_classification_data(group_classes: Tuple[str, ...]) -> Tuple[List[np.ndarray], List[int]]:
    """Legacy function for text-based datasets (for backward compatibility)"""
    print("Warning: This function is for legacy text-based datasets.")
    print("Consider using load_seizure_dataset() or load_predictive_dataset() instead.")
    return [], []
