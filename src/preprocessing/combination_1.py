"""
Robust EEG preprocessing with error handling
Based on CELL 1 logic from mitchb_comb1.ipynb
"""

import numpy as np
import os
from scipy import signal
import traceback
from typing import Tuple, Optional


def safe_np_load(file_path: str) -> Tuple[Optional[np.ndarray], bool]:
    """Safely load .npz files with error handling"""
    try:
        data = np.load(file_path)
        return data, True
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None, False


def preprocess_eeg_data(data: np.ndarray, fs: int = 256, window_size: float = 1, overlap: float = 0) -> np.ndarray:
    """
    Preprocess EEG data by applying segmentation and windowing
    
    Args:
        data: Input EEG data
        fs: Sampling frequency
        window_size: Window size in seconds
        overlap: Overlap ratio (0-1)
        
    Returns:
        Preprocessed EEG data
    """
    # Ensure data is 3D: (samples, channels, timepoints)
    if data.ndim == 2:
        data = data.reshape(data.shape[0], 1, data.shape[1])
    
    n_samples, n_channels, n_timepoints = data.shape
    
    # Adjust window size to not exceed signal length
    window_samples = min(int(window_size * fs), n_timepoints)
    step_samples = int(window_samples * (1 - overlap))
    
    print(f"Original data shape: {data.shape}")
    print(f"Window size: {window_size}s ({window_samples} samples)")
    print(f"Step size: {step_samples} samples")
    print(f"Overlap: {overlap*100}%")
    
    # Calculate number of windows per sample
    n_windows_per_sample = max(1, (n_timepoints - window_samples) // step_samples + 1)
    total_windows = n_samples * n_windows_per_sample
    
    # Initialize output array
    processed_data = np.zeros((total_windows, n_channels, window_samples))
    
    print(f"Creating {n_windows_per_sample} windows per sample")
    print(f"Total windows: {total_windows}")
    
    # Apply windowing to each sample
    window_idx = 0
    for sample_idx in range(n_samples):
        for window_start in range(0, n_timepoints - window_samples + 1, step_samples):
            window_end = window_start + window_samples
            processed_data[window_idx] = data[sample_idx, :, window_start:window_end]
            window_idx += 1
    
    print(f"Final processed data shape: {processed_data.shape}")
    return processed_data


def preprocess_single_file(file_path: str, window_size: float = 1, overlap: float = 0) -> bool:
    """
    Preprocess a single file with comprehensive error handling
    
    Args:
        file_path: Path to the input file
        window_size: Window size in seconds
        overlap: Overlap ratio
        
    Returns:
        Success status
    """
    print(f"\n{'='*50}")
    print(f"Preprocessing: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    # Try to load the file
    original_data, success = safe_np_load(file_path)
    if not success:
        return False
    
    keys = list(original_data.keys())
    print(f"Original keys: {keys}")
    
    # Identify data and label keys
    data_key = None
    label_key = None
    
    # Common key patterns
    data_patterns = ['x', 'train_signals', 'signals', 'data', 'eeg', 'X']
    label_patterns = ['y', 'train_labels', 'labels', 'target', 'Y']
    
    for key in keys:
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in data_patterns):
            data_key = key
        elif any(pattern in key_lower for pattern in label_patterns):
            label_key = key
    
    if data_key is None:
        data_key = keys[0]  # Use first key as fallback
    
    try:
        # Extract data
        eeg_data = original_data[data_key]
        print(f"Original EEG data shape: {eeg_data.shape}")
        
        # Extract labels if available
        if label_key and label_key in keys:
            original_labels = original_data[label_key]
            print(f"Original labels shape: {original_labels.shape}")
        else:
            original_labels = None
            print("No labels found in file")
        
        # Apply preprocessing
        processed_eeg = preprocess_eeg_data(eeg_data, window_size=window_size, overlap=overlap)
        
        # Handle labels
        if original_labels is not None:
            n_windows_per_sample = processed_eeg.shape[0] // eeg_data.shape[0]
            processed_labels = np.repeat(original_labels, n_windows_per_sample)
            print(f"Processed labels shape: {processed_labels.shape}")
        else:
            processed_labels = None
        
        # Save processed data
        output_path = file_path.replace('.npz', '_preprocessed.npz')
        save_dict = {'x': processed_eeg}
        if processed_labels is not None:
            save_dict['y'] = processed_labels
        
        # Add metadata
        save_dict['feature_names'] = [f'channel_{i}' for i in range(processed_eeg.shape[1])]
        save_dict['preprocessing_info'] = f'window_{window_size}s_overlap_{overlap}'
        
        np.savez(output_path, **save_dict)
        print(f"✅ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        traceback.print_exc()
        return False


def check_file_sizes(file_list: list, base_path: str = "") -> list:
    """Check if files exist and their sizes"""
    print("Checking files...")
    valid_files = []
    
    for file_name in file_list:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  ✓ {file_name} ({file_size:.1f} MB)")
            valid_files.append(file_name)
        else:
            print(f"  ✗ {file_name} (File not found)")
    
    return valid_files


def batch_preprocess_files(file_list: list, base_path: str = "", window_size: float = 1, overlap: float = 0) -> None:
    """
    Batch preprocess multiple files
    
    Args:
        file_list: List of file names to process
        base_path: Base directory path
        window_size: Window size in seconds
        overlap: Overlap ratio
    """
    print("Step 1: Checking files...")
    valid_files = check_file_sizes(file_list, base_path)
    
    print(f"\nStep 2: Preprocessing {len(valid_files)} valid files...")
    print(f"Using {window_size}-second windows ({overlap*100}% overlap)\n")
    
    success_count = 0
    for file_name in valid_files:
        file_path = os.path.join(base_path, file_name)
        if preprocess_single_file(file_path, window_size=window_size, overlap=overlap):
            success_count += 1
        print()  # Empty line for readability
    
    print(f"\nPreprocessing Summary:")
    print(f"Successfully processed: {success_count}/{len(valid_files)} files")
    print(f"Failed: {len(valid_files) - success_count} files")
    
    if success_count > 0:
        print("\nGenerated files:")
        for file_name in valid_files:
            preprocessed_name = file_name.replace('.npz', '_preprocessed.npz')
            if os.path.exists(os.path.join(base_path, preprocessed_name)):
                print(f"  ✓ {preprocessed_name}")


def main():
    """Main function for standalone execution"""
    # Example usage
    all_original_files = [
        "eeg-seizure_train.npz",
        "eeg-seizure_val.npz", 
        "eeg-seizure_test.npz",
        "eeg-seizure_val_balanced.npz",
        "eeg-predictive_train.npz",
        "eeg-predictive_val.npz",
        "eeg-predictive_val_balanced.npz"
    ]
    
    batch_preprocess_files(all_original_files, base_path="./", window_size=1, overlap=0)


if __name__ == "__main__":
    main()
