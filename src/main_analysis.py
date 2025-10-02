#!/usr/bin/env python3
"""
Advanced EEG Seizure Detection Analysis
Comprehensive pipeline for EEG feature extraction, selection, and classification
Based on the sophisticated logic from mitchb_comb1.ipynb
"""

import numpy as np
import os
import sys
from tqdm import tqdm
import warnings
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scientific computing imports
from scipy import signal, stats
from scipy.fftpack import fft
from scipy.linalg import eigh
import pywt

# Machine learning imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

warnings.filterwarnings('ignore')

class RobustEEGPreprocessor:
    """
    Robust EEG preprocessing with comprehensive error handling
    Based on CELL 1 logic from the notebook
    """
    
    def __init__(self, window_size: float = 1.0, overlap: float = 0.0, fs: int = 256):
        self.window_size = window_size
        self.overlap = overlap
        self.fs = fs
    
    def safe_np_load(self, file_path: str) -> Tuple[Optional[np.ndarray], bool]:
        """Safely load .npz files with error handling"""
        try:
            data = np.load(file_path)
            return data, True
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return None, False
    
    def preprocess_eeg_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data by applying segmentation and windowing
        """
        # Ensure data is 3D: (samples, channels, timepoints)
        if data.ndim == 2:
            data = data.reshape(data.shape[0], 1, data.shape[1])
        
        n_samples, n_channels, n_timepoints = data.shape
        
        # Adjust window size to not exceed signal length
        window_samples = min(int(self.window_size * self.fs), n_timepoints)
        step_samples = int(window_samples * (1 - self.overlap))
        
        print(f"Original data shape: {data.shape}")
        print(f"Window size: {self.window_size}s ({window_samples} samples)")
        print(f"Step size: {step_samples} samples")
        print(f"Overlap: {self.overlap*100}%")
        
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
    
    def preprocess_single_file(self, file_path: str) -> bool:
        """
        Preprocess a single file with comprehensive error handling
        """
        print(f"\n{'='*50}")
        print(f"Preprocessing: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        # Try to load the file
        original_data, success = self.safe_np_load(file_path)
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
            processed_eeg = self.preprocess_eeg_data(eeg_data)
            
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
            save_dict['preprocessing_info'] = f'window_{self.window_size}s_overlap_{self.overlap}'
            
            np.savez(output_path, **save_dict)
            print(f"✅ Saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False


class ComprehensiveFeatureExtractor:
    """
    Comprehensive feature extraction including time, frequency, and wavelet domains
    Based on CELLS 3, 4, and 5 logic from the notebook
    """
    
    def __init__(self, fs: int = 256):
        self.fs = fs
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        self.wavelet = 'db4'
        self.level = 4
    
    def extract_time_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract time domain features for all channels"""
        features = {}
        
        for i in range(data.shape[0]):
            channel_data = data[i]
            
            # Basic time domain features
            features[f'ch{i}_variance'] = np.var(channel_data)
            features[f'ch{i}_rms'] = np.sqrt(np.mean(channel_data**2))
            features[f'ch{i}_skewness'] = stats.skew(channel_data)
            features[f'ch{i}_kurtosis'] = stats.kurtosis(channel_data)
            
            # Additional features from notebook
            features[f'ch{i}_mean_amp'] = np.mean(np.abs(channel_data))
            features[f'ch{i}_line_length'] = np.sum(np.abs(np.diff(channel_data)))
            
            # Zero-crossing rate
            zero_crossings = np.where(np.diff(np.sign(channel_data)))[0]
            features[f'ch{i}_zcr'] = len(zero_crossings) / len(channel_data)
        
        return features
    
    def extract_frequency_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using Welch's method"""
        features = {}
        
        for i in range(data.shape[0]):
            channel_data = data[i]
            
            # Use appropriate parameters for short signals
            nperseg = min(64, len(channel_data))
            noverlap = min(32, nperseg - 1)
            
            try:
                freqs, psd = signal.welch(channel_data, fs=self.fs, nperseg=nperseg, 
                                         noverlap=noverlap, window='hann')
                
                total_power = np.trapz(psd, freqs)
                
                # Band power features
                for band, (low, high) in self.freq_bands.items():
                    idx_band = np.logical_and(freqs >= low, freqs <= high)
                    if np.any(idx_band):
                        band_power = np.trapz(psd[idx_band], freqs[idx_band])
                        features[f'ch{i}_{band}_abs_power'] = band_power
                        features[f'ch{i}_{band}_rel_power'] = band_power / total_power if total_power > 0 else 0
                    else:
                        features[f'ch{i}_{band}_abs_power'] = 0
                        features[f'ch{i}_{band}_rel_power'] = 0
                
                # Spectral edge frequency (SEF95)
                if len(psd) > 0:
                    cum_power = np.cumsum(psd)
                    cum_power_norm = cum_power / cum_power[-1]
                    sef95_idx = np.where(cum_power_norm >= 0.95)[0]
                    features[f'ch{i}_sef95'] = freqs[sef95_idx[0]] if len(sef95_idx) > 0 else 0
                else:
                    features[f'ch{i}_sef95'] = 0
                    
                # Additional frequency features from notebook
                if total_power > 0:
                    features[f'ch{i}_mean_freq'] = np.sum(freqs * psd) / total_power
                    peak_idx = np.argmax(psd)
                    features[f'ch{i}_peak_freq'] = freqs[peak_idx]
                    mean_freq = features[f'ch{i}_mean_freq']
                    features[f'ch{i}_bandwidth'] = np.sqrt(np.sum(psd * (freqs - mean_freq)**2) / total_power)
                else:
                    features[f'ch{i}_mean_freq'] = 0
                    features[f'ch{i}_peak_freq'] = 0
                    features[f'ch{i}_bandwidth'] = 0
                    
            except Exception as e:
                print(f"Error extracting frequency features for channel {i}: {e}")
                # Set default values
                for band in self.freq_bands.keys():
                    features[f'ch{i}_{band}_abs_power'] = 0
                    features[f'ch{i}_{band}_rel_power'] = 0
                features[f'ch{i}_sef95'] = 0
                features[f'ch{i}_mean_freq'] = 0
                features[f'ch{i}_peak_freq'] = 0
                features[f'ch{i}_bandwidth'] = 0
        
        return features
    
    def extract_wavelet_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract wavelet domain features using DWT"""
        features = {}
        
        for i in range(data.shape[0]):
            channel_data = data[i]
            
            max_level = pywt.dwt_max_level(len(channel_data), self.wavelet)
            actual_level = min(self.level, max_level)
            
            try:
                coeffs = pywt.wavedec(channel_data, self.wavelet, level=actual_level)
                energies = [np.sum(c**2) for c in coeffs]
                total_energy = np.sum(energies)
                
                for j, coeff in enumerate(coeffs):
                    features[f'ch{i}_w{j}_log_var'] = np.log(np.var(coeff) + 1e-10)
                    features[f'ch{i}_w{j}_rel_energy'] = energies[j] / total_energy if total_energy > 0 else 0
                    features[f'ch{i}_w{j}_std'] = np.std(coeff)
                    
                    # Additional wavelet features from notebook
                    if len(coeff) > 0:
                        features[f'ch{i}_w{j}_mean_abs'] = np.mean(np.abs(coeff))
                        features[f'ch{i}_w{j}_avg_power'] = np.mean(coeff**2)
                        if np.min(coeff) != 0:
                            features[f'ch{i}_w{j}_max_min_ratio'] = np.max(coeff) / np.min(coeff)
                        else:
                            features[f'ch{i}_w{j}_max_min_ratio'] = np.max(coeff) / (np.min(coeff) + 1e-10)
                    else:
                        features[f'ch{i}_w{j}_mean_abs'] = 0
                        features[f'ch{i}_w{j}_avg_power'] = 0
                        features[f'ch{i}_w{j}_max_min_ratio'] = 0
                        
            except Exception as e:
                print(f"Error extracting wavelet features for channel {i}: {e}")
                for j in range(actual_level + 1):
                    features[f'ch{i}_w{j}_log_var'] = 0
                    features[f'ch{i}_w{j}_rel_energy'] = 0
                    features[f'ch{i}_w{j}_std'] = 0
                    features[f'ch{i}_w{j}_mean_abs'] = 0
                    features[f'ch{i}_w{j}_avg_power'] = 0
                    features[f'ch{i}_w{j}_max_min_ratio'] = 0
        
        return features
    
    def extract_csp_features(self, data: np.ndarray, labels: np.ndarray, n_components: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Common Spatial Patterns features (supervised method)
        Returns CSP filters and projected features
        """
        if labels is None or len(np.unique(labels)) < 2:
            raise ValueError("CSP requires labeled data with at least 2 classes")
        
        # Ensure data is properly shaped: (n_samples, n_channels, n_timesteps)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        
        n_samples, n_channels, n_timesteps = data.shape
        
        # Separate data by class
        class_0_idx = np.where(labels == 0)[0]
        class_1_idx = np.where(labels == 1)[0]
        
        if len(class_0_idx) == 0 or len(class_1_idx) == 0:
            raise ValueError("Both classes must have samples for CSP")
        
        data_0 = data[class_0_idx]
        data_1 = data[class_1_idx]
        
        # Calculate covariance matrices for each class
        cov_0 = np.zeros((n_channels, n_channels))
        cov_1 = np.zeros((n_channels, n_channels))
        
        for i in range(data_0.shape[0]):
            cov_0 += np.cov(data_0[i])
        cov_0 /= data_0.shape[0]
        
        for i in range(data_1.shape[0]):
            cov_1 += np.cov(data_1[i])
        cov_1 /= data_1.shape[0]
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(cov_0, cov_0 + cov_1)
        
        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select first and last n_components
        selected_filters = np.hstack([eigenvectors[:, :n_components//2], 
                                     eigenvectors[:, -n_components//2:]])
        
        # Apply CSP filters to all data
        csp_features = []
        for i in range(data.shape[0]):
            projected_data = np.dot(selected_filters.T, data[i])
            # Extract log-variance of CSP components
            log_vars = np.log(np.var(projected_data, axis=1) + 1e-10)
            csp_features.append(log_vars)
        
        return np.array(csp_features), selected_filters
    
    def extract_all_features(self, data: np.ndarray, labels: Optional[np.ndarray] = None, csp_filters: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract all features from all domains"""
        features = {}
        
        # Time domain features
        time_features = self.extract_time_features(data)
        features.update(time_features)
        
        # Frequency domain features
        freq_features = self.extract_frequency_features(data)
        features.update(freq_features)
        
        # Wavelet domain features
        wavelet_features = self.extract_wavelet_features(data)
        features.update(wavelet_features)
        
        # CSP features if available
        try:
            if csp_filters is not None:
                projected_data = np.dot(csp_filters.T, data)
                csp_log_vars = np.log(np.var(projected_data, axis=1) + 1e-10)
                for comp_idx, log_var in enumerate(csp_log_vars):
                    features[f'csp{comp_idx}_log_var'] = log_var
        except Exception as e:
            print(f"Error in CSP feature extraction: {e}")
        
        return features


class AdvancedFeatureSelector:
    """
    Advanced feature selection using mutual information and sequential selection
    Based on CELL 6 logic from the notebook
    """
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def load_and_combine_features(self, files: List[str], base_path: str = "") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load features from multiple files and combine them"""
        X_list, y_list, names_list = [], [], []

        for file in files:
            full_path = os.path.join(base_path, file)
            if not os.path.exists(full_path):
                print(f"⚠️ File not found: {full_path}")
                continue
                
            data = np.load(full_path)
            keys = data.files

            # Extract data using standard keys
            feat_key = 'x' if 'x' in keys else keys[0]
            label_key = 'y' if 'y' in keys else None
            names_key = 'feature_names' if 'feature_names' in keys else None

            X_data = data[feat_key]
            
            # Check for NaN values
            nan_count = np.isnan(X_data).sum()
            if nan_count > 0:
                print(f"⚠️ {file} contains {nan_count} NaN values")
            
            X_list.append(X_data)
            
            if label_key and label_key in keys:
                y_list.append(data[label_key])
            else:
                print(f"⚠️ No labels found in {file}, using zeros")
                y_list.append(np.zeros(X_data.shape[0]))
                
            if names_key and names_key in keys:
                names_list.append(list(data[names_key]))
            else:
                num_features = X_data.shape[1]
                names_list.append([f'feature_{i}' for i in range(num_features)])

        # Find common number of samples across all files
        min_samples = min(X.shape[0] for X in X_list)
        
        # Truncate all arrays to the minimum number of samples
        X_truncated = [X[:min_samples] for X in X_list]
        y_truncated = [y[:min_samples] for y in y_list]
        
        # Concatenate features along columns
        X_combined = np.concatenate(X_truncated, axis=1)
        y_combined = y_truncated[0]  # Use labels from first file
        
        # Combine feature names
        all_names = []
        for names in names_list:
            all_names.extend(names)
        
        return X_combined, y_combined, all_names
    
    def clean_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                      nan_threshold: float = 0.1, constant_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Remove features with too many NaN values and constant features"""
        
        # Step 1: Remove features with too many NaN values
        nan_ratio = np.isnan(X).mean(axis=0)
        bad_nan_features = nan_ratio > nan_threshold
        good_features = ~bad_nan_features
        
        if bad_nan_features.sum() > 0:
            print(f"Removing {bad_nan_features.sum()} features with >{nan_threshold*100}% NaN values")
            X_clean = X[:, good_features]
            feature_names_clean = [feature_names[i] for i in range(len(feature_names)) if good_features[i]]
        else:
            X_clean = X
            feature_names_clean = feature_names
        
        # Step 2: Remove samples with any remaining NaN values
        nan_samples = np.isnan(X_clean).any(axis=1)
        if nan_samples.sum() > 0:
            print(f"Removing {nan_samples.sum()} samples with NaN values")
            X_clean = X_clean[~nan_samples]
            y_clean = y[~nan_samples]
        else:
            y_clean = y
        
        # Step 3: Remove constant features
        feature_stds = np.nanstd(X_clean, axis=0)
        constant_features = feature_stds < constant_threshold
        
        if constant_features.sum() > 0:
            print(f"Removing {constant_features.sum()} constant features")
            non_constant_features = ~constant_features
            X_clean = X_clean[:, non_constant_features]
            feature_names_clean = [feature_names_clean[i] for i in range(len(feature_names_clean)) if non_constant_features[i]]
        
        return X_clean, y_clean, feature_names_clean
    
    def robust_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                                n_features: int = 20) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Perform feature selection with robust handling of data issues"""
        
        # Handle any remaining NaN values with imputation
        if np.isnan(X).any():
            print("Imputing remaining NaN values...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use mutual information for initial feature selection
        print("Calculating mutual information scores...")
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42, n_neighbors=5)
        
        # Select top features based on mutual information
        n_pre_select = min(50, X_scaled.shape[1])
        top_indices = np.argsort(mi_scores)[-n_pre_select:][::-1]
        
        X_reduced = X_scaled[:, top_indices]
        feature_names_reduced = [feature_names[i] for i in top_indices]
        mi_scores_reduced = mi_scores[top_indices]
        
        print(f"Pre-selected {n_pre_select} features using mutual information")
        
        return X_reduced, feature_names_reduced, mi_scores_reduced
    
    def sequential_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                                   n_features_to_select: int = 15) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Perform sequential feature selection"""
        
        print(f"Performing sequential feature selection on {X.shape[1]} features...")
        print(f"Target: select {n_features_to_select} best features")
        
        try:
            sfs = SequentialFeatureSelector(
                self.estimator,
                n_features_to_select=n_features_to_select,
                direction='forward',
                scoring='accuracy',
                cv=3,
                n_jobs=-1
            )
            
            sfs.fit(X, y)
            selected_indices = sfs.get_support()
            selected_features = np.array(feature_names)[selected_indices]
            
            return X[:, selected_indices], selected_features, selected_indices
            
        except Exception as e:
            print(f"❌ Error in sequential feature selection: {e}")
            print("Using mutual information ranking instead...")
            
            # Fallback: use mutual information ranking
            mi_scores = mutual_info_classif(X, y, random_state=42)
            top_indices = np.argsort(mi_scores)[-n_features_to_select:][::-1]
            selected_features = np.array(feature_names)[top_indices]
            
            return X[:, top_indices], selected_features, top_indices


def main():
    """Main execution function"""
    print("🧠 Advanced EEG Seizure Detection Analysis")
    print("=" * 50)
    
    base_path = "./"
    
    # Initialize components
    preprocessor = RobustEEGPreprocessor()
    feature_extractor = ComprehensiveFeatureExtractor()
    feature_selector = AdvancedFeatureSelector()
    
    # List all feature files
    all_files = [f for f in os.listdir(base_path) 
                 if f.endswith('.npz') and any(prefix in f for prefix in ['features_', 'simple_features_', 'discriminant_features_'])]
    
    print("Available feature files:")
    for f in all_files:
        print(f"  - {f}")
    
    # Group by dataset type
    seizure_files = [f for f in all_files if 'seizure' in f and ('train' in f or 'val' in f)]
    
    print(f"\nSeizure files: {len(seizure_files)} files")
    
    if seizure_files:
        # Load and combine features
        print("\n📊 Loading and combining features...")
        X, y, feature_names = feature_selector.load_and_combine_features(seizure_files, base_path)
        print(f"Raw features: {X.shape}, labels: {y.shape}")
        
        # Clean features
        X, y, feature_names = feature_selector.clean_features(X, y, feature_names)
        print(f"Clean features: {X.shape}, labels: {y.shape}")
        
        # Robust feature selection
        print("\n🔍 Performing robust feature selection...")
        X_reduced, feature_names_reduced, mi_scores = feature_selector.robust_feature_selection(X, y, feature_names)
        
        # Sequential feature selection
        n_features_to_select = min(15, X_reduced.shape[1])
        X_final, selected_features, selected_indices = feature_selector.sequential_feature_selection(
            X_reduced, y, feature_names_reduced, n_features_to_select)
        
        # Get the mutual information scores for selected features
        selected_mi_scores = mi_scores[selected_indices]
        
        print(f"\n✅ Selected {len(selected_features)} features:")
        for i, (feature, mi_score) in enumerate(zip(selected_features, selected_mi_scores)):
            print(f"  {i+1:2d}. {feature} (MI: {mi_score:.4f})")
        
        # Evaluate performance
        print("\n📊 Final Model Performance:")
        estimator = KNeighborsClassifier(n_neighbors=5)
        final_scores = cross_val_score(estimator, X_final, y, cv=5)
        print(f"Cross-validated accuracy: {final_scores.mean():.3f} (+/- {final_scores.std() * 2:.3f})")
        print(f"Individual fold scores: {[f'{score:.3f}' for score in final_scores]}")
        
        # Feature type analysis
        print(f"\n🔍 Feature Type Analysis:")
        feature_types = {}
        for feature in selected_features:
            if 'ch' in feature and 'w' in feature:
                feature_types['Wavelet'] = feature_types.get('Wavelet', 0) + 1
            elif 'csp' in feature:
                feature_types['CSP'] = feature_types.get('CSP', 0) + 1
            elif any(x in feature for x in ['mean_amp', 'line_length', 'zcr']):
                feature_types['Time Domain'] = feature_types.get('Time Domain', 0) + 1
            elif any(x in feature for x in ['mean_freq', 'peak_freq', 'bandwidth']):
                feature_types['Frequency Domain'] = feature_types.get('Frequency Domain', 0) + 1
            elif 'power' in feature:
                feature_types['Power'] = feature_types.get('Power', 0) + 1
            else:
                feature_types['Other'] = feature_types.get('Other', 0) + 1
        
        for feature_type, count in feature_types.items():
            print(f"  {feature_type}: {count} features")
    
    else:
        print("❌ No seizure feature files found!")


if __name__ == "__main__":
    main()
