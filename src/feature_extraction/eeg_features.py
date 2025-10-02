"""
Comprehensive EEG feature extraction utilities
Based on CELLS 3, 4, and 5 logic from mitchb_comb1.ipynb
"""

import numpy as np
import pywt
from scipy import signal, stats
from scipy.fftpack import fft
from scipy.linalg import eigh
from typing import List, Tuple, Dict, Optional
import os
import sys
from tqdm import tqdm
import warnings

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import SAMPLING_RATE, FILTER_PARAMS, FREQUENCY_BANDS, WAVELET_TYPE, WAVELET_LEVELS

warnings.filterwarnings('ignore')


# Define frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Define wavelet parameters
WAVELET = 'db4'
LEVEL = 4
MAX_LEVEL = 3  # For WPD

# CSP frequency bands
FREQ_BANDS_CSP = {
    'beta': (13, 30),
    'gamma': (30, 45)
}


def extract_time_features_simple(data: np.ndarray) -> Dict[str, float]:
    """Extract simple time domain features for all channels"""
    features = {}
    
    for i in range(data.shape[0]):
        channel_data = data[i]
        
        # Mean Amplitude
        features[f'ch{i}_mean_amp'] = np.mean(np.abs(channel_data))
        
        # Line Length (sum of absolute differences between consecutive samples)
        features[f'ch{i}_line_length'] = np.sum(np.abs(np.diff(channel_data)))
        
        # Zero-Crossing Rate
        zero_crossings = np.where(np.diff(np.sign(channel_data)))[0]
        features[f'ch{i}_zcr'] = len(zero_crossings) / len(channel_data)
    
    return features


def extract_time_features(data: np.ndarray) -> Dict[str, float]:
    """Extract comprehensive time domain features for all channels"""
    features = {}
    
    for i in range(data.shape[0]):
        channel_data = data[i]
        
        # Basic statistical features
        features[f'ch{i}_variance'] = np.var(channel_data)
        features[f'ch{i}_rms'] = np.sqrt(np.mean(channel_data**2))
        features[f'ch{i}_skewness'] = stats.skew(channel_data)
        features[f'ch{i}_kurtosis'] = stats.kurtosis(channel_data)
        
        # Additional time domain features
        features[f'ch{i}_mean_amp'] = np.mean(np.abs(channel_data))
        features[f'ch{i}_line_length'] = np.sum(np.abs(np.diff(channel_data)))
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.sign(channel_data)))[0]
        features[f'ch{i}_zcr'] = len(zero_crossings) / len(channel_data)
    
    return features


def extract_frequency_features_simple(data: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """Extract simple frequency domain features using periodogram"""
    features = {}
    
    for i in range(data.shape[0]):
        channel_data = data[i]
        
        # Compute periodogram (simple PSD estimate)
        freqs, psd = signal.periodogram(channel_data, fs=fs, window='hann')
        
        if len(psd) > 0:
            # Mean Frequency (weighted average)
            total_power = np.sum(psd)
            if total_power > 0:
                features[f'ch{i}_mean_freq'] = np.sum(freqs * psd) / total_power
            else:
                features[f'ch{i}_mean_freq'] = 0
            
            # Peak Frequency (frequency with maximum power)
            peak_idx = np.argmax(psd)
            features[f'ch{i}_peak_freq'] = freqs[peak_idx]
            
            # Bandwidth (spread around mean frequency)
            if total_power > 0:
                mean_freq = features[f'ch{i}_mean_freq']
                bandwidth = np.sqrt(np.sum(psd * (freqs - mean_freq)**2) / total_power)
                features[f'ch{i}_bandwidth'] = bandwidth
            else:
                features[f'ch{i}_bandwidth'] = 0
        else:
            features[f'ch{i}_mean_freq'] = 0
            features[f'ch{i}_peak_freq'] = 0
            features[f'ch{i}_bandwidth'] = 0
    
    return features


def extract_frequency_features(data: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """Extract frequency domain features using Welch's method"""
    features = {}
    
    for i in range(data.shape[0]):
        channel_data = data[i]
        
        # Use appropriate parameters for short signals
        nperseg = min(64, len(channel_data))
        noverlap = min(32, nperseg - 1)
        
        freqs, psd = signal.welch(channel_data, fs=fs, nperseg=nperseg, 
                                 noverlap=noverlap, window='hann')
        
        total_power = np.trapz(psd, freqs)
        
        for band, (low, high) in FREQ_BANDS.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx_band):
                band_power = np.trapz(psd[idx_band], freqs[idx_band])
                features[f'ch{i}_{band}_abs_power'] = band_power
                features[f'ch{i}_{band}_rel_power'] = band_power / total_power if total_power > 0 else 0
            else:
                features[f'ch{i}_{band}_abs_power'] = 0
                features[f'ch{i}_{band}_rel_power'] = 0
        
        if len(psd) > 0:
            cum_power = np.cumsum(psd)
            cum_power_norm = cum_power / cum_power[-1]
            sef95_idx = np.where(cum_power_norm >= 0.95)[0]
            features[f'ch{i}_sef95'] = freqs[sef95_idx[0]] if len(sef95_idx) > 0 else 0
        else:
            features[f'ch{i}_sef95'] = 0
    
    return features


def extract_wavelet_features_simple(data: np.ndarray) -> Dict[str, float]:
    """Extract simple wavelet domain features using DWT"""
    features = {}
    
    for i in range(data.shape[0]):
        channel_data = data[i]
        
        max_level = pywt.dwt_max_level(len(channel_data), WAVELET)
        actual_level = min(LEVEL, max_level)
        
        try:
            coeffs = pywt.wavedec(channel_data, WAVELET, level=actual_level)
            
            for j, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    # Mean of absolute values
                    features[f'ch{i}_w{j}_mean_abs'] = np.mean(np.abs(coeff))
                    
                    # Average Power
                    features[f'ch{i}_w{j}_avg_power'] = np.mean(coeff**2)
                    
                    # Ratio of Maximum to Minimum
                    if np.min(coeff) != 0:  # Avoid division by zero
                        features[f'ch{i}_w{j}_max_min_ratio'] = np.max(coeff) / np.min(coeff)
                    else:
                        features[f'ch{i}_w{j}_max_min_ratio'] = np.max(coeff) / (np.min(coeff) + 1e-10)
                else:
                    features[f'ch{i}_w{j}_mean_abs'] = 0
                    features[f'ch{i}_w{j}_avg_power'] = 0
                    features[f'ch{i}_w{j}_max_min_ratio'] = 0
        except:
            for j in range(actual_level + 1):
                features[f'ch{i}_w{j}_mean_abs'] = 0
                features[f'ch{i}_w{j}_avg_power'] = 0
                features[f'ch{i}_w{j}_max_min_ratio'] = 0
    
    return features


def extract_wavelet_features(data: np.ndarray) -> Dict[str, float]:
    """Extract comprehensive wavelet domain features using DWT"""
    features = {}
    
    for i in range(data.shape[0]):
        channel_data = data[i]
        
        max_level = pywt.dwt_max_level(len(channel_data), WAVELET)
        actual_level = min(LEVEL, max_level)
        
        try:
            coeffs = pywt.wavedec(channel_data, WAVELET, level=actual_level)
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = np.sum(energies)
            
            for j, coeff in enumerate(coeffs):
                features[f'ch{i}_w{j}_log_var'] = np.log(np.var(coeff) + 1e-10)
                features[f'ch{i}_w{j}_rel_energy'] = energies[j] / total_energy if total_energy > 0 else 0
                features[f'ch{i}_w{j}_std'] = np.std(coeff)
        except:
            for j in range(actual_level + 1):
                features[f'ch{i}_w{j}_log_var'] = 0
                features[f'ch{i}_w{j}_rel_energy'] = 0
                features[f'ch{i}_w{j}_std'] = 0
    
    return features


def extract_csp_features(data: np.ndarray, labels: np.ndarray, n_components: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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


def extract_wpd_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract Wavelet Packet Decomposition features
    """
    features = {}
    
    for i in range(data.shape[0]):  # For each channel
        channel_data = data[i]
        
        # Create Wavelet Packet tree
        wp = pywt.WaveletPacket(data=channel_data, wavelet=WAVELET, mode='symmetric', maxlevel=MAX_LEVEL)
        
        # Get all nodes (packets) at the deepest level
        nodes = [node.path for node in wp.get_level(MAX_LEVEL, 'natural')]
        
        for node_path in nodes:
            # Extract coefficients for this packet
            coeffs = wp[node_path].data
            
            if len(coeffs) > 0:
                # Energy of packet coefficients
                energy = np.sum(coeffs**2)
                features[f'ch{i}_{node_path}_energy'] = energy
                
                # Variance of packet coefficients
                variance = np.var(coeffs)
                features[f'ch{i}_{node_path}_variance'] = variance
                
                # Coefficient of Variation (COV)
                mean_val = np.mean(np.abs(coeffs))
                if mean_val > 0:
                    cov = np.std(coeffs) / mean_val
                else:
                    cov = 0
                features[f'ch{i}_{node_path}_cov'] = cov
            else:
                features[f'ch{i}_{node_path}_energy'] = 0
                features[f'ch{i}_{node_path}_variance'] = 0
                features[f'ch{i}_{node_path}_cov'] = 0
    
    return features


def extract_all_features_simple(data: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """Extract all simple & efficient features from all domains"""
    features = {}
    
    # Time domain features
    time_features = extract_time_features_simple(data)
    features.update(time_features)
    
    # Frequency domain features
    freq_features = extract_frequency_features_simple(data, fs)
    features.update(freq_features)
    
    # Wavelet domain features
    wavelet_features = extract_wavelet_features_simple(data)
    features.update(wavelet_features)
    
    return features


def extract_all_features(data: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """Extract all comprehensive features from all domains"""
    features = {}
    
    # Time domain features
    time_features = extract_time_features(data)
    features.update(time_features)
    
    # Frequency domain features
    freq_features = extract_frequency_features(data, fs)
    features.update(freq_features)
    
    # Wavelet domain features
    wavelet_features = extract_wavelet_features(data)
    features.update(wavelet_features)
    
    return features


def extract_all_features_discriminant(data: np.ndarray, labels: Optional[np.ndarray] = None, 
                                    csp_filters: Optional[np.ndarray] = None, fs: int = 256) -> Dict[str, float]:
    """
    Extract all discriminant features from all domains
    """
    features = {}
    
    try:
        # CSP features (time domain)
        if csp_filters is not None:
            projected_data = np.dot(csp_filters.T, data)
            csp_log_vars = np.log(np.var(projected_data, axis=1) + 1e-10)
            for comp_idx, log_var in enumerate(csp_log_vars):
                features[f'csp{comp_idx}_log_var'] = log_var
            
            # Frequency features from CSP components
            freq_features = extract_frequency_features_csp(data, csp_filters, fs)
            features.update(freq_features)
        else:
            # If no CSP filters available, use simple variance
            for i in range(data.shape[0]):
                features[f'ch{i}_variance'] = np.var(data[i])
    except Exception as e:
        print(f"Error in CSP feature extraction: {e}")
        # Fallback to simple features
        for i in range(data.shape[0]):
            features[f'ch{i}_variance'] = np.var(data[i])
    
    # WPD features (wavelet domain)
    try:
        wpd_features = extract_wpd_features(data)
        features.update(wpd_features)
    except Exception as e:
        print(f"Error in WPD feature extraction: {e}")
        # Add zero features as fallback
        wp = pywt.WaveletPacket(data=np.zeros(256), wavelet=WAVELET, mode='symmetric', maxlevel=MAX_LEVEL)
        nodes = [node.path for node in wp.get_level(MAX_LEVEL, 'natural')]
        for i in range(data.shape[0]):
            for node_path in nodes:
                features[f'ch{i}_{node_path}_energy'] = 0
                features[f'ch{i}_{node_path}_variance'] = 0
                features[f'ch{i}_{node_path}_cov'] = 0
    
    return features


def extract_frequency_features_csp(data: np.ndarray, csp_filters: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """
    Extract spectral band power from CSP components
    """
    features = {}
    
    # Apply CSP filters to get components
    projected_data = np.dot(csp_filters.T, data)
    n_components = projected_data.shape[0]
    
    for comp_idx in range(n_components):
        component_data = projected_data[comp_idx]
        
        # Compute PSD using Welch's method
        nperseg = min(64, len(component_data))
        noverlap = min(32, nperseg - 1)
        
        freqs, psd = signal.welch(component_data, fs=fs, nperseg=nperseg, 
                                 noverlap=noverlap, window='hann')
        
        # Extract band power for specified bands
        for band, (low, high) in FREQ_BANDS_CSP.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx_band):
                band_power = np.trapz(psd[idx_band], freqs[idx_band])
                features[f'csp{comp_idx}_{band}_power'] = band_power
            else:
                features[f'csp{comp_idx}_{band}_power'] = 0
    
    return features


def process_npz_file(file_path: str, output_path: str, fs: int = 256, method: str = 'comprehensive') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Process a .npz file and extract features from all samples
    
    Args:
        file_path: Input file path
        output_path: Output file path
        fs: Sampling frequency
        method: Feature extraction method ('simple', 'comprehensive', 'discriminant')
    """
    # Load data
    data = np.load(file_path)
    
    # Get keys
    keys = list(data.keys())
    if 'x' in keys:
        eeg_data = data['x']
        labels = data['y'] if 'y' in keys else None
    else:
        eeg_data = data[keys[0]]
        labels = data[keys[1]] if len(keys) > 1 else None
    
    print(f"Processing {os.path.basename(file_path)}, data shape: {eeg_data.shape}")
    
    # Initialize feature matrix
    all_features = []
    all_labels = []
    feature_names = None
    
    # Process each sample
    for i in tqdm(range(eeg_data.shape[0]), desc="Extracting features"):
        sample = eeg_data[i]  # Shape: (23, 256) - 23 channels, 256 time points
        
        try:
            # Extract features based on method
            if method == 'simple':
                features = extract_all_features_simple(sample, fs=fs)
            elif method == 'comprehensive':
                features = extract_all_features(sample, fs=fs)
            elif method == 'discriminant':
                features = extract_all_features_discriminant(sample, fs=fs)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            feature_vector = list(features.values())
            
            # Store feature names only once
            if feature_names is None:
                feature_names = list(features.keys())
            
            all_features.append(feature_vector)
            
            if labels is not None:
                all_labels.append(labels[i])
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels) if labels is not None else None
    
    print(f"Final features shape: {features_array.shape}")
    
    # Save results
    if labels is not None:
        np.savez(output_path, x=features_array, y=labels_array, feature_names=feature_names)
    else:
        np.savez(output_path, x=features_array, feature_names=feature_names)
    
    print(f"Saved features to {output_path}")
    
    return features_array, labels_array, feature_names
