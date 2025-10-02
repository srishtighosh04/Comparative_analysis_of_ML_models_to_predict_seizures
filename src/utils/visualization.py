"""
Visualization utilities for EEG analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple, List
import os
import sys

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import FIGURES_DIR, FIGURE_SIZE, HEATMAP_COLORS


def setup_plot_style():
    """Setup consistent plot styling"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix with improved styling
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        title: Plot title
        save_path: Path to save figure
    """
    setup_plot_style()
    
    plt.figure(figsize=FIGURE_SIZE)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=HEATMAP_COLORS,
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")
    
    plt.show()


def plot_accuracy_vs_features(feature_counts: List[int], 
                             accuracies: List[float],
                             title: str = "Feature Selection Results",
                             save_path: Optional[str] = None) -> None:
    """
    Plot accuracy vs number of features
    
    Args:
        feature_counts: Number of features
        accuracies: Corresponding accuracies
        title: Plot title
        save_path: Path to save figure
    """
    setup_plot_style()
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(feature_counts, accuracies, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy plot to: {save_path}")
    
    plt.show()


def plot_p_values(p_values: List[float],
                 threshold: float = 0.05,
                 title: str = "T-test P-values",
                 save_path: Optional[str] = None) -> None:
    """
    Plot p-values from statistical tests
    
    Args:
        p_values: List of p-values
        threshold: Significance threshold
        title: Plot title
        save_path: Path to save figure
    """
    setup_plot_style()
    
    plt.figure(figsize=FIGURE_SIZE)
    feature_indices = range(1, len(p_values) + 1)
    
    plt.plot(feature_indices, p_values, marker='o', color='blue', 
             label='P-value', linewidth=2, markersize=4)
    plt.axhline(y=threshold, color='red', linestyle='--', 
                label=f'Threshold ({threshold})', linewidth=2)
    
    plt.xlabel('Feature Index')
    plt.ylabel('P-value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved p-values plot to: {save_path}")
    
    plt.show()


def plot_eeg_signal(signal: np.ndarray,
                   fs: int = 256,
                   title: str = "EEG Signal",
                   duration_sec: Optional[float] = None,
                   save_path: Optional[str] = None) -> None:
    """
    Plot EEG signal
    
    Args:
        signal: EEG signal data
        fs: Sampling frequency
        title: Plot title
        duration_sec: Duration to plot in seconds
        save_path: Path to save figure
    """
    setup_plot_style()
    
    if duration_sec:
        max_samples = int(duration_sec * fs)
        signal = signal[:max_samples]
    
    time = np.arange(len(signal)) / fs
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, linewidth=0.8)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved EEG signal plot to: {save_path}")
    
    plt.show()


def save_figure(filename: str, subfolder: str = "") -> str:
    """
    Generate save path for figures
    
    Args:
        filename: Name of the file
        subfolder: Subfolder within figures directory
        
    Returns:
        Full path for saving
    """
    if subfolder:
        save_dir = os.path.join(FIGURES_DIR, subfolder)
    else:
        save_dir = FIGURES_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


def plot_feature_distribution(features: np.ndarray,
                             labels: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             title: str = "Feature Distributions",
                             save_path: Optional[str] = None) -> None:
    """
    Plot feature distributions by class
    
    Args:
        features: Feature matrix
        labels: Class labels
        feature_names: Names of features
        title: Plot title
        save_path: Path to save figure
    """
    setup_plot_style()
    
    n_features = min(features.shape[1], 16)  # Limit to 16 features for readability
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    unique_labels = np.unique(labels)
    
    for i in range(n_features):
        ax = axes[i]
        
        for label in unique_labels:
            mask = labels == label
            ax.hist(features[mask, i], alpha=0.7, label=f'Class {label}', bins=20)
        
        if feature_names and i < len(feature_names):
            ax.set_title(feature_names[i], fontsize=10)
        else:
            ax.set_title(f'Feature {i+1}', fontsize=10)
        
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature distribution plot to: {save_path}")
    
    plt.show()
