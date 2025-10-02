# EEG Seizure Detection Project

## Overview
This project focuses on EEG signal analysis for epileptic seizure detection using machine learning techniques. The research involves preprocessing EEG data, extracting features from time-domain, frequency-domain, and wavelet transforms, and applying classification algorithms for various binary classification tasks.

## Project Structure
```
eeg-seizure-detection/
├── 📂 src/                           # Source code
│   ├── 📂 preprocessing/             # Data preprocessing scripts
│   │   └── 📄 combination_1.py       # MIT-CHB preprocessing
│   ├── 📂 feature_extraction/        # Feature extraction modules
│   ├── 📂 models/                    # Machine learning models
│   ├── 📂 utils/                     # Utility functions
│   │   └── 📄 data_loader.py         # Data loading utilities
│   └── 📄 main_analysis.py           # Main analysis script
├── 📂 data/                          # Data directory
│   ├── 📂 raw/                       # Raw EEG datasets (.npz files)
│   │   ├── 📄 eeg-seizure_train.npz
│   │   ├── 📄 eeg-seizure_test.npz
│   │   ├── 📄 eeg-seizure_val.npz
│   │   ├── 📄 eeg-seizure_val_balanced.npz
│   │   ├── 📄 eeg-predictive_train.npz
│   │   ├── 📄 eeg-predictive_val.npz
│   │   └── 📄 eeg-predictive_val_balanced.npz
│   └── 📂 processed/                 # Processed data
│       └── 📂 mit_chb/               # MIT-CHB processed datasets
├── 📂 docs/                          # Documentation
│   ├── 📂 papers/                    # Research papers and references (PDFs)
│   ├── 📂 reports/                   # Generated reports
│   └── 📄 citations-20250201T073032.txt # Citation references
├── 📂 notebooks/                     # Jupyter notebooks for analysis
│   └── 📄 mitchb_comb1.ipynb         # MIT-CHB analysis notebook
├── 📂 results/                       # Results and outputs
│   ├── 📂 figures/                   # Generated plots and visualizations
│   ├── 📂 models/                    # Trained model files
│   └── 📂 logs/                      # Experiment logs
├── 📂 config/                        # Configuration files
│   └── 📄 config.py                  # Project configuration
├── 📄 README.md                      # Project documentation
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                       # Package setup
├── 📄 run_analysis.py                # Main runner script
└── 📄 .gitignore                     # Git ignore file
```

## Datasets
The project works with preprocessed EEG datasets stored as NumPy archives (.npz files):

### **EEG Seizure Detection Datasets:**
- **eeg-seizure_train.npz**: Training data for seizure detection
- **eeg-seizure_test.npz**: Test data for seizure detection  
- **eeg-seizure_val.npz**: Validation data for seizure detection
- **eeg-seizure_val_balanced.npz**: Balanced validation data

### **EEG Predictive Datasets:**
- **eeg-predictive_train.npz**: Training data for seizure prediction
- **eeg-predictive_val.npz**: Validation data for seizure prediction
- **eeg-predictive_val_balanced.npz**: Balanced validation data for prediction

### **MIT-CHB Dataset:**
- Processed MIT-CHB pediatric seizure dataset files in `data/processed/mit_chb/`

## Features
- Time-domain features (mean, variance, skewness, kurtosis)
- Frequency-domain features (FFT-based spectral analysis)
- Wavelet-based features (Daubechies wavelet decomposition)
- Bandpass filtering (Delta, Theta, Alpha, Beta, Gamma bands)

## Analysis Methods
- **Machine Learning Models**: K-Nearest Neighbors (KNN) classifier
- **Feature Selection**: Sequential Forward Floating Selection (SFFS)
- **Statistical Analysis**: T-test based feature selection
- **Cross-validation**: 5-fold cross-validation for model evaluation
- **Visualization**: Confusion matrices, accuracy plots, feature importance analysis

## Usage

### **Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis
python run_analysis.py --analysis

# Launch Jupyter notebook
python run_analysis.py --notebook
```

### **Manual Execution:**
```bash
# Run main analysis script directly
python src/main_analysis.py

# Open specific notebook
jupyter notebook notebooks/mitchb_comb1.ipynb
```

### **Project Setup:**
```bash
# Install as package (optional)
pip install -e .
```

## Authors
- Srishti Ghosh
- Vandita Singh

## Dependencies
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- pywavelets
- mlxtend
