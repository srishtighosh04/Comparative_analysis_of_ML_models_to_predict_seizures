"""
Setup script for EEG Seizure Detection Project
"""

from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "EEG signal analysis for epileptic seizure detection using machine learning"

try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "PyWavelets>=1.3.0",
        "mlxtend>=0.19.0",
        "pandas>=1.3.0",
        "jupyter>=1.0.0",
    ]

setup(
    name="eeg-seizure-detection",
    version="1.0.0",
    author="Srishti Ghosh, Vandita Singh",
    author_email="",
    description="EEG signal analysis for epileptic seizure detection using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="eeg, seizure detection, machine learning, signal processing, epilepsy, neuroscience",
    entry_points={
        "console_scripts": [
            "eeg-analysis=run_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
