# Radio Frequency Classification and Anomaly Detection using Convolutional Neural Networks

This project explores the use of deep learning for classifying Wi-Fi radio frequency (RF) signals and identifying potentially anomalous wireless transmitters.

## Project Overview

The system uses complex In-phase and Quadrature (I/Q) samples extracted from Wi-Fi beacon recordings. Signal-processing techniques and a one-dimensional Convolutional Neural Network (1D-CNN) are used to investigate transmitter-specific signal characteristics.

The initial dataset contains 206 recordings associated with five distinct BSSIDs.

## Objectives

- Analyze and visualize Wi-Fi RF signals.
- Preprocess I/Q samples for deep-learning models.
- Train a CNN to classify signals from known transmitters.
- Evaluate classification performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Extend the system with anomaly detection for unfamiliar or unusual signals.

## Technologies

Python, NumPy, SciPy, Matplotlib, Scikit-learn, PyTorch, and SigMF.

## Current Progress

- [x] Python environment setup
- [x] Dataset inspection and label analysis
- [x] I/Q waveform and frequency-spectrum visualization
- [x] Comparison of recordings from five BSSIDs
- [x] Recording-level train/validation/test preparation
- [ ] CNN training and evaluation
- [ ] Anomaly detection implementation

## Dataset

Wi-Fi RF Fingerprinting Dataset, available on [Zenodo](https://zenodo.org/records/21237610).

The dataset files are not included in this repository.

## Project Status

**In development.** The initial implementation focuses on RF signal classification, with anomaly detection planned as a subsequent stage.