# Signal Processing for Machine Learning

This repo aims to help people with intrest about ML/DL but lack the foundation in signal processing, a topic that is essential for ML and DL applications such as NLP 

## Core Educational Modules

### 1. Optimal Estimation and Adaptive Filtering
Fundamentals of estimating signals and adapting system parameters in real-time, which are essential for ML model optimization:
* **Wiener Filtering**: Implementation of the Wiener-Hopf equations for optimal linear estimation.
* **Least Mean Squares (LMS) & NLMS**: Stochastic gradient descent applications in adaptive signal processing.
* **Recursive Least Squares (RLS)**: High-performance adaptive filtering for rapid convergence in stationary environments.
* **Steepest Descent Optimization**: Iterative optimization techniques and convergence analysis using eigenvalue decomposition.
* **Kalman Filtering**: State-space estimation and recursive Bayesian tracking for dynamic signals.

### 2. Feature Extraction and Representation
Techniques for transforming raw data into meaningful representations for downstream learning:
* **Linear Predictive Coding (LPC)**: Modeling signals as linear combinations of past samples, a precursor to many autoregressive ML models.
* **Vector and Scalar Quantization**: Data compression and codebook design using the LBG algorithm and various probability distributions (Gaussian, Laplacian).
* **Correlation Analysis**: Computing auto-correlation and cross-correlation matrices as features for statistical learning.

### 3. Signal Enhancement and Channel Modeling
Processing signals to improve quality and modeling the environments through which they travel:
* **Channel Estimation**: Using adaptive filters to identify unknown system impulses.
* **Adaptive Equalization**: Compacting channel distortion in communication signals using iterative algorithms.
* **Noise Characterization**: Methods for adding and scaling Gaussian noise based on specific SNR requirements.

## Repository Organization
* **/DSP_labs/src**: Core MATLAB/Source functions and algorithm implementations.
* **/Notebooks**: Links to Kaggle notebooks containing interactive visualizations, mathematical derivations, and performance analysis.

## Usage
The scripts in `/src` are designed for educational exploration. Start with the main simulation files (e.g., `wiener_filter_main.m` or `channel_Equalizer.m`) to observe how signal processing parameters impact algorithm performance.

## License
This project is licensed under the MIT License.
