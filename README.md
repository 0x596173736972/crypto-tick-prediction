# Crypto Tick-Level Micro-Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A high-frequency trading pipeline for cryptocurrency markets that predicts whether price will hit the high or low first within each 1-minute interval.

## 📌 Overview

This project implements:
- Tick data processing and feature engineering
- Machine learning model (XGBoost) to predict price path
- Comprehensive backtesting framework
- SHAP analysis for model interpretability
- Performance metrics and visualization

## 🛠️ Features

- **Data Processing**:
  - Handles raw tick data (CSV/Parquet)
  - Synthetic data generation for testing
  - Resampling to 1-minute candles

- **Feature Engineering**:
  - Tick imbalance metrics
  - Volume analysis
  - Time-based features with cyclical encoding
  - Rolling volatility and momentum

- **Modeling**:
  - XGBoost classifier with hyperparameter tuning
  - Time-series cross-validation
  - SHAP explainability

- **Backtesting**:
  - Realistic trade simulation
  - Comprehensive performance metrics
  - Detailed visualizations

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-tick-prediction.git
   cd crypto-tick-prediction
