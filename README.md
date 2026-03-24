# MABSS: Dynamic Predictor Selection for Financial Time Series

This repository contains the official code, experiments, and results to reproduce the findings of the paper: **"Dynamic Predictor Selection for Financial Time Series Using Contextual Multi-Armed Bandits in a Reinforcement Learning Framework"**. 

## 📖 Project Overview
Financial markets are inherently non-stationary, meaning the optimal predictive model often changes as the underlying market regime shifts. To address this, this project introduces a dynamic, Reinforcement Learning-based framework that leverages Contextual Multi-Armed Bandits (CMABs) to sequentially select the most effective predictor from a diverse pool of deep learning models (MLPs, CNNs, and RNNs) at each time step.

The framework evaluates three distinct CMAB exploration policies (Softmax, LinUCB, and Thompson Sampling) across five distinct asset classes:
* **Equities** (SPY, QQQ)
* **Cryptocurrency** (BTC-USD)
* **Foreign Exchange** (EURUSD=X) 
* **Commodities** (GC=F) 
* **Sovereign Bonds** (TLT) 

## 📁 Repository Structure
* `MABSS_price_predictions.ipynb`: The main Google Colab notebook used to run the walk-forward validation experiments, evaluate the CMAB policies, and generate the plots presented in the paper.
* `MABSS_utility.py`: A Python module containing all the underlying helper functions, including data fetching, deep learning model architectures, and CMAB policy implementations.
* `experiments/` and `experiments_cluster/`: Folders containing the pre-calculated experiment results and performance metrics used for the paper's analysis.

## 🚀 How to Run the Experiments
The easiest way to reproduce the experiments is using Google Colab.
1. Open the `.ipynb` notebook file in this repository.
2. Click the **"Open in Colab"** button at the top of the file view.
3. Ensure that the `MABSS_utility.py` file and the necessary `experiments/` folders are accessible in your Colab environment (e.g., by mounting your Google Drive or cloning the repository directly into the Colab runtime).
4. Run all cells to execute the code and reproduce the plots.

## 🛠️ Prerequisites
If you prefer to run the code locally, you will need Python 3.x and the following key libraries:
* `numpy`
* `pandas`
* [cite_start]`yfinance` (for historical market data retrieval) [cite: 397]
* Deep learning libraries (e.g., `PyTorch` or `TensorFlow`, depending on what you used for the MLPs, CNNs, and RNNs).
* For the other required libraries check the requirements.txt file

## 📄 Citation
If you use this code or framework in your own research, please consider citing our paper:
> Saviozzi, S., Romito, M., & Carlei, V. (2026). Dynamic Predictor Selection for Financial Time Series Using Contextual Multi-Armed Bandits in a Reinforcement Learning Framework.
