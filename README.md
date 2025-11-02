# Event Classification — Enhanced PyTorch Version

This repository integrated **Sherpa optimization** on [mrheng9/Event-Classification-PyTorch](https://github.com/mrheng9/Event-Classification-PyTorch) into a **complete, reproducible machine learning workflow**.  

---

## What’s New

- **Workflow**: separate scripts for training (`train.py`), optimization (`optimize.py`), inference (`predict.py`), and plotting (`plot.py`) plus model/optimizer utilities (`model_optim.py`).
- **Sherpa integration**: automatic search over LR/activation/optimizer, with logs saved for reference.

---

## Install

> Environment for this repo: Python 3.10; Cuda 12.4 + CuDNN 8.9.0 ; PyTorch 2.6.0

```bash
git clone https://github.com/AhrixRain/Event-Classification-PyTorch.git
cd Event-Classification-PyTorch

# If you use conda:
# conda create -n eventcls python=3.10 -y && conda activate eventcls

# Install deps (pins reduce weird version conflicts, esp. with Sherpa & NumPy)
pip install -r requirements.txt
