# Multimodal Knee Osteoarthritis Diagnosis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A Comprehensive Multimodal Approach for Knee Osteoarthritis Diagnosis Integrating Radiographic and Clinical Data**

This repository implements a deep learning framework that combines **X-ray imaging** and **clinical tabular data** to predict knee osteoarthritis (OA) severity across multiple diagnostic sectors, including the Kellgren-Lawrence (KL) grading system.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [5-Fold Cross-Validation](#5-fold-cross-validation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

Knee osteoarthritis (OA) is a leading cause of disability worldwide. This project addresses the limitations of single-modality approaches by integrating:

- **Radiographic data**: Knee X-ray images processed through EfficientNetV2-S
- **Clinical data**: Patient demographics, pain scores, joint space narrowing (JSN), osteophyte presence, and functional limitations

The model simultaneously predicts **11 OA diagnostic sectors**:
| Sector | Description |
|--------|-------------|
| `kl_grade` | Kellgren-Lawrence grade (0-4) |
| `osteophytes` | Osteophyte presence/severity |
| `jsn` | Joint space narrowing |
| `osfl` | Osteophyte functional limitations |
| `scfl` | Subchondral functional limitations |
| `ostm` | Osteophyte timepoint measurement |
| `sctm` | Subchondral timepoint measurement |
| `osfm` | Osteophyte functional measurement |
| `scfm` | Subchondral functional measurement |
| `ostl` | Osteophyte timepoint limitation |
| `sctl` | Subchondral timepoint limitation |

---

## ✨ Features

- **Multimodal Fusion**: Combines CNN image features with traditional ML clinical features
- **Multitask Learning**: Simultaneous prediction of 11 OA sectors
- **Test-Time Augmentation (TTA)**: Improves prediction robustness
- **5-Fold Stratified Cross-Validation**: Rigorous evaluation with ensemble averaging
- **Class Imbalance Handling**: Balanced class weights and label smoothing
- **Automatic Mixed Precision (AMP)**: Faster training with minimal accuracy loss
- **Comprehensive Logging**: TensorBoard integration for experiment tracking

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-knee-oa.git
cd multimodal-knee-oa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt