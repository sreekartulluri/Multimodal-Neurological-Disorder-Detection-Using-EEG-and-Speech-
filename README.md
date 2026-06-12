# 🧠 AURA-Net: Multimodal Neurological Disorder Detection Using EEG and Speech

> **A**ttention-based m**U**ltimodal neu**R**ological dis**A**gnostic **Net**work

An end-to-end deep learning framework that fuses **EEG brain signals** and **speech acoustic biomarkers** using **bilateral cross-attention** to classify neurological disorders — Parkinson's Disease, Alzheimer's Disease, Epilepsy, and Healthy controls.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features Extracted](#features-extracted)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Explainability](#explainability)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Overview

Neurological disorders such as Parkinson's, Alzheimer's, and Epilepsy affect millions worldwide. Traditional diagnosis relies heavily on subjective clinical assessments. **AURA-Net** aims to provide an objective, AI-assisted diagnostic tool by leveraging two complementary physiological modalities:

1. **EEG (Electroencephalography)** — captures brain electrical activity patterns
2. **Speech Audio** — captures vocal biomarkers affected by motor and cognitive impairment

The system extracts clinically meaningful features from both modalities, fuses them through a **multi-head cross-attention mechanism**, and classifies subjects into one of four categories:

| Class | Description |
|-------|-------------|
| Healthy | Normal neurological function |
| Parkinson's Disease | Motor neurodegenerative disorder |
| Alzheimer's Disease | Cognitive neurodegenerative disorder |
| Epilepsy | Seizure disorder with abnormal brain activity |

---

## Architecture

```
┌─────────────────┐      ┌──────────────────┐
│  Speech Audio   │      │   EEG Signal     │
│   (.wav file)   │      │   (.edf file)    │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌──────────────────┐
│ Speech Feature  │      │  EEG Feature     │
│  Extraction     │      │  Extraction      │
│  (66 features)  │      │  (494 features)  │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌──────────────────┐
│  Dense Encoder  │      │  Dense Encoder   │
│  (128 → 64-d)   │      │  (256 → 64-d)    │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Bilateral       │
         │  Cross-Attention │
         │  Fusion (MHA)    │
         │  (4 heads)       │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Deep Classifier │
         │  (128 → 64 → 4)  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   Prediction     │
         │   (4 Classes)    │
         └──────────────────┘
```

### Cross-Attention Fusion

The fusion module performs **bilateral multi-head cross-attention**:
- **Speech-to-EEG Attention**: Speech queries attend to EEG keys/values — learns what brain activity patterns are relevant for each speech feature
- **EEG-to-Speech Attention**: EEG queries attend to Speech keys/values — learns what vocal biomarkers complement each brain signal

The original embeddings and cross-attended representations are concatenated (256-d) and projected to a 128-d fused embedding.

---

## Features Extracted

### Speech Features (66 dimensions)
| Feature | Count | Description |
|---------|-------|-------------|
| MFCCs (mean + std) | 26 | Mel-Frequency Cepstral Coefficients — vocal tract shape |
| Chroma (mean + std) | 24 | Pitch class energy distribution |
| Pitch (mean + std) | 2 | Fundamental frequency via YIN algorithm |
| RMS Energy | 2 | Root mean square energy |
| Zero Crossing Rate | 2 | Signal sign change rate |
| Spectral Centroid | 2 | Center of spectral mass |
| Spectral Bandwidth | 2 | Spread of spectrum |
| Spectral Rolloff | 2 | Frequency below which 85% energy exists |
| Tempo | 1 | Estimated speech rhythm |
| Jitter | 1 | Pitch period perturbation (vocal tremor) |
| Shimmer | 1 | Amplitude perturbation (vocal instability) |
| HNR | 1 | Harmonic-to-Noise Ratio (breathiness) |

### EEG Features (494 dimensions)
| Feature | Count | Description |
|---------|-------|-------------|
| Band Powers (δ, θ, α, β, γ) | 95 | Absolute spectral power per channel (19 channels × 5 bands) |
| Relative Band Powers | 95 | Normalized power relative to total |
| Band Ratios (θ/α, α/β, θ/β) | 57 | Clinical ratios per channel |
| Shannon Entropy | 19 | Spectral complexity per channel |
| Hjorth Parameters | 57 | Activity, mobility, complexity per channel |
| Functional Connectivity | 171 | Pearson correlation (upper triangle of 19×19 matrix) |

EEG channels follow the **international 10-20 system** (19 channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz).

---

## Project Structure

```
├── model.py              # Neural network architecture (DenseEncoder, CrossAttentionFusion, Classifier)
├── eeg_features.py       # EEG feature extraction from .edf files (494-d vector)
├── speech_features.py    # Speech feature extraction from .wav files (66-d vector)
├── dataset.py            # PyTorch Dataset and DataLoader with train/val split & scaling
├── train.py              # Training loop with early stopping, LR scheduling, metrics
├── predict.py            # Inference pipeline for new patient files
├── explainability.py     # SHAP-based model explainability and visualization
├── generate_mock_data.py # Synthetic dataset generation for demonstration
├── app.py                # Streamlit clinical dashboard (AURA-Net UI)
├── samples/              # Demo patient EEG (.edf) and Speech (.wav) files
│   ├── sample_healthy.*
│   ├── sample_parkinsons_disease.*
│   ├── sample_alzheimers_disease.*
│   └── sample_epilepsy.*
├── confusion_matrix.png  # Generated confusion matrix from training
├── shap_explanation.png  # Generated SHAP explanation plot
└── .gitignore
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/sreekartulluri/Multimodal-Neurological-Disorder-Detection-Using-EEG-and-Speech-.git
cd Multimodal-Neurological-Disorder-Detection-Using-EEG-and-Speech-

# Install dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn librosa mne shap streamlit pandas soundfile
```

---

## Usage

### 1. Generate Mock Dataset & Train the Model

```bash
python train.py
```

This will:
- Auto-generate synthetic mock data if `features_dataset.pt` doesn't exist
- Train the multimodal model for 30 epochs with early stopping
- Save the best checkpoint to `multimodal_checkpoint.pth`
- Generate a confusion matrix plot

### 2. Run Predictions on New Data

```python
from predict import load_prediction_pipeline, predict_disorder

pipeline = load_prediction_pipeline('multimodal_checkpoint.pth')
results = predict_disorder('path/to/speech.wav', 'path/to/eeg.edf', pipeline)

print(f"Prediction: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.2%}")
```

### 3. Launch the Clinical Dashboard

```bash
streamlit run app.py
```

The Streamlit dashboard provides:
- 🎤 Upload or select demo Speech (.wav) and EEG (.edf) files
- 📊 Real-time waveform and signal visualization
- 🩺 Multimodal diagnostic prediction with confidence scores
- 🔍 SHAP-based explainability with clinical interpretations
- 📈 Model training performance metrics and confusion matrix

---

## Model Details

| Component | Details |
|-----------|---------|
| Speech Encoder | Linear(66→128) → BN → ReLU → Dropout → Linear(128→64) → BN → ReLU → Dropout |
| EEG Encoder | Linear(494→256) → BN → ReLU → Dropout → Linear(256→64) → BN → ReLU → Dropout |
| Cross-Attention | 4-head Multi-Head Attention (bilateral) with residual concatenation |
| Fusion Projection | Linear(256→128) → BN → ReLU → Dropout |
| Classifier | Linear(128→64) → BN → ReLU → Dropout → Linear(64→4) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | Patience = 7 epochs |

---

## Explainability

AURA-Net uses **SHAP (SHapley Additive exPlanations)** with a KernelExplainer to provide interpretable predictions:

- Computes per-feature SHAP values for the predicted class
- Identifies the top 15 contributing biomarkers
- Visualizes feature importances with directional impact (positive/negative)
- Provides clinical interpretation for key features:
  - **MFCC changes** → Vocal tract rigidity (dysarthria)
  - **Jitter/Shimmer** → Laryngeal muscle tremor
  - **HNR** → Vocal breathiness/hoarseness
  - **Theta power** → Cortical slowing (Alzheimer's biomarker)
  - **Beta oscillations** → Motor cortex inhibition (Parkinson's biomarker)
  - **Hjorth parameters** → Brain rhythm irregularity
  - **Connectivity** → Functional network disintegration

---

## Results

The model outputs:
- **Accuracy**, **Precision**, **Recall**, and **F1-Score** (macro-averaged)
- A **confusion matrix** heatmap across all four classes
- Per-prediction **probability distributions** and **SHAP explanations**

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch |
| Signal Processing | MNE-Python, SciPy, Librosa |
| Machine Learning | scikit-learn |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Web Dashboard | Streamlit |
| Data Format | EDF (EEG), WAV (Speech) |

---

## ⚠️ Disclaimer

> This project is developed for **academic and research purposes only**. The predictions generated by AURA-Net are **not** a substitute for professional medical diagnosis. All results must be verified by a board-certified neurologist.

---

## License

This project is part of an academic coursework submission for Deep Learning and Neural Networks.
