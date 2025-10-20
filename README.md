# Feature Extraction and Classification of 2D Point Clouds using PointNet

Author: Alessio Bocini

University of Florence – Department of Information Engineering (DINFO)
Academic Year: 2025/2026
---

## Brief Overview
This thesis presents a two-stage pipeline for the representation and classification of 2D point cloud data.  
The first stage (System 1) uses a pretrained **PointNet** model to extract compact, high-level **H-vectors** (global shape descriptors).  
The second stage (System 2) trains an **MLP classifier** on these H-vectors to recognize object categories.  

---

## Project Structure
```text
.
├── alessio_thesis/
│   ├── models/
│   │   └── mlp_head.py                 # MLP classifier head for H-vectors
│   │
│   ├── system_1/                       # System 1: PointNet feature extraction
│   │   ├── input/
│   │   │   ├── datasets/               # Input .pkl datasets (e.g. full_dataset_*.pkl)
│   │   │   └── pointnet_weights.pth    # Pretrained PointNet weights
│   │   ├── output/
│   │   │   └── pointnet_features.npz   # Extracted H-vectors
│   │   ├── extract_h_vectors.py        # Single-dataset feature extraction
│   │   └── multi_h_vectors.py          # Multi-dataset feature extraction
│   │
│   ├── system_2/                       # System 2: H-vector classifier
│   │   ├── utils/
│   │   │   ├── checkpoint.py
│   │   │   ├── inference_utils.py
│   │   │   └── training_utils.py
│   │   └── classifier_h.py             # MLP classifier trained on H-vectors
│   │
│   └── wrapper/
│       └── h_dataset.py                # Dataset wrapper for loading H-vectors and labels
│
└── PointNet/                           # External module (not part of the thesis exercise)
    └── models/
        ├── pointnet_cls.py             # Original PointNet architecture (classification)
        └── pointnet_utils.py           # Utility layers and transformation nets
```
## Note
The PointNet/ directory is external to the thesis project (alessio_thesis/).
It contains the reference implementation of PointNet, taken from the original paper: “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation”, Stanford University.

## System 1 – Feature Extraction

```text
.
cd alessio_thesis/system_1

# Single dataset (pattern: full_dataset.pkl)
python extract_h_vectors.py

# Multiple datasets (pattern: full_dataset_*.pkl)
python multi_h_vectors.py
```

## System 2 – Classifier Training
```text
.
cd alessio_thesis/system_2
python classifier_h.py
```
## Versions used
```text
python 3.13.5
torch 2.7.1
numpy 2.3.2
scikit-learn 1.7.0
tqdm 4.67.1
```
