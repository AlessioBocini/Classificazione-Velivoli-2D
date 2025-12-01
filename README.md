# Feature Extraction and Classification of 2D Point Clouds using PointNet
```text
Author: Alessio Bocini
Supervisor: Prof. Giorgio Battistelli
Tutor and Researcher: Matteo Tesori
University of Florence – Department of Information Engineering (DINFO)
Academic Year: 2025/2026
```
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
## Note on the Project
The PointNet/ directory is external to the thesis project (alessio_thesis/).
It contains the reference implementation of PointNet, taken from the original paper: “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation”, Stanford University.

## Note on the I/O

To run the `system_1` procedure (`[multi/extract]_h_vectors.py`), place at least one dataset inside the `system_1/input/datasets/` folder.
`system_1` uses pretrained weights stored in `system_1/input/pointnet_weights.pth`, coming from the first execution.  
You can regenerate and save new weights by setting `TO_LOAD_POINTNET_MODEL_WEIGHTS = False`, but this is usually unnecessary. The initial weights were randomly initialized; loading them only ensures reproducibility.
After completion, `system_1` outputs `pointnet_features.npz` inside the `system_1/output/` folder.

`System2` requires `system_1/output/pointnet_features.npz` to exist.  
During training, System2 loads this file and produces `cls_best.pth` in the project root directory.
For inference-only execution, both `cls_best.pth` and `system_1/output/pointnet_features.npz` must be present.  
The system loads `cls_best.pth` and applies inference to the new `pointnet_features.npz`.


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
