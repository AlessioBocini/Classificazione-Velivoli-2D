"""
extract_pointnet_features.py

Extract global PointNet feature vectors (H) and labels from a dataset .pkl.

Expected dataset structure (in the pickle):
    data_dict = {
        "data": [
            {"point_cloud": np.ndarray shape [N, 2], "shape_idx": int},
            ...
        ]
    }
"""

import torch
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PointNet.models.pointnet_cls import get_model

# ------------------     Config     ------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TO_LOAD_POINTNET_MODEL_WEIGHTS = True
POINTNET_MODEL_WEIGHTS = "alessio_thesis/system_1/input/pointnet_weights.pth"
PKL_PATTERN = "alessio_thesis/system_1/input/datasets/full_dataset.pkl"
OUT_PATH    = "alessio_thesis/system_1/output/pointnet_features.npz"

# Load your .pkl data
with open(PKL_PATTERN, 'rb') as f:
    data_dict = pickle.load(f)

##################################################################
#                                                                #
#               System 1 - PointNet Feature Extraction           #
#                                                                #
##################################################################
# Instantiate PointNet model (no normals, so normal_channel=False)

model = get_model(k=40, normal_channel=False).to(DEVICE)

if TO_LOAD_POINTNET_MODEL_WEIGHTS:
    model.load_state_dict(torch.load(POINTNET_MODEL_WEIGHTS, map_location=DEVICE))

model.eval()

feature_vectors = []
labels = []
data_list = data_dict['data']

for item in data_list:

    pc = item['point_cloud']  # shape [N, 2]
    label = item['shape_idx'] # true class

    # Pad with zeros
    if pc.shape[1] == 2:
        pc = np.pad(pc, ((0, 0), (0, 1)), mode='constant')  # shape [N, 3]

    pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(DEVICE) 

    with torch.no_grad():
        # Get global feature vector H from encoder
        H, _, _ = model.feat(pc_tensor)
        feature_vectors.append(H.squeeze(0).detach().cpu().numpy())  # shape [1024]
        labels.append(label)

# Now feature_vectors is a list of 1024-dim numpy arrays, one per point cloud
# labels is a list of true classes

features = np.stack(feature_vectors).astype(np.float32)
labels = np.asarray(labels, dtype=np.int64)

os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
np.savez_compressed(OUT_PATH, features=features, labels=labels)
if not TO_LOAD_POINTNET_MODEL_WEIGHTS:
    torch.save(model.state_dict(), POINTNET_MODEL_WEIGHTS)

print(f"Saved features and labels to: {OUT_PATH}")
print(f"features.shape={features.shape}, labels.shape={labels.shape}")