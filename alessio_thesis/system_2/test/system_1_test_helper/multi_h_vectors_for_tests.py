##################################################################
#               System 1 - Multi-Dataset PointNet Extraction
##################################################################
import os, glob, pickle, torch, numpy as np
from tqdm import tqdm

# ------------------     Config     ------------------
PKL_PATTERN = "alessio_thesis/system_2/test/datasets_of_single_pointclouds/dataset_*.pkl"
OUT_PATH    = "alessio_thesis/system_2/test/output_system1/pointnet_features.npz"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POINTNET_MODEL_WEIGHTS = "alessio_thesis/system_1/input/pointnet_weights.pth"
TO_LOAD_POINTNET_MODEL_WEIGHTS = True


# TODO 
# Esiste un modo di importare direttamente system_1/multi_h_vectors.py senza copiare il file?
# Però andando a modificarne i parametri PKL_PATTERN e OUT_PATH ?

# ------------------    Model   ------------------
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from PointNet.models.pointnet_cls import get_model

model = get_model(k=40, normal_channel=False).to(DEVICE)

if TO_LOAD_POINTNET_MODEL_WEIGHTS:
    model.load_state_dict(torch.load(POINTNET_MODEL_WEIGHTS, map_location=DEVICE))

model.eval()

# ------------------     Scan all datasets  ------------------
pkl_files = sorted(glob.glob(PKL_PATTERN))
if not pkl_files:
    raise FileNotFoundError(f"Nessun file trovato con pattern: {PKL_PATTERN}")

print(f"[INFO] Trovati {len(pkl_files)} dataset:")
for f in pkl_files:
    print("   ", f)

feature_vectors, labels = [], []

# ------------------     Extraction loop    ------------------
for file_id, path in enumerate(pkl_files, start=1):
    print(f"\n[INFO] Elaboro {path}")
    with open(path, "rb") as f:
        data_dict = pickle.load(f)

    data_list = data_dict["data"]
    for item in tqdm(data_list, desc=f"Dataset {file_id}", ncols=80):
        pc    = item["point_cloud"]        # [N,2]
        label = int(item["shape_idx"])

        # Pad to 3D (as it is 2D)
        if pc.shape[1] == 2:
            pc = np.pad(pc, ((0,0),(0,1)), mode="constant")

        # [1,3,N]
        pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).transpose(1,2).to(DEVICE)
        with torch.no_grad():
            H, _, _ = model.feat(pc_tensor)
        feature_vectors.append(H.squeeze(0).cpu().numpy().astype(np.float32))
        labels.append(label)

# ------------------     Save combined arrays   ------------------
features = np.stack(feature_vectors, axis=0).astype(np.float32)
labels   = np.asarray(labels, dtype=np.int64)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.savez_compressed(OUT_PATH, features=features, labels=labels)

if not TO_LOAD_POINTNET_MODEL_WEIGHTS:
    torch.save(model.state_dict(), POINTNET_MODEL_WEIGHTS)

print(f"\n[OK] Salvato: {OUT_PATH}")
print(f"     features.shape={features.shape}, labels.shape={labels.shape}")
