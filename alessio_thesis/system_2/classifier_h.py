##################################################################
#                                                                #
#               System 2 - Classifier on H                       #
#                                                                #
##################################################################
import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( confusion_matrix, classification_report )
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mlp_head import HClassifier
from wrapper.h_dataset import HDataset
from utils.checkpoint import load_ckpt, save_ckpt
from utils.training_utils import run_epoch
from utils.inference_utils import run_inference

# ----   CONFIGURATIONS   ----
EVAL_ONLY = True     # set True to skip training and just eval

CKPT_PATH = "alessio_thesis/cls_best.pth"
NPZ_PATH = 'alessio_thesis/system_1/output/pointnet_features.npz'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR_DEFAULT = 3e-4
WD_DEFAULT = 1e-3
DROPOUT = 0.35
# ----  DATASET / DATALOADER     ----
BATCH_SIZE = 64
# ----  TRAINING CONFIGS  ----
EPOCHS = 10 
VALIDATION_SIZE = 0.2 

if not os.path.isfile(NPZ_PATH):
    raise FileNotFoundError(f"File not found: {NPZ_PATH}")

# load arrays
data = np.load(NPZ_PATH)
feature_vectors = data["features"]   # shape [N, 1024]
labels = data["labels"]              # shape [N]

# ------------------    Data   ------------------

X = feature_vectors.astype(np.float32)
y = labels.astype(np.int64)
NUM_CLASSES = int(np.unique(y).size)
INPUT_DIM   = int(X.shape[1])

print(f"[INFO] Samples: {len(X)} | Classes: {NUM_CLASSES} | Input dim: {INPUT_DIM}")

clf = HClassifier(INPUT_DIM, NUM_CLASSES, p_drop=DROPOUT).to(DEVICE)

if not EVAL_ONLY:

    train_idx, val_idx = train_test_split(
        np.arange(len(y)),          # indices to divide
        test_size=VALIDATION_SIZE,  # validation size (e.g. 0.2 = 20%)
        random_state=42,            # reproducible shuffle
        stratify=y                  # keeps the class proportions in a well distributed way
    )

    print(f"[INFO] Train: {len(train_idx)} | Val: {len(val_idx)}")

    # ----  Dataset / Dataloader    ----
    # HDataset converts the subsets into tensors compatible with PyTorch
    train_ds = HDataset(X, y, train_idx)
    val_ds   = HDataset(X, y, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ----  Training setup   ----
    criterion = nn.CrossEntropyLoss()                                                           # loss function for multi-class classification
    optimizer = torch.optim.AdamW(clf.parameters(), lr=LR_DEFAULT, weight_decay=WD_DEFAULT)     # AdamW optimizer with weight decay (L2 regularization)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)             # cosine annealing learning rate scheduler over epochs
    
    prev_best, prev_epoch, loaded = load_ckpt(
        clf, CKPT_PATH, DEVICE,
        expected_num_classes=NUM_CLASSES,
        expected_input_dim=INPUT_DIM,
        resume=False,         
        optimizer=optimizer,   # only used if resume=True
        scheduler=scheduler,   # only used if resume=True
        strict=True
    )

    best_val_acc = prev_best if prev_best >= 0 else 0.0
    best_state = None
    prev_best = -1.0

    # ----  Training     ----
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(train_loader, model=clf, optimizer=optimizer, criterion=criterion, device=DEVICE, train=True)
        val_loss, val_acc = run_epoch(val_loader, model=clf, optimizer=optimizer, criterion=criterion, device=DEVICE, train=False)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "f"val loss {val_loss:.4f} acc {val_acc:.3f}")
            
    if best_state is not None and best_val_acc > prev_best:
        clf.load_state_dict(best_state, strict=True)
        save_ckpt(
            clf, CKPT_PATH,
            best_val_acc=best_val_acc,
            num_classes=NUM_CLASSES,
            input_dim=INPUT_DIM,
            optimizer=optimizer,          
            scheduler=scheduler,          
            epoch=epoch,                  # last epoch reached
            hparams={"lr": LR_DEFAULT, "wd": WD_DEFAULT, "dropout": DROPOUT, "batch": BATCH_SIZE}
        )
else:
    # ----  EVAL_ONLY    ----
    prev_best, _, loaded = load_ckpt(
        clf, CKPT_PATH, DEVICE,
        expected_num_classes=NUM_CLASSES,
        expected_input_dim=INPUT_DIM,
        resume=False,
        strict=True
    )
    assert loaded, f"Checkpoint non trovato o incompatibile: {CKPT_PATH}"

    if not loaded:
        raise RuntimeError(f"Checkpoint non trovato o incompatibile: {CKPT_PATH}")

    val_idx = np.arange(len(y))
    val_ds = HDataset(X, y, val_idx)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---- Final validation metrics ----
preds, probs, y_true = run_inference(clf, val_loader, device=DEVICE, return_probs=True, return_true=True)
K = probs.shape[1]
labels_order = np.arange(K)

print("\nClassification Report:")
print(classification_report(y_true, preds, labels=labels_order, digits=4))

# print cm
cm    = confusion_matrix(y_true, preds,labels=labels_order)

col_w = max(5, len(str(cm.max())))
header = " " * col_w + "".join(f"{i:>{col_w}d}" for i in labels_order)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(header)
for i in labels_order:
    print(f"{i:>{col_w}d}" + "".join(f"{cm[i,j]:>{col_w}d}" for j in labels_order))


# ----  UNIT TESTS - SYSTEM 2 ----
from system_2.test.functions.test_system_2 import TestRunInference
import unittest

if __name__ == "__main__":

    TestRunInference.model = clf
    TestRunInference.device = DEVICE

    NPZ_PATH_TESTS = "alessio_thesis/system_2/test/output_system1/pointnet_features.npz"

    data_dict = np.load(NPZ_PATH_TESTS)

    testing_labels = data_dict['labels']
    testing_features = data_dict['features']

    list_samples = []

    for i in range(len(testing_labels)):
        list_samples.append({
            'vector_h_cloud': testing_features[i],
            'correct_label': int(testing_labels[i])
        })
        
    for i, case in enumerate(list_samples):
        test_fn = TestRunInference._make_case_test(i, case['vector_h_cloud'], case['correct_label'])
        setattr(TestRunInference, test_fn.__name__, test_fn)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRunInference)
    unittest.TextTestRunner(verbosity=2).run(suite)