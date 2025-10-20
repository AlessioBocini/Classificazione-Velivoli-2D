import torch
from torch.utils.data import Dataset

class HDataset(Dataset):
    """
    Simple dataset wrapper for precomputed feature vectors (H) and labels.
    Parameters
    ----------
    X : np.ndarray, shape [N, D]
        Feature vectors of PointNet embeddings.
    y : np.ndarray, shape [N]
        Integer class labels (0..K-1).
    idx : list or np.ndarray
        Indices to subset the dataset (train/val/test split).
    """
    def __init__(self, X, y, idx):
        self.X = torch.from_numpy(X[idx]).float()
        self.y = torch.from_numpy(y[idx]).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]