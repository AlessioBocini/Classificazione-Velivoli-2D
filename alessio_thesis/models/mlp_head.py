import torch.nn as nn

class HClassifier(nn.Module):
    """
    Simple MLP classifier head for PointNet global feature vectors (H).
    
    Parameters
    ----------
    d : int
        Input dimension (default: 1024 for PointNet global feature).
    num_classes : int
        Number of output classes.
    p_drop : float
        Dropout probability.
    """
    def __init__(self, d=1024, num_classes=3, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, num_classes)
        )
    def forward(self, H):
        return self.net(H)  # logits (B, K)
