# utils/checkpoint.py
import os
import torch
from typing import Optional, Dict, Any, Tuple

__all__ = ["save_ckpt", "load_ckpt"]

def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_ckpt(
    model: torch.nn.Module,
    path: str,
    *,
    best_val_acc: float,
    num_classes: int,
    input_dim: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    hparams: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a rich checkpoint with model/optim/scheduler + metadata.
    Overwrites atomically.

    Fields:
      - model_state, optimizer_state, scheduler_state
      - best_val_acc, num_classes, input_dim, epoch, hparams, extra
    """
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "best_val_acc": float(best_val_acc),
        "num_classes": int(num_classes),
        "input_dim": int(input_dim),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if hparams:
        payload["hparams"] = dict(hparams)
    if extra:
        payload.update(extra)

    _atomic_save(payload, path)

def load_ckpt(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    *,
    expected_num_classes: Optional[int] = None,
    expected_input_dim: Optional[int] = None,
    resume: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    strict: bool = True,
) -> Tuple[float, Optional[int], bool]:
    """
    Load a checkpoint into `model`.

    Returns: (best_val_acc, epoch, loaded)
      - best_val_acc: float (=-1.0 if unknown)
      - epoch: int or None
      - loaded: bool (True if anything was loaded)

    Behavior:
      - If `path` is a rich checkpoint: validates shapes (when metadata is present),
        loads model_state, optionally optimizer/scheduler if `resume=True`.
      - If `path` is a plain state_dict: loads directly into model; best_val_acc=-1.0, epoch=None.
      - If `path` missing: returns (-1.0, None, False).
    """
    if not os.path.exists(path):
        return -1.0, None, False

    obj = torch.load(path, map_location=device)

    # Case A: rich checkpoint dict
    if isinstance(obj, dict) and "model_state" in obj:
        # shape checks if metadata present
        if expected_num_classes is not None and "num_classes" in obj:
            if int(obj["num_classes"]) != int(expected_num_classes):
                raise ValueError(
                    f"[ckpt] num_classes mismatch: ckpt={obj['num_classes']} vs expected={expected_num_classes}"
                )
        if expected_input_dim is not None and "input_dim" in obj:
            if int(obj["input_dim"]) != int(expected_input_dim):
                raise ValueError(
                    f"[ckpt] input_dim mismatch: ckpt={obj['input_dim']} vs expected={expected_input_dim}"
                )

        model.load_state_dict(obj["model_state"], strict=strict)

        if resume:
            if optimizer is not None and "optimizer_state" in obj:
                optimizer.load_state_dict(obj["optimizer_state"])
            if scheduler is not None and "scheduler_state" in obj:
                scheduler.load_state_dict(obj["scheduler_state"])

        best = float(obj.get("best_val_acc", -1.0))
        epoch = obj.get("epoch", None)
        if isinstance(epoch, int):
            epoch_loaded = epoch
        else:
            epoch_loaded = None
        return best, epoch_loaded, True

    # Case B: plain state_dict (nn.Module.state_dict())
    if isinstance(obj, (dict, torch.nn.modules.module.OrderedDict)):
        # Basic sanity check: try to load; if shapes mismatch, PyTorch will raise
        model.load_state_dict(obj, strict=strict)
        return -1.0, None, True

    raise ValueError("[ckpt] Unrecognized checkpoint format.")