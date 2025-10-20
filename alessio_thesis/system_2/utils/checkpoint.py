# utils/checkpoint.py
import os
import torch
from typing import Optional, Dict, Any, Tuple

__all__ = ["save_ckpt", "load_ckpt"]

def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    ## Questo si occupa di salvare in modo atomico, evitando file corrotti in caso di crash
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

    # Casi opzionali
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    hparams: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:

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
        # ? Ho scelto di usare la funzione update per includere i campi extra direttamente nella radice del checkpoint, invece che in un sotto-dizionario
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
    
    if not os.path.exists(path):
        return -1.0, None, False

    obj = torch.load(path, map_location=device)

    # Case A: caso tipico con dizionario completo
    if isinstance(obj, dict) and "model_state" in obj:
        # Controlla coerenza dei parametri attesi
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
    
    # ? Altri formati di checkpoint possono essere gestiti qui, tuttavia il caso più well-defined è quello sopra

    raise ValueError("[ckpt] Unrecognized checkpoint format.")
