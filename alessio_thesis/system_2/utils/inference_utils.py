import torch
import numpy as np

def run_inference(model, inputs, device='cpu', return_probs=False, return_true=False, return_indices=False):
    """
    Inferenza generica:
      - inputs: np.ndarray [D] o [B,D], torch.Tensor, oppure DataLoader
      - return_probs: restituisce softmax [N,K]
      - return_true:  restituisce y_true [N] quando disponibile
      - return_indices: prova a restituire idx originali se il Dataset li fornisce

    Returns (in ordine):
      preds [N], (probs [N,K] se richieste), (y_true [N] se richieste), (idx [N] se richiesti)
    """
    model.eval()
    out_preds, out_probs, out_true, out_idx = [], [], [], []

    with torch.no_grad():
        # Caso DataLoader
        if hasattr(inputs, '__iter__') and not isinstance(inputs, (np.ndarray, torch.Tensor)):
            for batch in inputs:
                # batch può essere (x,y), (x,y,idx) o (x,) a seconda del Dataset
                # sposta input su device, esegue forward -> logits [B,K]
                # preds: classi stimate per batch (argmax sui logits)
                # opzionale: probabilità (softmax); y_true e indici se forniti dal Dataset
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        xb, yb, ib = batch
                    elif len(batch) == 2:
                        xb, yb = batch
                        ib = None
                    else:  # len==1
                        xb, = batch
                        yb, ib = None, None
                else:
                    xb, yb, ib = batch, None, None

                xb = xb.to(device)
                logits = model(xb)                 # [B,K]
                preds  = logits.argmax(dim=1).cpu().numpy()
                out_preds.append(preds)

                if return_probs:
                    probs = logits.softmax(dim=1).cpu().numpy()
                    out_probs.append(probs)

                if return_true and (yb is not None):
                    out_true.append(yb.cpu().numpy())

                if return_indices and (ib is not None):
                    if return_indices and (ib is not None):
                        if isinstance(ib, torch.Tensor):
                            out_idx.append(ib.detach().cpu().numpy().astype(np.int64))
                        else:
                            out_idx.append(np.asarray(ib, dtype=np.int64))

        # Caso singolo/batch np/tensor
        else:
            # converte/assicura tensore float32, aggiunge batch dim se serve, sposta su device
            # forward -> logits, poi preds (+ softmax se richiesto)

            x = torch.as_tensor(inputs, dtype=torch.float32) if isinstance(inputs, np.ndarray) else inputs.float()

            if x.ndim == 1:
                x = x.unsqueeze(0)  # [1,D]
            x = x.to(device)

            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            out_preds.append(preds)

            if return_probs:
                probs = logits.softmax(dim=1).cpu().numpy()
                out_probs.append(probs)


    # concatena batch results e costruisce l'output nell'ordine:
    # preds [, probs] [, y_true] [, idx]
    # solleva errori in caso di richieste fallite

    preds = np.concatenate(out_preds, axis=0)
    ret = [preds]
    if return_probs:
        ret.append(np.concatenate(out_probs, axis=0))
    if return_true:
        if len(out_true) == 0:
            raise ValueError("return_true=True ma l'input non forniva label (DataLoader senza y o input singolo).")
        ret.append(np.concatenate(out_true, axis=0))
    if return_indices:
        if len(out_idx) == 0:
            raise ValueError("return_indices=True ma il Dataset non restituisce indici (serve __getitem__ -> (x,y,idx)).")
        ret.append(np.concatenate(out_idx, axis=0))
    return tuple(ret) if len(ret) > 1 else ret[0]
