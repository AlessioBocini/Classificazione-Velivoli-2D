import torch
import numpy as np

def run_inference(model, inputs, device='cpu', return_probs=False, return_true=False, return_indices=False):

    model.eval()
    out_preds, out_probs, out_true, out_idx = [], [], [], []

    with torch.no_grad():
        # Caso DataLoader
        if hasattr(inputs, '__iter__') and not isinstance(inputs, (np.ndarray, torch.Tensor)):
            for batch in inputs:
                # batch può essere (x,y), (x,y,idx) a seconda del Dataset
                # ? Il caso specfico su cui ho lavorato è stato (x,y)
                # sposta input su device, esegue forward -> logits [B,K]
                # preds: classi stimate per batch (argmax sui logits)
                # opzionale: probabilità (softmax); y_true e indici se forniti dal Dataset

                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        xb, yb, ib = batch
                    elif len(batch) == 2:
                        xb, yb = batch
                        ib = None
                else:
                    xb, yb, ib = batch, None, None

                xb = xb.to(device)
                logits = model(xb)                              # che richiama model.forward(xb)
                preds  = logits.argmax(dim=1).cpu().numpy()     # previsione classi per batch
                out_preds.append(preds)                         # accumula predizioni

                if return_probs:
                    probs = logits.softmax(dim=1).cpu().numpy() # probabilità per batch
                    out_probs.append(probs)                     # accumula probabilità

                #? C'è Differenza tra i due casi
                #? "preds" rappresenta la decisione finale (classe vincente per ciascun campione), utile per metriche discrete come accuracy o F1.
                #? "probs" conserva la distribuzione completa di confidenza (softmax sui logits), utile per analisi probabilistiche e calibrazioni.
                #?  Si calcolano separatamente perché la predizione finale (argmax) non può essere riconvertita in probabilità => è già una scelta discreta.

                if return_true and (yb is not None):
                    # salva etichette vere
                    out_true.append(yb.cpu().numpy())

                # ! Questo è utile solo se il Dataset fornisce gli indici, e per correttezza ho effettuato il controllo
                # ! Tuttavia, non ho mai usato questa funzionalità nei miei esperimenti
                if return_indices and (ib is not None):
                    if isinstance(ib, torch.Tensor):
                        out_idx.append(ib.detach().cpu().numpy().astype(np.int64))
                    else:
                        out_idx.append(np.asarray(ib, dtype=np.int64))

            

        # Caso singolo/batch np/tensor
        else:

            # ? Questo caso si riferisce a un singolo esempio, questo caso specifico è quello utilizzato per gli unit test

            x = torch.as_tensor(inputs, dtype=torch.float32) if isinstance(inputs, np.ndarray) else inputs.float()

            if x.ndim == 1:
                x = x.unsqueeze(0)  # ora rappresenta un batch di 1 dimensione

            x = x.to(device)

            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            out_preds.append(preds)

            if return_probs:
                probs = logits.softmax(dim=1).cpu().numpy()
                out_probs.append(probs)


    # concatena batch results e costruisce l'output nell'ordine:
    # preds [, probs] [, y_true] [, idx]
    # ed infine solleva errori in caso di richieste fallite

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
