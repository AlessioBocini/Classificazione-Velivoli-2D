import torch.nn as nn
import torch
def run_epoch(loader, model, optimizer, criterion, device='cpu', train=True):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        # Iterate over batches
        # xb is the features, yb the true labels
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)                                                  # which is calling model.forward(xb)
            loss = criterion(logits, yb)                                        # it calculates the loss over the batch

            if train:
                optimizer.zero_grad(set_to_none=True)                           # clear gradients from the previous step
                loss.backward()                                                 # compute gradients via backpropagation
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)      # gradient clipping to avoid exploding gradients
                optimizer.step()                                                # update model parameters using the optimizer   

            with torch.no_grad():
                preds = logits.argmax(dim=1)                                    # predicted class labels
                correct += (preds == yb).sum().item()                           # count correct predictions
                total += yb.numel()                                             # total number of samples
                loss_sum += loss.item() * yb.numel()                            # accumulate loss

        acc = correct / max(1,total)                                            # accuracy over the epoch
        avg_loss = loss_sum / max(1,total)                                      # average loss over the epoch
        return avg_loss, acc