import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for *model_inputs, score in train_loader:
        # Move inputs and score to GPU
        model_inputs = [input_.to(device, non_blocking=True) for input_ in model_inputs]
        score = score.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(*model_inputs)  # Unpack the model inputs
        loss = loss_fn(logits, score)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for *model_inputs, score in val_loader:
            # Move inputs and score to GPU
            model_inputs = [input_.to(device, non_blocking=True) for input_ in model_inputs]
            score = score.to(device, non_blocking=True)

            logits = model(*model_inputs)  # Unpack the model inputs
            loss = loss_fn(logits, score)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def get_optimizer_scheduler(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Reduce patience for scheduler to trigger earlier when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
    return optimizer, scheduler
