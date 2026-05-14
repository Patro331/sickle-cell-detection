"""
train.py
--------
Training loop with early stopping and learning rate scheduling
for the sickle cell detection pipeline.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter


def get_pos_weight(train_dataset, device: torch.device) -> torch.Tensor:
    """
    Compute the positive class weight for BCEWithLogitsLoss.

    Weight = count(negative) / count(positive)
    This penalises the model more heavily for missing positive (sickle cell) cases.

    Parameters
    ----------
    train_dataset : ImageFolder dataset
    device : torch.device

    Returns
    -------
    torch.Tensor of shape (1,)
    """
    class_counts = Counter(train_dataset.targets)
    neg_count    = class_counts[0]
    pos_count    = class_counts[1]
    return torch.tensor([neg_count / pos_count]).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one training epoch.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : loss function
    optimizer : torch.optim.Optimizer
    device    : torch.device

    Returns
    -------
    tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds        = (torch.sigmoid(outputs) >= 0.5).float()
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / len(loader), correct / total


def validate_one_epoch(model, loader, criterion, device):
    """
    Run one validation epoch.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : loss function
    device    : torch.device

    Returns
    -------
    tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs      = model(images)
            loss         = criterion(outputs, labels)
            running_loss += loss.item()
            preds        = (torch.sigmoid(outputs) >= 0.5).float()
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / len(loader), correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    model_name: str,
    save_dir: str,
    device: torch.device,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    patience: int = 10
) -> dict:
    """
    Full training loop with early stopping and learning rate scheduling.

    Saves the best model weights (lowest validation loss) to save_dir.

    Parameters
    ----------
    model         : nn.Module
    train_loader  : DataLoader
    val_loader    : DataLoader
    train_dataset : ImageFolder — used to compute pos_weight
    model_name    : str — used for saving weights file
    save_dir      : str — directory to save model weights
    device        : torch.device
    learning_rate : float — Adam learning rate (default 1e-4)
    epochs        : int — maximum training epochs (default 50)
    patience      : int — early stopping patience (default 10)

    Returns
    -------
    dict with keys: 'history', 'best_val_loss', 'save_path'
        history contains lists: train_loss, val_loss, train_acc, val_acc
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_best.pt")

    pos_weight = get_pos_weight(train_dataset, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss     = float("inf")
    patience_counter  = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    print(f"\nTraining {model_name} | lr={learning_rate} | patience={patience}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'Status':>12}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate_one_epoch(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            status = "saved ✓"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{patience}"

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>10.4f} {status:>12}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    return {
        "history":       history,
        "best_val_loss": best_val_loss,
        "save_path":     save_path
    }
