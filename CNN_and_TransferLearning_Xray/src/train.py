# src/train.py
import yaml
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.data import make_loaders
from src.model import create_model, build_model


def _merge_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Shallow merge with one-level nested dict support; simple and readable."""
    if not overrides:
        return cfg
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)  # targets: [B,1] float {0,1}
        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)  # [B,1], already sigmoid
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.numel()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, desc="val"):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return loss_sum / total, correct / total


def main(cfg_path: str = "config.yaml", 
         ckpt_filename: str = "cnn_best.pt", 
         overrides: dict | None = None,
         loss_history_filename: str = "loss_history.csv"):
    
    # 1) load config (+ optional overrides)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _merge_overrides(cfg, overrides)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) data
    train_loader, val_loader, _ = make_loaders(
        data_dir=cfg["paths"]["data_dir"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )

    print(f"[train] cfg={cfg_path} data_dir={cfg['paths']['data_dir']} device={device}", flush=True)
    print(f"[train] batches: train={len(train_loader)} val={len(val_loader)}", flush=True)

    sample_imgs, _ = next(iter(train_loader))
    h, w = sample_imgs.shape[-2:]
    if h != w:
        raise ValueError(f"Expected square images, got {h}x{w}")
    input_size = int(h) # config file's input_size ss not gonna be overwritten here

    # 3) model + optim
    #model = create_model(device, input_size) # old version with only CNN
    # --- add to config so build_model() sees it ---
    cfg.setdefault("model", {})
    cfg["model"]["input_size"] = input_size  # pass to the builder (cnn or tl)
    # --- build model ---
    model = build_model(cfg["model"], device)

    criterion = nn.BCELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    out_dir = Path(cfg["paths"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / ckpt_filename

    # 4) Training loop setup

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience = cfg["train"].get("early_stop_patience", 3)
    no_improve = 0

    hist_path = out_dir / loss_history_filename
    with open(hist_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, desc="val")

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}",
            flush=True,
        )

        # Early stopping logic
        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.", flush=True)
                break

        # Append epoch to CSV
        with open(hist_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{va_loss:.6f}\n")


    print(f"Training finished. Best val loss: {best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
