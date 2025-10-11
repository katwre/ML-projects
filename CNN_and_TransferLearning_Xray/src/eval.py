# src/eval.py
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report


# dual import style (works from notebook + CLI)
try:
    from data import make_loaders
    from model import create_model
except ImportError:
    from src.data import make_loaders
    from src.model import create_model


# plotting helpers 
def plot_prob_hist(y_prob, out_path: Path):
    plt.figure(figsize=(6,4))
    plt.hist(y_prob, bins=40)
    plt.title("Distribution of predicted probs")
    plt.xlabel("Predicted probability for PNEUMONIA")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _confmat(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["NORMAL","PNEUMONIA"])
    disp.plot(ax=ax, colorbar=True, cmap="viridis", values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

def plot_confmats(y_true, y_prob, thresholds, out_path: Path):
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, thresholds):
        y_pred = (y_prob >= t).astype(int)
        _confmat(ax, y_true, y_pred, f"Confusion Matrix @ {t:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc(y_true, y_prob, out_path: Path, title="ROC — CNN"):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_loss_history(hist_csv: Path, out_path: Path):
    if not hist_csv.exists():
        print(f"[warn] {hist_csv} not found — skipping loss plot.")
        return
    data = np.genfromtxt(hist_csv, delimiter=",", names=True)
    epochs = data["epoch"]
    tr = data["train_loss"]
    vl = data["val_loss"]
    best_epoch = int(epochs[np.argmin(vl)])

    plt.figure(figsize=(7,4))
    plt.plot(epochs, tr, label="train")
    plt.plot(epochs, vl, label="validation")
    plt.axvline(best_epoch, linestyle="--", label="Early Stopping Checkpoint")
    plt.xlabel("epoch"); plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(cfg_path: str = "config.yaml", ckpt_path: str | None = None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we evaluate on the same split used as val in the training script
    _, val_loader, _ = make_loaders(
        data_dir=cfg["paths"]["data_dir"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )

    model = create_model(device)

    if ckpt_path is None:
        ckpt_path = Path(cfg["paths"]["out_dir"]) / "cnn_best.pt"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    criterion = nn.BCELoss()
    loss_sum, correct, total = 0.0, 0, 0
    # collect y_true and y_prob for metrics/plots
    y_true, y_prob = [], []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="test"):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)                 # [B,1] sigmoid
            probs = outputs.squeeze(1).cpu().numpy()
            y_prob.extend(probs.tolist())

            t = targets.squeeze(1).cpu().numpy()
            y_true.extend(t.tolist())

            loss = criterion(outputs, targets)
            loss_sum += loss.item() * images.size(0)

            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.numel()

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    print({"test_loss": loss_sum / total, "test_acc": correct / total}, flush=True)

    # compute a secondary threshold (Youden's J) like your notebook
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden_idx = np.argmax(tpr - fpr)
    best_thr = float(thr[youden_idx]) if youden_idx < len(thr) else 0.5
    print(f"[info] AUC={roc_auc_score(y_true, y_prob):.3f} | Youden threshold={best_thr:.3f}", flush=True)

    # classification report at 0.5 (optional)
    rep = classification_report(y_true, (y_prob >= 0.5).astype(int), target_names=["NORMAL","PNEUMONIA"], digits=3)
    print("\n=== Validation @ threshold 0.5 ===\n" + rep)

    # make and save the figures
    plot_prob_hist(y_prob, out_dir / "probs_hist.png")
    plot_confmats(y_true, y_prob, thresholds=[0.5, best_thr], out_path=out_dir / "confmats.png")
    plot_roc(y_true, y_prob, out_path=out_dir / "roc.png", title="ROC — CNN (Validation)")
    plot_loss_history(out_dir / "loss_history.csv", out_path=out_dir / "loss_curve.png")
    print("[saved]",
          out_dir / "probs_hist.png",
          out_dir / "confmats.png",
          out_dir / "roc.png",
          out_dir / "loss_curve.png")
    


if __name__ == "__main__":
    main()
