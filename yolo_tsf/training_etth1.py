import argparse
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

sys.path.insert(0, "..")  # Add ultralytics to path for ForecastModel import

from backbone_model import YOLO11_TSF as BackboneOnlyModel
from backbone_neck_model import YOLO11_TSF as BackboneNeckModel
from backbone_neck_branch_specific_aggregation_model import YOLO11_TSF as FullModel
from ultralytics.nn.tasks import ForecastModel as _ForecastModel

# CLI
parser = argparse.ArgumentParser(description="Train and compare YOLO11-TSF model variants on ETTh1")
parser.add_argument("--no-skip", action="store_true", help="Disable delta+skip connection (use direct output)")
parser.add_argument("--no-revin", action="store_true", help="Disable RevIN (Reversible Instance Normalization)")
parser.add_argument("--patch-len", type=int, default=16, help="Patch length for PatchTST embedding")
parser.add_argument("--patch-stride", type=int, default=8, help="Patch stride for PatchTST embedding")
# parser.add_argument("--mix-ratio", type=float, default=0.0, help="Ratio of train dataset to mix into training")
args = parser.parse_args()

USE_SKIP = not args.no_skip
USE_REVIN = not args.no_revin
PATCH_LEN = args.patch_len
PATCH_STRIDE = args.patch_stride
# MIX_RATIO = args.mix_ratio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
CSV_PATH = "../ETTh1.csv"
WINDOW = 512#96
HORIZON = 24
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
EPOCHS = 100#30#50
LR = 3e-4#1e-3
BATCH_SIZE = 64#32
MAX_SAMPLES = 4  # For stacked viz plot


class ForecastModelWrapper(nn.Module):
    """
    Wrapper around ForecastModel for to use w/ this script
    - Reshapes input from (B, 1, L) to (B, 1, 1, L)
    - Uses internal RevIN normalization
    - Overrides horizon in YAML
    """
    
    def __init__(self, horizon=24, use_skip=False, use_revin=True, seq_len=96, patch_len=16, patch_stride=8):
        super().__init__()
        self.horizon = horizon
        self.use_revin = use_revin
        self.use_skip = use_skip  # Unused here
        
        # RevIN stats stored during forward (for denorm)
        self.mean = None
        self.std = None
        self.eps = 1e-5
        
        yaml_path = "../ultralytics/cfg/models/11/yolo11-forecast.yaml"
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Override horizon in Forecast head (last item in head)
        # Format: [[14, 11, 8], 1, Forecast, [horizon, quantiles, hidden, dropout, agg, pool_size]]
        cfg["head"][-1][-1][0] = horizon
        
        self.model = _ForecastModel(cfg=cfg, ch=1, verbose=False)
    
    def forward(self, x):
        # x: (B, 1, L)
        last_val = x[:, :, -1:] if self.use_skip else None
        
        # RevIN normalize
        if self.use_revin:
            self.mean = x.mean(dim=-1, keepdim=True).detach()  # (B, 1, 1)
            self.std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.std
        
        # Pad input to next multiple of 64 for FPN compatibility
        L = x.shape[-1]
        target_L = ((L + 63) // 64) * 64  # Round up to next multiple of 64
        if target_L > L:
            pad_len = target_L - L
            # Replicate last value for padding (like PatchTST)
            pad = x[:, :, -1:].expand(-1, -1, pad_len)
            x = torch.cat([x, pad], dim=-1)
        
        # Reshape: (B, 1, L) -> (B, 1, 1, L) for ForecastModel
        x = x.unsqueeze(2)  # (B, 1, 1, L)
        
        # Forward through YOLO11 ForecastModel
        out = self.model(x)  # (B, H)
        
        # RevIN denormalize
        if self.use_revin:
            # mean/std: (B, 1, 1), out: (B, H)
            out = out * self.std.squeeze(-1) + self.mean.squeeze(-1)
        
        if self.use_skip and last_val is not None:
            out = out + last_val.squeeze(1)
        
        return out


MODEL_CONFIGS = {
    "YOLO11 Forecast": {"class": ForecastModelWrapper, "color": "purple"},
    "v1 - Backbone Only": {"class": BackboneOnlyModel, "color": "orangered"},
    "v2 - Backbone + Neck": {"class": BackboneNeckModel, "color": "dodgerblue"},
    "v3 - Full (Multiscale)": {"class": FullModel, "color": "mediumseagreen"},
}


# Data loading
def load_etth1(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ot = df["OT"].values.astype(np.float32)
    return ot


def create_windows(data, window, horizon, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window - horizon + 1, stride):
        X.append(data[i : i + window])
        y.append(data[i + window : i + window + horizon])
    return np.array(X), np.array(y)


# Training
def train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # AdamW with weight decay for regularization
    loss_fn = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)  # Cosine annealing LR scheduler
    
    best_val_loss = float("inf")
    best_weights = None
    history = {"train": [], "val": []}
    
    pbar = tqdm(range(epochs), desc="Training", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()  # Step scheduler after each epoch
        
        avg_train_loss = epoch_loss / n_batches
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
        
        history["train"].append(avg_train_loss)
        history["val"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{val_loss:.4f}", best=f"{best_val_loss:.4f}", lr=f"{current_lr:.1e}")
    
    model.load_state_dict(best_weights)
    return best_val_loss, history


def compute_metrics(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean().item()
    mae = (y_true - y_pred).abs().mean().item()
    return mse, mae


# Visualization
def plot_comparison(X_test, y_test, predictions, save_path="etth1_comparison.png", max_samples=MAX_SAMPLES):
    X_np = X_test.squeeze(1).cpu().numpy()
    y_true_np = y_test.cpu().numpy()
    
    # Evenly spread-out indices
    total_samples = len(X_np)
    n = min(total_samples, max_samples)
    sample_indices = np.linspace(0, total_samples - 1, n, dtype=int)
    
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = axes[plot_idx]
        input_len = len(X_np[sample_idx])
        horizon = len(y_true_np[sample_idx])
        
        t_input = np.arange(input_len)
        t_forecast = np.arange(input_len - 1, input_len + horizon)
        
        # Input (last 50 points)
        ax.plot(t_input[-50:], X_np[sample_idx][-50:], "k-", alpha=0.5, lw=1.5, label="Input (last 50)")
        
        # Ground truth
        true_line = np.concatenate([[X_np[sample_idx][-1]], y_true_np[sample_idx]])
        ax.plot(t_forecast, true_line, "k-", lw=2.5, label="Ground Truth")
        
        # Each model prediction
        for name, (y_pred, color) in predictions.items():
            pred_np = y_pred.cpu().numpy()
            pred_line = np.concatenate([[X_np[sample_idx][-1]], pred_np[sample_idx]])
            ax.plot(t_forecast, pred_line, "--", color=color, lw=2, label=name, alpha=0.85)
        
        ax.axvline(x=input_len - 1, color="gray", ls=":", alpha=0.4)
        ax.set_title(f"Sample {sample_idx+1} (Test Index {sample_idx})", fontweight="bold")
        ax.set_ylabel("OT")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_loss_curves(all_histories, save_path="etth1_loss_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, (history, color) in all_histories.items():
        ax1.plot(history["train"], label=name, color=color, lw=2)
        ax2.plot(history["val"], label=name, color=color, lw=2)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# Data prep
print("Loading ETTh1...")
ot = load_etth1(CSV_PATH)
print(f"Total samples: {len(ot)}")

X_all, y_all = create_windows(ot, WINDOW, HORIZON, stride=1)
print(f"Windows: {len(X_all)} (window={WINDOW}, horizon={HORIZON})")

n = len(X_all)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
X_test, y_test = X_all[val_end:], y_all[val_end:]

# Try to add extra random samples from training region for more variety
# At 0.2, this worsened MSE but pred vs. ground truth looked a bit closer
# if MIX_RATIO > 0:
#     n_mix = int(train_end * MIX_RATIO)  # Percentage of training set, not full dataset
#     mix_indices = np.random.choice(train_end, size=n_mix, replace=True)  # Sample from train only
#     X_mix, y_mix = X_all[mix_indices], y_all[mix_indices]
#     X_train = np.concatenate([X_train, X_mix], axis=0)
#     y_train = np.concatenate([y_train, y_mix], axis=0)
#     print(f"Safe mixed training: added {n_mix} samples from train region (total: {len(X_train)})")

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Device: {DEVICE}")

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)


# Train + evaluate all models
results = {}
predictions = {}
all_histories = {}

print("\n" + "=" * 70)
print(f"TRAINING ALL MODELS (use_skip={USE_SKIP}, use_revin={USE_REVIN}, patch_len={PATCH_LEN}, patch_stride={PATCH_STRIDE})")
print("=" * 70)

for name, cfg in MODEL_CONFIGS.items():
    print(f"\nTraining: {name}")
    print("-" * 40)
    
    model = cfg["class"](
        horizon=HORIZON,
        use_skip=USE_SKIP,
        use_revin=USE_REVIN,
        seq_len=WINDOW,
        patch_len=PATCH_LEN,
        patch_stride=PATCH_STRIDE
    ).to(DEVICE)
    best_val_loss, history = train_model(model, X_train_t, y_train_t, X_val_t, y_val_t)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t)
    
    mse, mae = compute_metrics(y_test_t, y_pred)
    
    results[name] = {"best_val_loss": best_val_loss, "test_mse": mse, "test_mae": mae}
    predictions[name] = (y_pred, cfg["color"])
    all_histories[name] = (history, cfg["color"])
    
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Test MSE: {mse:.4f} | Test MAE: {mae:.4f}")


# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<25} {'Val Loss':>12} {'Test MSE':>12} {'Test MAE':>12}")
print("-" * 70)

for name, metrics in results.items():
    print(f"{name:<25} {metrics['best_val_loss']:>12.4f} {metrics['test_mse']:>12.4f} {metrics['test_mae']:>12.4f}")

best_model = min(results, key=lambda x: results[x]["test_mse"])
print("-" * 70)
print(f"Best (by Test MSE): {best_model}")

plot_loss_curves(all_histories)
plot_comparison(X_test_t, y_test_t, predictions)