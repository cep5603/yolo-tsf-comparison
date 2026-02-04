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
from stl_decomp_bias_model import YOLO11_TSF as STLDecompBiasModel
from shape_aware_loss_model import YOLO11_TSF as ShapeAwareLossModel
from ultralytics.nn.tasks import ForecastModel as _ForecastModel

# CLI
parser = argparse.ArgumentParser(description="Train and compare YOLO11-TSF model variants on ETTh1")
parser.add_argument("--no-skip", action="store_true", help="Disable delta+skip connection (use direct output)")
parser.add_argument("--no-revin", action="store_true", help="Disable RevIN (Reversible Instance Normalization)")
parser.add_argument("--patch-len", type=int, default=16, help="Patch length for PatchTST embedding")
parser.add_argument("--patch-stride", type=int, default=8, help="Patch stride for PatchTST embedding")
parser.add_argument("--stl-loss", action="store_true", help="Enable STL-style component loss for v4/v5")
parser.add_argument("--stl-low-frac", type=float, default=0.05, help="Low-frequency fraction for STL spectral split")
parser.add_argument("--stl-mid-frac", type=float, default=0.2, help="Mid-frequency fraction for STL spectral split")
parser.add_argument("--stl-trend-weight", type=float, default=0.25, help="Weight for trend component loss (p5) when no learnable weight is available")
# parser.add_argument("--mix-ratio", type=float, default=0.0, help="Ratio of train dataset to mix into training")
args = parser.parse_args()

USE_SKIP = not args.no_skip
USE_REVIN = not args.no_revin
PATCH_LEN = args.patch_len
PATCH_STRIDE = args.patch_stride
USE_STL_LOSS = args.stl_loss
STL_LOW_FRAC = args.stl_low_frac
STL_MID_FRAC = args.stl_mid_frac
STL_TREND_WEIGHT = args.stl_trend_weight
# MIX_RATIO = args.mix_ratio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
CSV_PATH = "../ETTh1.csv"
WINDOW = 512#96
HORIZON = 24
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TRAIN_STRIDE = 8
EPOCHS = 1000#30#50
LR = 3e-4#1e-3
BATCH_SIZE = 64#32
PATIENCE = 9999#15 <-- train to max for now to see behavior
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
            if self.use_skip:
                out = out * self.std.squeeze(-1)
            else:
                out = out * self.std.squeeze(-1) + self.mean.squeeze(-1)
        
        if self.use_skip and last_val is not None:
            out = out + last_val.squeeze(1)
        
        return out


MODEL_CONFIGS = {
    "YOLO11 Forecast": {"class": ForecastModelWrapper, "color": "purple"},
    "v1 - Backbone Only": {"class": BackboneOnlyModel, "color": "orangered"},
    "v2 - Backbone + Neck": {"class": BackboneNeckModel, "color": "dodgerblue"},
    "v3 - Full (Multiscale)": {"class": FullModel, "color": "mediumseagreen"},
    "v4 - STL Decomp Bias": {"class": STLDecompBiasModel, "color": "darkorange", "supports_stl": True},
    "v5 - Shape-Aware Loss": {"class": ShapeAwareLossModel, "color": "hotpink", "supports_stl": True},
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


def spectral_decompose_batch(y, low_frac=0.05, mid_frac=0.2):
    """
    Spectral split into low/mid/high frequency components.
    Args:
        y: (batch, horizon)
    Returns:
        trend (low), seasonal (mid), resid (high)
    """
    spectrum = torch.fft.rfft(y, dim=-1)
    bins = spectrum.shape[-1]

    low = spectrum.clone()
    mid = spectrum.clone()
    high = spectrum.clone()

    low[..., int(low_frac * bins) :] = 0
    mid[..., : int(low_frac * bins)] = 0
    mid[..., int(mid_frac * bins) :] = 0
    high[..., : int(mid_frac * bins)] = 0

    trend = torch.fft.irfft(low, n=y.shape[-1], dim=-1)
    seasonal = torch.fft.irfft(mid, n=y.shape[-1], dim=-1)
    resid = torch.fft.irfft(high, n=y.shape[-1], dim=-1)
    return trend, seasonal, resid


# Training
def train_model(
    model,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    use_stl_loss=False,
    stl_low_frac=0.05,
    stl_mid_frac=0.2,
    stl_trend_weight=0.25,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # AdamW with weight decay for regularization
    loss_fn = nn.MSELoss()
    eps = 1e-5
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)  # Cosine annealing LR scheduler
    
    best_val_loss_norm = float("inf")
    best_val_loss_raw = float("inf")
    best_weights = None
    no_improve_epochs = 0
    history = {"train_norm": [], "val_norm": [], "train_raw": [], "val_raw": []}

    # Initialize lazy modules before any state_dict cloning
    model.train()
    with torch.no_grad():
        _ = model(X_train_t[:1])
    
    pbar = tqdm(range(epochs), desc="Training", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        epoch_loss_raw = 0.0
        n_batches = 0
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            if use_stl_loss:
                pred, pred_p3, pred_p4, pred_p5 = model(X_batch, return_components=True)
                loss_raw = loss_fn(pred, y_batch)

                trend, seasonal, resid = spectral_decompose_batch(
                    y_batch,
                    low_frac=stl_low_frac,
                    mid_frac=stl_mid_frac,
                )
                loss_p3 = loss_fn(pred_p3, resid)
                loss_p4 = loss_fn(pred_p4, seasonal)
                loss_p5 = loss_fn(pred_p5, trend)

                if hasattr(model, "log_sigma_p5"):
                    p5_weight = torch.exp(-model.log_sigma_p5)
                else:
                    p5_weight = stl_trend_weight

                loss = loss_raw + loss_p3 + loss_p4 + p5_weight * loss_p5

                if USE_REVIN:
                    mean = X_batch[:, :, -1:] if USE_SKIP else X_batch.mean(dim=-1, keepdim=True)
                    std = (X_batch.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                    pred_n = (pred - mean.squeeze(-1)) / std.squeeze(-1)
                    y_batch_n = (y_batch - mean.squeeze(-1)) / std.squeeze(-1)
                    loss_log = loss_fn(pred_n, y_batch_n)
                else:
                    loss_log = loss_raw
            else:
                pred = model(X_batch)
                loss_raw = loss_fn(pred, y_batch)
                if USE_REVIN:
                    mean = X_batch[:, :, -1:] if USE_SKIP else X_batch.mean(dim=-1, keepdim=True)
                    std = (X_batch.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                    pred = (pred - mean.squeeze(-1)) / std.squeeze(-1)
                    y_batch = (y_batch - mean.squeeze(-1)) / std.squeeze(-1)
                loss = loss_fn(pred, y_batch)
                loss_log = loss

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            
            optimizer.step()
            
            epoch_loss += loss_log.item()
            epoch_loss_raw += loss_raw.item()
            n_batches += 1
        
        scheduler.step()  # Step scheduler after each epoch
        
        avg_train_loss = epoch_loss / max(1, n_batches)
        avg_train_loss_raw = epoch_loss_raw / max(1, n_batches)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss_raw = loss_fn(val_pred, y_val_t).item()
            if USE_REVIN:
                mean = X_val_t[:, :, -1:] if USE_SKIP else X_val_t.mean(dim=-1, keepdim=True)
                std = (X_val_t.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                val_pred = (val_pred - mean.squeeze(-1)) / std.squeeze(-1)
                y_val_n = (y_val_t - mean.squeeze(-1)) / std.squeeze(-1)
                val_loss_norm = loss_fn(val_pred, y_val_n).item()
            else:
                val_loss_norm = val_loss_raw
        
        history["train_norm"].append(avg_train_loss)
        history["val_norm"].append(val_loss_norm)
        history["train_raw"].append(avg_train_loss_raw)
        history["val_raw"].append(val_loss_raw)
        
        if val_loss_norm < best_val_loss_norm:
            best_val_loss_norm = val_loss_norm
            best_val_loss_raw = val_loss_raw
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= PATIENCE:
            break
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{val_loss_norm:.4f}", best=f"{best_val_loss_norm:.4f}", lr=f"{current_lr:.1e}")
    
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_val_loss_norm, best_val_loss_raw, history


def compute_metrics(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean().item()
    mae = (y_true - y_pred).abs().mean().item()
    return mse, mae


def compute_normalized_metrics(X, y_true, y_pred, eps=1e-5):
    mean = X[:, :, -1:] if USE_SKIP else X.mean(dim=-1, keepdim=True)
    std = (X.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
    y_pred_n = (y_pred - mean.squeeze(-1)) / std.squeeze(-1)
    y_true_n = (y_true - mean.squeeze(-1)) / std.squeeze(-1)
    return compute_metrics(y_true_n, y_pred_n)


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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    
    for name, (history, color) in all_histories.items():
        ax1.plot(history["train_norm"], label=name, color=color, lw=2)
        ax2.plot(history["val_norm"], label=name, color=color, lw=2)
        ax3.plot(history["train_raw"], label=name, color=color, lw=2)
        ax4.plot(history["val_raw"], label=name, color=color, lw=2)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss (Normalized)")
    ax1.set_title("Training Loss (Normalized)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss (Normalized)")
    ax2.set_title("Validation Loss (Normalized)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train Loss (Denormalized)")
    ax3.set_title("Training Loss (Denormalized)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Val Loss (Denormalized)")
    ax4.set_title("Validation Loss (Denormalized)", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# Data prep
print("Loading ETTh1...")
ot = load_etth1(CSV_PATH)
print(f"Total samples: {len(ot)}")

T = len(ot)
t_train_end = int(T * TRAIN_RATIO)
t_val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train_series = ot[:t_train_end]
val_series = ot[t_train_end - WINDOW : t_val_end]
test_series = ot[t_val_end - WINDOW :]

X_train, y_train = create_windows(train_series, WINDOW, HORIZON, stride=TRAIN_STRIDE)
X_val, y_val = create_windows(val_series, WINDOW, HORIZON, stride=1)
X_test, y_test = create_windows(test_series, WINDOW, HORIZON, stride=1)

print(f"Windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} (window={WINDOW}, horizon={HORIZON})")

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
print(f"TRAINING ALL MODELS (use_skip={USE_SKIP}, use_revin={USE_REVIN}, patch_len={PATCH_LEN}, patch_stride={PATCH_STRIDE}), stl_loss={USE_STL_LOSS}")
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
    use_stl_loss = USE_STL_LOSS and cfg.get("supports_stl", False)
    if use_stl_loss:
        print(f"Using STL loss (low_frac={STL_LOW_FRAC}, mid_frac={STL_MID_FRAC}, trend_weight={STL_TREND_WEIGHT})")

    best_val_loss_norm, best_val_loss_raw, history = train_model(
        model,
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        use_stl_loss=use_stl_loss,
        stl_low_frac=STL_LOW_FRAC,
        stl_mid_frac=STL_MID_FRAC,
        stl_trend_weight=STL_TREND_WEIGHT,
    )
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t)
    
    test_mse_raw, test_mae_raw = compute_metrics(y_test_t, y_pred)
    if USE_REVIN:
        test_mse_norm, test_mae_norm = compute_normalized_metrics(X_test_t, y_test_t, y_pred)
    else:
        test_mse_norm, test_mae_norm = test_mse_raw, test_mae_raw
    
    results[name] = {
        "best_val_loss_norm": best_val_loss_norm,
        "best_val_loss_raw": best_val_loss_raw,
        "test_mse_raw": test_mse_raw,
        "test_mae_raw": test_mae_raw,
        "test_mse_norm": test_mse_norm,
        "test_mae_norm": test_mae_norm,
    }
    predictions[name] = (y_pred, cfg["color"])
    all_histories[name] = (history, cfg["color"])
    
    print(f"  Best Val Loss (Normalized): {best_val_loss_norm:.4f}")
    print(f"  Best Val Loss (Denormalized): {best_val_loss_raw:.4f}")
    print(f"  Test MSE (Denormalized): {test_mse_raw:.4f} | Test MAE (Denormalized): {test_mae_raw:.4f}")
    print(f"  Test MSE (Normalized): {test_mse_norm:.4f} | Test MAE (Normalized): {test_mae_norm:.4f}")


# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<25} {'Val(N)':>12} {'Val(D)':>12} {'TestMSE(D)':>12} {'TestMAE(D)':>12} {'TestMSE(N)':>12} {'TestMAE(N)':>12}")
print("-" * 70)

for name, metrics in results.items():
    print(
        f"{name:<25}"
        f" {metrics['best_val_loss_norm']:>12.4f}"
        f" {metrics['best_val_loss_raw']:>12.4f}"
        f" {metrics['test_mse_raw']:>12.4f}"
        f" {metrics['test_mae_raw']:>12.4f}"
        f" {metrics['test_mse_norm']:>12.4f}"
        f" {metrics['test_mae_norm']:>12.4f}"
    )

best_model = min(results, key=lambda x: results[x]["test_mse_raw"])
print("-" * 70)
print(f"Best (by Test MSE (Denormalized)): {best_model}")

plot_loss_curves(all_histories)
plot_comparison(X_test_t, y_test_t, predictions)