import argparse
import json
import os
import sys
import importlib.util
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from plots import plot_comparison, plot_loss_curves, plot_run_variance
import yaml
from tqdm import tqdm

sys.path.insert(0, "..")  # Add ultralytics to path for ForecastModel import

from backbone_model import YOLO11_TSF as BackboneOnlyModel
from backbone_neck_model import YOLO11_TSF as BackboneNeckModel
from backbone_neck_branch_specific_aggregation_model import YOLO11_TSF as FullModel
from spectral_decomp_bias_model import YOLO11_TSF as SpectralDecompBiasModel
from shape_aware_loss_model import YOLO11_TSF as ShapeAwareLossModel
from temporal_gating_model import YOLO11_TSF as TemporalGatingModel
from multivariate_context_model import YOLO11_TSF as MultivariateContextModel
from ultralytics.nn.tasks import ForecastModel as _ForecastModel

# CLI
parser = argparse.ArgumentParser(description="Train and compare YOLO11-TSF model variants on time series datasets")
parser.add_argument(
    "--dataset",
    type=str,
    default="etth1",
    choices=["etth1", "etth2", "ettm2", "ili", "weather", "exchange_rate"],
    help="Dataset to train/evaluate on",
)
parser.add_argument("--csv-path", type=str, default=None, help="Optional override path for dataset CSV")
parser.add_argument("--window", type=int, default=None, help="Optional override lookback window length")
parser.add_argument("--horizon", type=int, default=None, help="Optional override forecast horizon length")
parser.add_argument("--train-stride", type=int, default=None, help="Optional override train window stride")
parser.add_argument("--no-skip", action="store_true", help="Disable delta+skip connection (use direct output)")
parser.add_argument("--no-revin", action="store_true", help="Disable RevIN (Reversible Instance Normalization)")
parser.add_argument("--patch-len", type=int, default=16, help="Patch length for PatchTST embedding")
parser.add_argument("--patch-stride", type=int, default=8, help="Patch stride for PatchTST embedding")
parser.add_argument("--use-last-epoch", action="store_true", help="Report results using the last epoch model instead of the best epoch model")
parser.add_argument(
    "--eval-protocol",
    type=str,
    default="legacy",
    choices=["legacy", "patchtst"],
    help="Metric/split protocol: 'legacy' keeps current behavior, 'patchtst' uses PatchTST ETTh1 split + train-set standardization",
)
parser.add_argument(
    "--features",
    type=str,
    default=None,
    choices=["S", "M"],
    help="Forecast feature mode: S=univariate target-only, M=multivariate all non-date columns",
)
parser.add_argument("--band-split", nargs=3, type=int, default=[10, 80, 10], help="Frequency band split percentages [low, mid, high] (e.g. 10 80 10)")
parser.add_argument("--decomp-mode", type=str, default="moving_avg", choices=["moving_avg", "fft"], help="Auxiliary component decomposition mode")
parser.add_argument("--trend-kernel", type=int, default=0, help="Moving-average trend kernel for --decomp-mode moving_avg (0 uses horizon//4)")
parser.add_argument("--seasonal-kernel", type=int, default=0, help="Moving-average seasonal smoothing kernel for --decomp-mode moving_avg (0 uses trend_kernel//4; keep below half the dominant period so the cycle stays in the seasonal band)")
parser.add_argument("--level-loss-weight", type=float, default=0.5, help="Weight for horizon-mean level consistency loss")
parser.add_argument("--target-loss-weight", type=float, default=1.0, help="Extra loss weight for target column in FEATURES=M channel-independent YOLO training")
parser.add_argument("--monitor-target-channel", action="store_true", help="Select best epoch using target-column validation MSE instead of all-channel validation MSE")
parser.add_argument("--repeats", type=int, default=1, help="Number of repeated training runs per model (each with a different seed)")
parser.add_argument("--no-ridge-init", action="store_true", help="Disable closed-form ridge initialization of models' direct linear forecast path")
parser.add_argument("--freeze-direct", action="store_true", help="Freeze the ridge-initialized direct linear path so SGD trains only the deep residual")
parser.add_argument("--eval-batch-size", type=int, default=4096, help="Batch size for validation/test inference (bounds peak VRAM; lower if you hit CUDA OOM during eval)")
parser.add_argument("--spectral-trend-weight", type=float, default=0.25, help="Weight for spectral trend component loss (p5) when no learnable weight is available")
parser.add_argument("--spectral-warmup-epochs", type=int, default=10, help="Linearly ramp spectral auxiliary losses from 0 to full weight over this many initial epochs (0 disables warmup)")
parser.add_argument(
    "--shape-slope-weight",
    type=float,
    default=0.25,
    help="Additional slope-consistency loss weight for v5 trend head",
)
parser.add_argument(
    "--shape-curvature-weight",
    type=float,
    default=0.10,
    help="Additional curvature-consistency loss weight for v5 trend head",
)
args = parser.parse_args()

PATCH_LEN_SET_BY_USER = "--patch-len" in sys.argv
PATCH_STRIDE_SET_BY_USER = "--patch-stride" in sys.argv

USE_SKIP = not args.no_skip
USE_REVIN = not args.no_revin
PATCH_LEN = args.patch_len
PATCH_STRIDE = args.patch_stride
USE_LAST_EPOCH = args.use_last_epoch
EVAL_PROTOCOL = args.eval_protocol
TARGET_COL = "OT"
REPEATS = max(1, args.repeats)

DATASET_CONFIGS = {
    "etth1": {
        "csv_path": "../ETTh1.csv",
        "window": 720,
        "horizon": 336,
        "train_stride": 8,
        "label": "ETTh1",
        "output_prefix": "etth1",
    },
    "etth2": {
        "csv_path": "../ETTh2.csv",
        "window": 720,
        "horizon": 336,
        "train_stride": 8,
        "label": "ETTh2",
        "output_prefix": "etth2",
    },
    "ettm2": {
        "csv_path": "../ETTm2.csv",
        "window": 336,  # PatchTST seq_len
        "horizon": 96,
        "train_stride": 4,  # stride 1 matches PatchTST exactly but is ~4x slower
        "label": "ETTm2",
        "output_prefix": "ettm2",
    },
    "ili": {
        "csv_path": "../national_illness.csv",
        "window": 104,  # Same as PatchTST
        "horizon": 48,  # Highest that works rn
        "train_stride": 1,
        "label": "National Illness",
        "output_prefix": "ili",
    },
    "weather": {
        "csv_path": "../weather.csv",
        "window": 336,
        "horizon": 96,
        "train_stride": 1,
        "label": "Weather",
        "output_prefix": "weather",
    },
    "exchange_rate": {
        "csv_path": "../exchange_rate.csv",
        "window": 336,  # Same as PatchTST/DLinear seq_len
        "horizon": 96,
        "train_stride": 1,
        "label": "Exchange Rate",
        "output_prefix": "exchange_rate",
    },
}
dataset_cfg = DATASET_CONFIGS[args.dataset]

FEATURES = args.features
if FEATURES is None:
    FEATURES = "M" if (EVAL_PROTOCOL == "patchtst" and args.dataset in ("etth1", "etth2", "ettm2", "ili", "weather", "exchange_rate")) else "S"

# Calculate cutoffs from band split
total_split = sum(args.band_split)
SPECTRAL_LOW_FRAC = args.band_split[0] / total_split
SPECTRAL_MID_FRAC = (args.band_split[0] + args.band_split[1]) / total_split

SPECTRAL_TREND_WEIGHT = args.spectral_trend_weight
SPECTRAL_WARMUP_EPOCHS = max(0, args.spectral_warmup_epochs)
DECOMP_MODE = args.decomp_mode
TREND_KERNEL = max(0, args.trend_kernel)
SEASONAL_KERNEL = max(0, args.seasonal_kernel)
LEVEL_LOSS_WEIGHT = max(0.0, args.level_loss_weight)
TARGET_LOSS_WEIGHT = max(1.0, args.target_loss_weight)
MONITOR_TARGET_CHANNEL = args.monitor_target_channel
EVAL_BATCH_SIZE = max(1, args.eval_batch_size)
SHAPE_SLOPE_WEIGHT = max(0.0, args.shape_slope_weight)
SHAPE_CURVATURE_WEIGHT = max(0.0, args.shape_curvature_weight)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
CSV_PATH = args.csv_path if args.csv_path else dataset_cfg["csv_path"]
WINDOW = args.window if args.window is not None else dataset_cfg["window"]
HORIZON = args.horizon if args.horizon is not None else dataset_cfg["horizon"]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TRAIN_STRIDE = args.train_stride if args.train_stride is not None else dataset_cfg["train_stride"]

# PatchTST scripts use seq_len=336 and stride=1 for the hourly ETT datasets.
if EVAL_PROTOCOL == "patchtst" and args.dataset in ("etth1", "etth2"):
    if args.window is None:
        WINDOW = 336
    if args.train_stride is None:
        TRAIN_STRIDE = 1

EPOCHS = 50#30
LR = 1e-4#3e-4#1e-3
BATCH_SIZE = 64
PATIENCE = 9999#15 <-- train to max for now to see behavior
MAX_SAMPLES = 7  # For stacked viz plot

DATASET_LABEL = dataset_cfg["label"]
OUTPUT_PREFIX = dataset_cfg["output_prefix"]

SPECTRAL_P3_WEIGHT = 1.0  # residual / high-freq
SPECTRAL_P4_WEIGHT = 1.0  # seasonal / mid-freq
SPECTRAL_P5_WEIGHT = 1.0  # trend / low-freq

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


class LinearForecastModel(nn.Module):
    """NLinear/DLinear-style baseline: a single linear map on the RevIN-normalized window.

    Establishes the linear floor for this harness; literature (DLinear/NLinear) shows
    this is near state-of-the-art on ETTh1 long-horizon.
    """

    def __init__(self, horizon=24, use_skip=True, use_revin=True, seq_len=96, patch_len=16, patch_stride=8):
        super().__init__()
        self.horizon = horizon
        self.use_skip = use_skip
        self.use_revin = use_revin
        self.eps = 1e-5
        self.direct = nn.Linear(seq_len, horizon)

    def forward(self, x):
        # x: (B, 1, L)
        last_val = x[:, :, -1:] if self.use_skip else None
        if self.use_revin:
            mean = x.mean(dim=-1, keepdim=True)
            std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
            x = (x - mean) / std
        out = self.direct(x.squeeze(1))
        if self.use_revin:
            if self.use_skip:
                out = out * std.squeeze(-1)
            else:
                out = out * std.squeeze(-1) + mean.squeeze(-1)
        if self.use_skip and last_val is not None:
            out = out + last_val.squeeze(1)
        return out


_PATCHTST_MODEL_CLASS = None


def _load_patchtst_supervised_model_class():
    """Load PatchTST_supervised/models/PatchTST.py without clashing with yolo_tsf/layers.py."""
    global _PATCHTST_MODEL_CLASS
    if _PATCHTST_MODEL_CLASS is not None:
        return _PATCHTST_MODEL_CLASS

    patchtst_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PatchTST_supervised"))
    patchtst_model_file = os.path.join(patchtst_root, "models", "PatchTST.py")
    patchtst_layers_dir = os.path.join(patchtst_root, "layers")

    if not os.path.isfile(patchtst_model_file):
        raise FileNotFoundError(f"PatchTST model file not found: {patchtst_model_file}")
    if not os.path.isdir(patchtst_layers_dir):
        raise FileNotFoundError(f"PatchTST layers directory not found: {patchtst_layers_dir}")

    layers_mod = sys.modules.get("layers")
    if layers_mod is None:
        raise ImportError("Expected module 'layers' to be loaded before importing PatchTST_supervised.")

    had_layers_path = hasattr(layers_mod, "__path__")
    original_layers_path = getattr(layers_mod, "__path__", None)

    module_name = "_patchtst_supervised_model"
    spec = importlib.util.spec_from_file_location(module_name, patchtst_model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {patchtst_model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        # Temporarily make the already-loaded `layers` module package-like so
        # PatchTST_supervised absolute imports (layers.*) resolve correctly.
        layers_mod.__path__ = [patchtst_layers_dir]
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if had_layers_path:
            layers_mod.__path__ = original_layers_path
        elif hasattr(layers_mod, "__path__"):
            delattr(layers_mod, "__path__")

    if not hasattr(module, "Model"):
        raise ImportError("PatchTST_supervised Model class not found in models/PatchTST.py")

    _PATCHTST_MODEL_CLASS = module.Model
    return _PATCHTST_MODEL_CLASS


class PatchTSTSupervisedWrapper(nn.Module):
    """Adapter around the official PatchTST supervised forecasting model."""

    def __init__(self, horizon=24, use_skip=False, use_revin=True, seq_len=96, patch_len=16, patch_stride=8, n_vars=1,
                 d_model=128, n_heads=16, d_ff=256, e_layers=3, dropout=0.2, fc_dropout=0.2, head_dropout=0.0):
        super().__init__()
        self.seq_len = seq_len

        if use_skip:
            print("[PatchTST] use_skip is ignored to match official supervised PatchTST behavior.")

        configs = SimpleNamespace(
            enc_in=int(n_vars),
            seq_len=seq_len,
            pred_len=horizon,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            individual=False,
            patch_len=patch_len,
            stride=patch_stride,
            padding_patch="end",
            revin=bool(use_revin),
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )

        patchtst_model_class = _load_patchtst_supervised_model_class()
        self.model = patchtst_model_class(configs).float()

    def forward(self, x):
        # Preferred input layout: (B, seq_len, C). Allow (B, C, seq_len) for compatibility.
        if x.ndim != 3:
            raise ValueError(f"PatchTST expects a 3D tensor, got shape={tuple(x.shape)}")
        if x.shape[1] != self.seq_len and x.shape[2] == self.seq_len:
            x = x.transpose(1, 2)
        return self.model(x)


MODEL_CONFIGS = {
    "YOLO11 Forecast": {"class": ForecastModelWrapper, "color": "purple"},
    "Linear (Direct)": {"class": LinearForecastModel, "color": "slategray"},
    "PatchTST (Supervised)": {"class": PatchTSTSupervisedWrapper, "color": "seagreen", "trainer": "patchtst_supervised", "multivariate": True},
    # "v1 - Backbone Only": {"class": BackboneOnlyModel, "color": "orangered"},
    # "v2 - Backbone + Neck": {"class": BackboneNeckModel, "color": "dodgerblue"},
    # "v3 - Full (Multiscale)": {"class": FullModel, "color": "mediumseagreen"},
    # "v4 - Spectral Decomp Bias": {"class": SpectralDecompBiasModel, "color": "darkorange", "supports_spectral_loss": True},
    "v5 - Shape-Aware Loss": {"class": ShapeAwareLossModel, "color": "hotpink", "supports_spectral_loss": True, "force_spectral_loss": True, "shape_aware": True},  # MUST CHANGE force_spectral_loss if you want to test with spectral aux loss off (also disables shape-aware loss)
    "v6 - Temporal Gating": {"class": TemporalGatingModel, "color": "cyan", "supports_spectral_loss": True, "force_spectral_loss": True, "shape_aware": True},
    # "v7 - Multivariate Context": {"class": MultivariateContextModel, "color": "gold", "supports_spectral_loss": True, "force_spectral_loss": True, "shape_aware": True, "true_multivariate": True},
}

# Per-dataset PatchTST training recipes, matching official scripts in PatchTST_supervised/scripts/PatchTST/ and scripts/Linear/
# Applied only when --eval-protocol patchtst and model trainer == "patchtst_supervised"
PATCHTST_RECIPES = {
    "etth1": {
        "epochs": 100, "lr": 1e-4, "batch_size": 128, "patience": 100,
        "lradj": "type3", "pct_start": 0.3,
        "d_model": 16, "n_heads": 4, "d_ff": 128, "e_layers": 3,
        "dropout": 0.3, "fc_dropout": 0.3, "head_dropout": 0.0,
        "patch_len": 16, "stride": 8,
    },
    "ili": {
        "epochs": 100, "lr": 2.5e-3, "batch_size": 16, "patience": 100,
        "lradj": "constant",
        "d_model": 16, "n_heads": 4, "d_ff": 128, "e_layers": 3,
        "dropout": 0.3, "fc_dropout": 0.3, "head_dropout": 0.0,
        "patch_len": 24, "stride": 2,
    },
    "weather": {
        "epochs": 100, "lr": 1e-4, "batch_size": 128, "patience": 20,
        "lradj": "type3",
        "d_model": 128, "n_heads": 16, "d_ff": 256, "e_layers": 3,
        "dropout": 0.2, "fc_dropout": 0.2, "head_dropout": 0.0,
        "patch_len": 16, "stride": 8,
    },
    "exchange_rate": {
        # No official PatchTST script; aligned with DLinear script data processing and using PatchTST-style defaults for comparison
        "epochs": 100, "lr": 5e-4, "batch_size": 8, "patience": 100,
        "lradj": "type3",
        "d_model": 128, "n_heads": 16, "d_ff": 256, "e_layers": 3,
        "dropout": 0.2, "fc_dropout": 0.2, "head_dropout": 0.0,
        "patch_len": 16, "stride": 8,
    },
}


# Data loading
def load_time_series(csv_path, features="S", target_col="OT", time_col="date"):
    df = pd.read_csv(csv_path)

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)

    if features == "S":
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in {csv_path}. Available columns: {list(df.columns)}")
        data_cols = [target_col]
    else:
        data_cols = [c for c in df.columns if c != time_col]
        if not data_cols:
            raise ValueError(f"No usable feature columns found in {csv_path}.")
        # Match PatchTST Dataset_Custom column ordering: ['date'] + other_cols + [target]
        if target_col in data_cols and data_cols[-1] != target_col:
            data_cols = [c for c in data_cols if c != target_col] + [target_col]

    values_df = df[data_cols].apply(pd.to_numeric, errors="coerce")
    if values_df.isna().any().any():
        n_bad = int(values_df.isna().sum().sum())
        raise ValueError(f"Found {n_bad} non-numeric values in selected columns {data_cols} from {csv_path}")

    values = values_df.values.astype(np.float32)
    if values.ndim == 1:
        values = values[:, None]

    return values, data_cols


def create_windows(data, window, horizon, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window - horizon + 1, stride):
        X.append(data[i : i + window])
        y.append(data[i + window : i + window + horizon])
    return np.array(X), np.array(y)


def to_channel_independent_dataset(X, y):
    """Reshape (N, L, C)/(N, H, C) -> (N*C, 1, L)/(N*C, H)."""
    n_samples, window, n_vars = X.shape
    horizon = y.shape[1]
    X_ci = np.transpose(X, (0, 2, 1)).reshape(n_samples * n_vars, 1, window)
    y_ci = np.transpose(y, (0, 2, 1)).reshape(n_samples * n_vars, horizon)
    return X_ci, y_ci


def to_channel_independent_tensors(X, y):
    """Torch equivalent of to_channel_independent_dataset for training/eval batches.

    Supports both:
      - channel-independent layout: X=(N,1,L), y=(N,H)
      - multivariate layout:        X=(N,L,C), y=(N,H,C)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D, got shape={tuple(X.shape)}")

    if y.ndim == 2:
        if X.shape[1] != 1:
            raise ValueError(
                f"Channel-independent tensors must be X=(N,1,L), got shape={tuple(X.shape)}"
            )
        return X, y

    if y.ndim == 3:
        n_samples, window, n_vars = X.shape
        if y.shape[0] != n_samples or y.shape[2] != n_vars:
            raise ValueError(
                f"Mismatched multivariate shapes: X={tuple(X.shape)}, y={tuple(y.shape)}"
            )
        horizon = y.shape[1]
        X_ci = X.permute(0, 2, 1).reshape(n_samples * n_vars, 1, window)
        y_ci = y.permute(0, 2, 1).reshape(n_samples * n_vars, horizon)
        return X_ci, y_ci

    raise ValueError(f"Expected y to be 2D or 3D, got shape={tuple(y.shape)}")


def ridge_init_direct_path(model, X_train_t, y_train_t, lam=1.0, X_val_t=None, y_val_t=None):
    """Closed-form ridge fit of model.direct in the model's normalized target space.

    With RevIN (+ optional delta-skip), the direct path maps the per-window
    normalized input to the normalized (or last-value-anchored) target. Solving
    this linear regression in closed form lets the model start training at the
    linear optimum, so the deep pyramid only has to learn a residual correction.
    """
    if not hasattr(model, "direct") or not isinstance(model.direct, nn.Linear):
        return False
    if not USE_REVIN:
        print("[ridge-init] Skipped: requires RevIN-normalized direct path.")
        return False
    eps = 1e-5
    with torch.no_grad():
        X_ci, y_ci = to_channel_independent_tensors(X_train_t, y_train_t)
        x = X_ci.squeeze(1).double()  # (N, L)
        y = y_ci.double()  # (N, H)
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
        x_n = (x - mean) / std
        anchor = x[:, -1:] if USE_SKIP else mean
        y_n = (y - anchor) / std

        ones = torch.ones(x_n.shape[0], 1, dtype=x_n.dtype, device=x_n.device)
        A = torch.cat([x_n, ones], dim=1)  # (N, L+1)
        AtA = A.T @ A
        AtA += lam * torch.eye(AtA.shape[0], dtype=AtA.dtype, device=AtA.device)
        Aty = A.T @ y_n
        sol = torch.linalg.solve(AtA, Aty)  # (L+1, H)

        # Only accept the ridge solution if it beats the persistence baseline on the validation split
        if X_val_t is not None and y_val_t is not None:
            Xv_ci, yv_ci = to_channel_independent_tensors(X_val_t, y_val_t)
            xv = Xv_ci.squeeze(1).double()
            yv = yv_ci.double()
            mv = xv.mean(dim=-1, keepdim=True)
            sv = (xv.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
            xv_n = (xv - mv) / sv
            av = xv[:, -1:] if USE_SKIP else mv
            yv_n = (yv - av) / sv
            ones_v = torch.ones(xv_n.shape[0], 1, dtype=xv_n.dtype, device=xv_n.device)
            pred_v = torch.cat([xv_n, ones_v], dim=1) @ sol
            ridge_val_mse = ((pred_v - yv_n) ** 2).mean().item()
            zero_val_mse = (yv_n**2).mean().item()
            if ridge_val_mse >= zero_val_mse:
                nn.init.zeros_(model.direct.weight)
                nn.init.zeros_(model.direct.bias)
                print(
                    f"[ridge-init] Rejected: ridge val MSE {ridge_val_mse:.4f} >= "
                    f"persistence baseline {zero_val_mse:.4f}; using zero init instead."
                )
                return False

        weight = model.direct.weight
        weight.copy_(sol[:-1].T.to(weight.dtype))
        model.direct.bias.copy_(sol[-1].to(weight.dtype))
    return True


def make_channel_sample_weights(y_raw, target_channel_idx, target_loss_weight):
    if y_raw.ndim != 3 or target_loss_weight <= 1.0:
        return None
    n_samples, _, n_vars = y_raw.shape
    if target_channel_idx < 0 or target_channel_idx >= n_vars:
        return None
    weights = torch.ones(n_samples, n_vars, dtype=y_raw.dtype, device=y_raw.device)
    weights[:, target_channel_idx] = target_loss_weight
    return weights.reshape(n_samples * n_vars)


def weighted_mse_loss(pred, target, sample_weights=None):
    sample_loss = (pred - target).pow(2).mean(dim=-1)
    if sample_weights is None:
        return sample_loss.mean()
    weights = sample_weights.to(dtype=sample_loss.dtype, device=sample_loss.device)
    return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def weighted_mse_loss_mv(pred, target, target_channel_idx=None, target_loss_weight=1.0):
    sample_loss = (pred - target).pow(2).mean(dim=1)
    if target_channel_idx is None or target_loss_weight <= 1.0 or pred.ndim != 3:
        return sample_loss.mean()
    weights = torch.ones_like(sample_loss)
    weights[:, int(target_channel_idx)] = target_loss_weight
    return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def flatten_horizon_channels(y):
    return y.permute(0, 2, 1).reshape(y.shape[0] * y.shape[2], y.shape[1])


def flatten_mv_predictions(y):
    return y.permute(0, 2, 1).reshape(-1, y.shape[1])


def batched_inference(model, X, batch_size=None):
    """Run a model forward pass in chunks to bound peak activation memory.

    Full-split eval forwards (tens of thousands of channel-independent samples
    at once) are the dominant VRAM consumer in this script; chunking keeps the
    peak proportional to the chunk size instead of the split size.
    """
    if batch_size is None:
        batch_size = EVAL_BATCH_SIZE
    outs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            outs.append(model(X[start : start + batch_size]))
    return torch.cat(outs, dim=0)


def get_split_indices(total_len, dataset_name, window, protocol):
    """Return (train_end, val_end, test_end). test_end is exclusive."""
    if protocol == "patchtst" and dataset_name in {"etth1", "etth2", "ettm2"}:
        # Matches PatchTST Dataset_ETT_hour / Dataset_ETT_minute fixed borders:
        # 12/4/4 months (discards the last ~4 months)
        # from: "ultralytics\PatchTST_supervised\data_provider\data_loader.py"
        steps_per_month = 30 * 24 * (4 if dataset_name == "ettm2" else 1)
        train_end = 12 * steps_per_month
        val_end = train_end + 4 * steps_per_month
        test_end = train_end + 8 * steps_per_month
        if total_len < test_end:
            raise ValueError(
                f"{dataset_name} requires at least {test_end} rows for PatchTST protocol, got {total_len}."
            )
        if train_end <= window:
            raise ValueError(
                f"window={window} is too large for PatchTST train split boundary ({train_end})."
            )
        return train_end, val_end, test_end

    if protocol == "patchtst" and dataset_name in {"ili", "weather", "exchange_rate"}:
        # Match PatchTST Dataset_Custom split exactly (70/10/20 with integer truncation for train/test).
        num_train = int(total_len * 0.7)
        num_test = int(total_len * 0.2)
        num_vali = total_len - num_train - num_test
        train_end = num_train
        val_end = num_train + num_vali
        return train_end, val_end, total_len

    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end, total_len


def compute_train_stats(series, train_end, eps=1e-8):
    """Train-set mean/std used for PatchTST-style standardized metrics."""
    train_values = series[:train_end]
    mean = train_values.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_values.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def format_stats(values, max_items=4):
    flat = np.asarray(values).reshape(-1)
    if flat.size == 1:
        return f"{flat[0]:.4f}"
    shown = ", ".join(f"{v:.4f}" for v in flat[:max_items])
    suffix = ", ..." if flat.size > max_items else ""
    return f"[{shown}{suffix}]"


def format_channel_metric_pairs(mse_map, mae_map, ordered_names):
    return ", ".join(f"{name}:{mse_map[name]:.4f}/{mae_map[name]:.4f}" for name in ordered_names)


def to_patch_scale_tensor(y, train_mean, train_std):
    """Convert model outputs/targets to PatchTST standardized scale for fair metric comparison."""
    if EVAL_PROTOCOL == "patchtst":
        return y
    mean = torch.as_tensor(train_mean, dtype=y.dtype, device=y.device).reshape(-1)
    std = torch.as_tensor(train_std, dtype=y.dtype, device=y.device).reshape(-1)
    if mean.numel() != 1 or std.numel() != 1:
        raise ValueError("Non-patchtst per-channel scaling requires already standardized inputs.")
    return (y - mean[0]) / std[0]


def extract_channel_targets(y_flat, n_windows, n_vars, channel_idx):
    """Extract one channel from flattened (N*C, H) tensor into (N, H)."""
    return y_flat.reshape(n_windows, n_vars, -1)[:, channel_idx, :]


def extract_channel_inputs(x_flat, n_windows, n_vars, channel_idx):
    """Extract one channel from flattened (N*C, 1, L) tensor into (N, L)."""
    return x_flat.squeeze(1).reshape(n_windows, n_vars, -1)[:, channel_idx, :]


def compute_per_channel_metrics(y_true, y_pred, n_windows, n_vars, feature_names):
    y_true_ch = y_true.reshape(n_windows, n_vars, -1)
    y_pred_ch = y_pred.reshape(n_windows, n_vars, -1)
    mse_per = ((y_true_ch - y_pred_ch) ** 2).mean(dim=(0, 2)).detach().cpu().numpy()
    mae_per = (y_true_ch - y_pred_ch).abs().mean(dim=(0, 2)).detach().cpu().numpy()
    mse_map = {feature_names[i]: float(mse_per[i]) for i in range(n_vars)}
    mae_map = {feature_names[i]: float(mae_per[i]) for i in range(n_vars)}
    return mse_map, mae_map


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


def _odd_kernel(kernel, max_len):
    kernel = max(3, min(int(kernel), int(max_len)))
    if kernel % 2 == 0:
        kernel = max(3, kernel - 1)
    return kernel


def moving_average_1d(y, kernel):
    kernel = _odd_kernel(kernel, y.shape[-1])
    pad = kernel // 2
    y_pad = F.pad(y.unsqueeze(1), (pad, pad), mode="replicate")
    return F.avg_pool1d(y_pad, kernel_size=kernel, stride=1).squeeze(1)


def decompose_batch(y, mode="moving_avg", low_frac=0.05, mid_frac=0.2, trend_kernel=0, seasonal_kernel=0):
    if mode == "fft":
        return spectral_decompose_batch(y, low_frac=low_frac, mid_frac=mid_frac)

    kernel = trend_kernel if trend_kernel > 0 else max(3, y.shape[-1] // 4)
    trend = moving_average_1d(y, kernel)
    remainder = y - trend
    seasonal_kernel = seasonal_kernel if seasonal_kernel > 0 else max(3, kernel // 4)
    seasonal = moving_average_1d(remainder, seasonal_kernel)
    resid = remainder - seasonal
    return trend, seasonal, resid


def slope_mse_loss(pred, target):
    if pred.shape[-1] < 2:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    pred_delta = pred[..., 1:] - pred[..., :-1]
    target_delta = target[..., 1:] - target[..., :-1]
    return ((pred_delta - target_delta) ** 2).mean()


def curvature_mse_loss(pred, target):
    if pred.shape[-1] < 3:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    pred_curve = pred[..., 2:] - 2.0 * pred[..., 1:-1] + pred[..., :-2]
    target_curve = target[..., 2:] - 2.0 * target[..., 1:-1] + target[..., :-2]
    return ((pred_curve - target_curve) ** 2).mean()


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
    use_spectral_loss=False,
    spectral_low_frac=0.05,
    spectral_mid_frac=0.2,
    spectral_trend_weight=0.25,
    spectral_warmup_epochs=0,
    use_shape_aware_loss=False,
    shape_slope_weight=0.0,
    shape_curvature_weight=0.0,
    decomp_mode="moving_avg",
    trend_kernel=0,
    seasonal_kernel=0,
    level_loss_weight=0.0,
    target_channel_idx=None,
    target_loss_weight=1.0,
    monitor_target_channel=False,
    use_last_epoch=False,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # AdamW with weight decay for regularization
    loss_fn = nn.MSELoss()
    eps = 1e-5
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)  # Cosine annealing LR scheduler
    
    best_val_loss_norm = float("inf")
    best_val_loss_raw = float("inf")
    best_monitor_loss = float("inf")
    best_weights = None

    no_improve_epochs = 0
    history = {"train_norm": [], "val_norm": [], "train_raw": [], "val_raw": []}
    best_epoch = epochs - 1  # default to last epoch if use_last_epoch

    # Initialize lazy modules before any state_dict cloning
    # Use eval mode for warmup to avoid BatchNorm errors on tiny warmup shapes
    warmup_bs = min(2, len(X_train_t))
    X_warm, y_warm = to_channel_independent_tensors(X_train_t[:warmup_bs], y_train_t[:warmup_bs])
    model.eval()
    with torch.no_grad():
        _ = model(X_warm)
    
    pbar = tqdm(range(epochs), desc="Training", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        indices = torch.randperm(len(X_train_t))
        spectral_aux_scale = 1.0
        if use_spectral_loss and spectral_warmup_epochs > 0:
            spectral_aux_scale = min(1.0, float(epoch + 1) / float(spectral_warmup_epochs))

        epoch_loss = 0.0
        epoch_loss_raw = 0.0
        epoch_loss_p3 = 0.0
        epoch_loss_p4 = 0.0
        epoch_loss_p5 = 0.0
        epoch_loss_level = 0.0
        epoch_loss_shape_slope = 0.0
        epoch_loss_shape_curvature = 0.0
        n_batches = 0
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch_raw = X_train_t[batch_idx]
            y_batch_raw = y_train_t[batch_idx]
            X_batch, y_batch = to_channel_independent_tensors(X_batch_raw, y_batch_raw)
            sample_weights = make_channel_sample_weights(y_batch_raw, target_channel_idx, target_loss_weight)
            
            optimizer.zero_grad()
            if use_spectral_loss:
                pred, pred_p3, pred_p4, pred_p5 = model(X_batch, return_components=True)
                loss_raw = weighted_mse_loss(pred, y_batch, sample_weights)

                # Normalize y_batch into the same space as the component predictions
                # 2/25/26 change: for spectral loss, when using skip, use the last value instead of window mean (this makes it use the same rule as the main loss path)
                mean = X_batch[:, :, -1:] if USE_SKIP else X_batch.mean(dim=-1, keepdim=True)  # (B, 1, 1)
                std = (X_batch.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                if USE_REVIN:
                    y_batch_norm = (y_batch - mean.squeeze(-1)) / std.squeeze(-1)

                    # Component heads are returned after RevIN denorm/denorm_delta in model.forward().
                    # Bring them back to normalized space so they match spectral targets.
                    if USE_SKIP:
                        pred_p3_aux = pred_p3 / std.squeeze(-1)
                        pred_p4_aux = pred_p4 / std.squeeze(-1)
                        pred_p5_aux = pred_p5 / std.squeeze(-1)
                    else:
                        pred_p3_aux = (pred_p3 - mean.squeeze(-1)) / std.squeeze(-1)
                        pred_p4_aux = (pred_p4 - mean.squeeze(-1)) / std.squeeze(-1)
                        pred_p5_aux = (pred_p5 - mean.squeeze(-1)) / std.squeeze(-1)
                else:
                    y_batch_norm = y_batch
                    pred_p3_aux = pred_p3
                    pred_p4_aux = pred_p4
                    pred_p5_aux = pred_p5
                
                trend, seasonal, resid = decompose_batch(
                    y_batch_norm,
                    mode=decomp_mode,
                    low_frac=spectral_low_frac,
                    mid_frac=spectral_mid_frac,
                    trend_kernel=trend_kernel,
                    seasonal_kernel=seasonal_kernel,
                )
                loss_p3 = weighted_mse_loss(pred_p3_aux, resid, sample_weights)
                loss_p4 = weighted_mse_loss(pred_p4_aux, seasonal, sample_weights)
                loss_p5 = weighted_mse_loss(pred_p5_aux, trend, sample_weights)
                pred_aux = (pred - mean.squeeze(-1)) / std.squeeze(-1) if USE_REVIN else pred
                target_aux = y_batch_norm if USE_REVIN else y_batch
                loss_level = weighted_mse_loss(
                    pred_aux.mean(dim=-1, keepdim=True),
                    target_aux.mean(dim=-1, keepdim=True),
                    sample_weights,
                )

                shape_loss_slope = torch.zeros((), dtype=loss_p5.dtype, device=loss_p5.device)
                shape_loss_curvature = torch.zeros((), dtype=loss_p5.dtype, device=loss_p5.device)
                if use_shape_aware_loss:
                    # Trend-branch shape loss (existing behavior)
                    shape_loss_slope_trend = (
                        model.slope_loss(pred_p5_aux, trend)
                        if hasattr(model, "slope_loss")
                        else slope_mse_loss(pred_p5_aux, trend)
                    )
                    shape_loss_curvature_trend = (
                        model.curvature_loss(pred_p5_aux, trend)
                        if hasattr(model, "curvature_loss")
                        else curvature_mse_loss(pred_p5_aux, trend)
                    )

                    # Final-output shape loss (new): directly regularize the forecast path
                    # to reduce amplitude underprediction not captured by trend-only penalties.
                    shape_loss_slope_main = (
                        model.slope_loss(pred_aux, target_aux)
                        if hasattr(model, "slope_loss")
                        else slope_mse_loss(pred_aux, target_aux)
                    )
                    shape_loss_curvature_main = (
                        model.curvature_loss(pred_aux, target_aux)
                        if hasattr(model, "curvature_loss")
                        else curvature_mse_loss(pred_aux, target_aux)
                    )

                    # Keep overall scale close to previous behavior while broadening supervision.
                    shape_loss_slope = 0.5 * (shape_loss_slope_trend + shape_loss_slope_main)
                    shape_loss_curvature = 0.5 * (
                        shape_loss_curvature_trend + shape_loss_curvature_main
                    )

                if hasattr(model, "log_sigma_p5"):
                    # 2/25/26 change: clamp log_sigma within [-4, 4] (before exponentiation) to prevent big loss spikes
                    # We already had standard regularization terms (log_sigmas), so this is to further stabilize it
                    log_sigma_p3 = torch.clamp(model.log_sigma_p3, min=-4.0, max=4.0)
                    log_sigma_p4 = torch.clamp(model.log_sigma_p4, min=-4.0, max=4.0)
                    log_sigma_p5 = torch.clamp(model.log_sigma_p5, min=-4.0, max=4.0)
                    p3_weight = torch.exp(-log_sigma_p3)
                    p4_weight = torch.exp(-log_sigma_p4)
                    p5_weight = torch.clamp(torch.exp(-log_sigma_p5), min=0.25)
                    loss = (
                        loss_raw
                        + spectral_aux_scale
                        * (
                            p3_weight * loss_p3
                            + log_sigma_p3
                            + p4_weight * loss_p4
                            + log_sigma_p4
                            + p5_weight * loss_p5
                            + log_sigma_p5
                        )
                    )
                else:
                    p3_weight = spectral_trend_weight
                    p4_weight = spectral_trend_weight
                    p5_weight = spectral_trend_weight
                    loss = loss_raw + spectral_aux_scale * (
                        p3_weight * loss_p3 + p4_weight * loss_p4 + p5_weight * loss_p5
                    )

                # Keep shape penalties outside the uncertainty-weighted p5 trend term so
                # shape-related CLI weights have direct and predictable influence.
                loss = loss + spectral_aux_scale * (
                    shape_slope_weight * shape_loss_slope
                    + shape_curvature_weight * shape_loss_curvature
                    + level_loss_weight * loss_level
                )

                epoch_loss_p3 += (spectral_aux_scale * p3_weight * loss_p3).item() #loss_p3.item()
                epoch_loss_p4 += (spectral_aux_scale * p4_weight * loss_p4).item() #loss_p4.item()
                epoch_loss_p5 += (spectral_aux_scale * p5_weight * loss_p5).item() #loss_p5.item()
                epoch_loss_level += (spectral_aux_scale * level_loss_weight * loss_level).item()
                epoch_loss_shape_slope += (
                    spectral_aux_scale * shape_slope_weight * shape_loss_slope
                ).item()
                epoch_loss_shape_curvature += (
                    spectral_aux_scale * shape_curvature_weight * shape_loss_curvature
                ).item()

                if USE_REVIN:
                    mean = X_batch[:, :, -1:] if USE_SKIP else X_batch.mean(dim=-1, keepdim=True)
                    std = (X_batch.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                    pred_n = (pred - mean.squeeze(-1)) / std.squeeze(-1)
                    y_batch_n = (y_batch - mean.squeeze(-1)) / std.squeeze(-1)
                    loss_log = weighted_mse_loss(pred_n, y_batch_n, sample_weights)
                else:
                    loss_log = loss_raw
            else:
                pred = model(X_batch)
                loss_raw = weighted_mse_loss(pred, y_batch, sample_weights)
                if USE_REVIN:
                    mean = X_batch[:, :, -1:] if USE_SKIP else X_batch.mean(dim=-1, keepdim=True)
                    std = (X_batch.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                    pred = (pred - mean.squeeze(-1)) / std.squeeze(-1)
                    y_batch = (y_batch - mean.squeeze(-1)) / std.squeeze(-1)
                loss_log = weighted_mse_loss(pred, y_batch, sample_weights)
                loss = loss_log
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            
            optimizer.step()
            
            epoch_loss += loss_log.item()
            epoch_loss_raw += loss_raw.item()
            n_batches += 1
        
        scheduler.step()  # Step scheduler after each epoch
        
        avg_train_loss = epoch_loss / max(1, n_batches)
        avg_train_loss_raw = epoch_loss_raw / max(1, n_batches)
        if use_spectral_loss:
            avg_p3 = epoch_loss_p3 / max(1, n_batches)
            avg_p4 = epoch_loss_p4 / max(1, n_batches)
            avg_p5 = epoch_loss_p5 / max(1, n_batches)
            avg_level = epoch_loss_level / max(1, n_batches)
            if use_shape_aware_loss:
                avg_shape_slope = epoch_loss_shape_slope / max(1, n_batches)
                avg_shape_curvature = epoch_loss_shape_curvature / max(1, n_batches)
            # print(f"Epoch {epoch + 1}: spectral losses p3={avg_p3:.4f}, p4={avg_p4:.4f}, p5={avg_p5:.4f}")
        
        model.eval()
        with torch.no_grad():
            X_val_ci, y_val_ci = to_channel_independent_tensors(X_val_t, y_val_t)
            val_pred_raw = batched_inference(model, X_val_ci)
            val_loss_raw = loss_fn(val_pred_raw, y_val_ci).item()
            if USE_REVIN:
                mean = X_val_ci[:, :, -1:] if USE_SKIP else X_val_ci.mean(dim=-1, keepdim=True)
                std = (X_val_ci.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                val_pred_norm = (val_pred_raw - mean.squeeze(-1)) / std.squeeze(-1)
                y_val_n = (y_val_ci - mean.squeeze(-1)) / std.squeeze(-1)
                val_loss_norm = loss_fn(val_pred_norm, y_val_n).item()
            else:
                val_pred_norm = val_pred_raw
                y_val_n = y_val_ci
                val_loss_norm = val_loss_raw
        
        history["train_norm"].append(avg_train_loss)
        history["val_norm"].append(val_loss_norm)
        history["train_raw"].append(avg_train_loss_raw)
        history["val_raw"].append(val_loss_raw)

        monitor_val_loss = val_loss_raw if EVAL_PROTOCOL == "patchtst" else val_loss_norm
        if monitor_target_channel and y_val_t.ndim == 3 and target_channel_idx is not None:
            n_val_vars = y_val_t.shape[2]
            target_rows = torch.arange(
                int(target_channel_idx),
                y_val_ci.shape[0],
                n_val_vars,
                device=y_val_ci.device,
            )
            if EVAL_PROTOCOL == "patchtst":
                monitor_val_loss = loss_fn(val_pred_raw[target_rows], y_val_ci[target_rows]).item()
            else:
                monitor_val_loss = loss_fn(val_pred_norm[target_rows], y_val_n[target_rows]).item()
        
        if monitor_val_loss < best_monitor_loss:
            best_monitor_loss = monitor_val_loss
            best_val_loss_norm = val_loss_norm
            best_val_loss_raw = val_loss_raw
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            best_epoch = epoch
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= PATIENCE:
            break
        
        current_lr = scheduler.get_last_lr()[0]
        postfix = {
            "train": f"{avg_train_loss:.4f}",
            "val": f"{monitor_val_loss:.4f}",
            "best": f"{best_monitor_loss:.4f}",
            "lr": f"{current_lr:.1e}",
        }
        if use_spectral_loss and spectral_warmup_epochs > 0:
            postfix["aux"] = f"{spectral_aux_scale:.2f}"
        if use_spectral_loss and level_loss_weight > 0:
            postfix["level"] = f"{avg_level:.4f}"
        if use_shape_aware_loss:
            postfix["slope"] = f"{avg_shape_slope:.4f}"
            postfix["curv"] = f"{avg_shape_curvature:.4f}"
        pbar.set_postfix(**postfix)
    
    if not use_last_epoch and best_weights is not None:
        model.load_state_dict(best_weights)
    return best_val_loss_norm, best_val_loss_raw, history, best_epoch + 1


def train_model_multivariate(
    model,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    use_spectral_loss=False,
    spectral_low_frac=0.05,
    spectral_mid_frac=0.2,
    spectral_trend_weight=0.25,
    spectral_warmup_epochs=0,
    use_shape_aware_loss=False,
    shape_slope_weight=0.0,
    shape_curvature_weight=0.0,
    decomp_mode="moving_avg",
    trend_kernel=0,
    seasonal_kernel=0,
    level_loss_weight=0.0,
    target_channel_idx=None,
    target_loss_weight=1.0,
    monitor_target_channel=False,
    use_last_epoch=False,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    eps = 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    best_val_loss_norm = float("inf")
    best_val_loss_raw = float("inf")
    best_monitor_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    history = {"train_norm": [], "val_norm": [], "train_raw": [], "val_raw": []}
    best_epoch = epochs - 1

    warmup_bs = min(2, len(X_train_t))
    model.eval()
    with torch.no_grad():
        _ = model(X_train_t[:warmup_bs])

    pbar = tqdm(range(epochs), desc="Training", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        indices = torch.randperm(len(X_train_t))
        spectral_aux_scale = 1.0
        if use_spectral_loss and spectral_warmup_epochs > 0:
            spectral_aux_scale = min(1.0, float(epoch + 1) / float(spectral_warmup_epochs))

        epoch_loss = 0.0
        epoch_loss_raw = 0.0
        epoch_loss_p3 = 0.0
        epoch_loss_p4 = 0.0
        epoch_loss_p5 = 0.0
        epoch_loss_level = 0.0
        epoch_loss_shape_slope = 0.0
        epoch_loss_shape_curvature = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            optimizer.zero_grad()

            if use_spectral_loss:
                pred, pred_p3, pred_p4, pred_p5 = model(X_batch, return_components=True)
                loss_raw = weighted_mse_loss_mv(pred, y_batch, target_channel_idx, target_loss_weight)

                X_cf = X_batch.transpose(1, 2)
                y_cf = y_batch.transpose(1, 2)
                mean = X_cf[:, :, -1:] if USE_SKIP else X_cf.mean(dim=-1, keepdim=True)
                std = (X_cf.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                if USE_REVIN:
                    y_norm_cf = (y_cf - mean) / std
                    if USE_SKIP:
                        pred_p3_aux_cf = pred_p3.transpose(1, 2) / std
                        pred_p4_aux_cf = pred_p4.transpose(1, 2) / std
                        pred_p5_aux_cf = pred_p5.transpose(1, 2) / std
                    else:
                        pred_p3_aux_cf = (pred_p3.transpose(1, 2) - mean) / std
                        pred_p4_aux_cf = (pred_p4.transpose(1, 2) - mean) / std
                        pred_p5_aux_cf = (pred_p5.transpose(1, 2) - mean) / std
                else:
                    y_norm_cf = y_cf
                    pred_p3_aux_cf = pred_p3.transpose(1, 2)
                    pred_p4_aux_cf = pred_p4.transpose(1, 2)
                    pred_p5_aux_cf = pred_p5.transpose(1, 2)

                y_norm_flat = y_norm_cf.reshape(-1, y_norm_cf.shape[-1])
                pred_p3_aux = pred_p3_aux_cf.reshape(-1, pred_p3_aux_cf.shape[-1])
                pred_p4_aux = pred_p4_aux_cf.reshape(-1, pred_p4_aux_cf.shape[-1])
                pred_p5_aux = pred_p5_aux_cf.reshape(-1, pred_p5_aux_cf.shape[-1])
                sample_weights = make_channel_sample_weights(y_batch, target_channel_idx, target_loss_weight)
                trend, seasonal, resid = decompose_batch(
                    y_norm_flat,
                    mode=decomp_mode,
                    low_frac=spectral_low_frac,
                    mid_frac=spectral_mid_frac,
                    trend_kernel=trend_kernel,
                    seasonal_kernel=seasonal_kernel,
                )
                loss_p3 = weighted_mse_loss(pred_p3_aux, resid, sample_weights)
                loss_p4 = weighted_mse_loss(pred_p4_aux, seasonal, sample_weights)
                loss_p5 = weighted_mse_loss(pred_p5_aux, trend, sample_weights)

                pred_cf = pred.transpose(1, 2)
                pred_aux_cf = (pred_cf - mean) / std if USE_REVIN else pred_cf
                pred_aux = pred_aux_cf.reshape(-1, pred_aux_cf.shape[-1])
                target_aux = y_norm_flat
                loss_level = weighted_mse_loss(
                    pred_aux.mean(dim=-1, keepdim=True),
                    target_aux.mean(dim=-1, keepdim=True),
                    sample_weights,
                )

                shape_loss_slope = torch.zeros((), dtype=loss_p5.dtype, device=loss_p5.device)
                shape_loss_curvature = torch.zeros((), dtype=loss_p5.dtype, device=loss_p5.device)
                if use_shape_aware_loss:
                    shape_loss_slope_trend = (
                        model.slope_loss(pred_p5_aux, trend)
                        if hasattr(model, "slope_loss")
                        else slope_mse_loss(pred_p5_aux, trend)
                    )
                    shape_loss_curvature_trend = (
                        model.curvature_loss(pred_p5_aux, trend)
                        if hasattr(model, "curvature_loss")
                        else curvature_mse_loss(pred_p5_aux, trend)
                    )
                    shape_loss_slope_main = (
                        model.slope_loss(pred_aux, target_aux)
                        if hasattr(model, "slope_loss")
                        else slope_mse_loss(pred_aux, target_aux)
                    )
                    shape_loss_curvature_main = (
                        model.curvature_loss(pred_aux, target_aux)
                        if hasattr(model, "curvature_loss")
                        else curvature_mse_loss(pred_aux, target_aux)
                    )
                    shape_loss_slope = 0.5 * (shape_loss_slope_trend + shape_loss_slope_main)
                    shape_loss_curvature = 0.5 * (shape_loss_curvature_trend + shape_loss_curvature_main)

                if hasattr(model, "log_sigma_p5"):
                    log_sigma_p3 = torch.clamp(model.log_sigma_p3, min=-4.0, max=4.0)
                    log_sigma_p4 = torch.clamp(model.log_sigma_p4, min=-4.0, max=4.0)
                    log_sigma_p5 = torch.clamp(model.log_sigma_p5, min=-4.0, max=4.0)
                    p3_weight = torch.exp(-log_sigma_p3)
                    p4_weight = torch.exp(-log_sigma_p4)
                    p5_weight = torch.clamp(torch.exp(-log_sigma_p5), min=0.25)
                    loss = loss_raw + spectral_aux_scale * (
                        p3_weight * loss_p3
                        + log_sigma_p3
                        + p4_weight * loss_p4
                        + log_sigma_p4
                        + p5_weight * loss_p5
                        + log_sigma_p5
                    )
                else:
                    p3_weight = spectral_trend_weight
                    p4_weight = spectral_trend_weight
                    p5_weight = spectral_trend_weight
                    loss = loss_raw + spectral_aux_scale * (
                        p3_weight * loss_p3 + p4_weight * loss_p4 + p5_weight * loss_p5
                    )

                loss = loss + spectral_aux_scale * (
                    shape_slope_weight * shape_loss_slope
                    + shape_curvature_weight * shape_loss_curvature
                    + level_loss_weight * loss_level
                )
                epoch_loss_p3 += (spectral_aux_scale * p3_weight * loss_p3).item()
                epoch_loss_p4 += (spectral_aux_scale * p4_weight * loss_p4).item()
                epoch_loss_p5 += (spectral_aux_scale * p5_weight * loss_p5).item()
                epoch_loss_level += (spectral_aux_scale * level_loss_weight * loss_level).item()
                epoch_loss_shape_slope += (spectral_aux_scale * shape_slope_weight * shape_loss_slope).item()
                epoch_loss_shape_curvature += (spectral_aux_scale * shape_curvature_weight * shape_loss_curvature).item()

                if USE_REVIN:
                    pred_n = pred_aux_cf.transpose(1, 2)
                    y_batch_n = y_norm_cf.transpose(1, 2)
                    loss_log = weighted_mse_loss_mv(pred_n, y_batch_n, target_channel_idx, target_loss_weight)
                else:
                    loss_log = loss_raw
            else:
                pred = model(X_batch)
                loss_raw = weighted_mse_loss_mv(pred, y_batch, target_channel_idx, target_loss_weight)
                if USE_REVIN:
                    X_cf = X_batch.transpose(1, 2)
                    y_cf = y_batch.transpose(1, 2)
                    mean = X_cf[:, :, -1:] if USE_SKIP else X_cf.mean(dim=-1, keepdim=True)
                    std = (X_cf.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                    pred_n = ((pred.transpose(1, 2) - mean) / std).transpose(1, 2)
                    y_batch_n = ((y_cf - mean) / std).transpose(1, 2)
                    loss_log = weighted_mse_loss_mv(pred_n, y_batch_n, target_channel_idx, target_loss_weight)
                else:
                    loss_log = loss_raw
                loss = loss_log

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss_log.item()
            epoch_loss_raw += loss_raw.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, n_batches)
        avg_train_loss_raw = epoch_loss_raw / max(1, n_batches)
        if use_spectral_loss:
            avg_level = epoch_loss_level / max(1, n_batches)
            if use_shape_aware_loss:
                avg_shape_slope = epoch_loss_shape_slope / max(1, n_batches)
                avg_shape_curvature = epoch_loss_shape_curvature / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_pred_raw = batched_inference(model, X_val_t)
            val_loss_raw = loss_fn(val_pred_raw, y_val_t).item()
            if USE_REVIN:
                X_val_cf = X_val_t.transpose(1, 2)
                y_val_cf = y_val_t.transpose(1, 2)
                mean = X_val_cf[:, :, -1:] if USE_SKIP else X_val_cf.mean(dim=-1, keepdim=True)
                std = (X_val_cf.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
                val_pred_norm = ((val_pred_raw.transpose(1, 2) - mean) / std).transpose(1, 2)
                y_val_n = ((y_val_cf - mean) / std).transpose(1, 2)
                val_loss_norm = loss_fn(val_pred_norm, y_val_n).item()
            else:
                val_pred_norm = val_pred_raw
                y_val_n = y_val_t
                val_loss_norm = val_loss_raw

        history["train_norm"].append(avg_train_loss)
        history["val_norm"].append(val_loss_norm)
        history["train_raw"].append(avg_train_loss_raw)
        history["val_raw"].append(val_loss_raw)
        monitor_val_loss = val_loss_raw if EVAL_PROTOCOL == "patchtst" else val_loss_norm
        if monitor_target_channel and target_channel_idx is not None:
            if EVAL_PROTOCOL == "patchtst":
                monitor_val_loss = loss_fn(val_pred_raw[:, :, int(target_channel_idx)], y_val_t[:, :, int(target_channel_idx)]).item()
            else:
                monitor_val_loss = loss_fn(val_pred_norm[:, :, int(target_channel_idx)], y_val_n[:, :, int(target_channel_idx)]).item()

        if monitor_val_loss < best_monitor_loss:
            best_monitor_loss = monitor_val_loss
            best_val_loss_norm = val_loss_norm
            best_val_loss_raw = val_loss_raw
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            best_epoch = epoch
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= PATIENCE:
            break

        current_lr = scheduler.get_last_lr()[0]
        postfix = {
            "train": f"{avg_train_loss:.4f}",
            "val": f"{monitor_val_loss:.4f}",
            "best": f"{best_monitor_loss:.4f}",
            "lr": f"{current_lr:.1e}",
        }
        if use_spectral_loss and spectral_warmup_epochs > 0:
            postfix["aux"] = f"{spectral_aux_scale:.2f}"
        if use_spectral_loss and level_loss_weight > 0:
            postfix["level"] = f"{avg_level:.4f}"
        if use_shape_aware_loss:
            postfix["slope"] = f"{avg_shape_slope:.4f}"
            postfix["curv"] = f"{avg_shape_curvature:.4f}"
        pbar.set_postfix(**postfix)

    if not use_last_epoch and best_weights is not None:
        model.load_state_dict(best_weights)
    return best_val_loss_norm, best_val_loss_raw, history, best_epoch + 1


def train_model_patchtst(
    model,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    use_last_epoch=False,
    patience=PATIENCE,
    lradj="TST",
    pct_start=0.3,
    drop_last=True,
):
    """Train loop aligned with official PatchTST supervised setup.

    lradj modes (matching PatchTST_supervised/utils/tools.py):
      - 'constant': LR stays fixed at *lr* for all epochs.
      - 'TST':      OneCycleLR with pct_start, stepped per batch.
      - 'type3':    LR = *lr* for first 3 epochs, then *lr* * 0.9^(epoch-3) per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = None
    if lradj == "TST":
        steps_per_epoch = max(1, int(np.ceil(len(X_train_t) / float(batch_size))))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )

    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    history = {"train_norm": [], "val_norm": [], "train_raw": [], "val_raw": []}
    best_epoch = epochs - 1

    warmup_bs = min(2, len(X_train_t))
    model.eval()
    with torch.no_grad():
        _ = model(X_train_t[:warmup_bs])

    pbar = tqdm(range(epochs), desc="Training", unit="epoch", leave=False)
    for epoch in pbar:
        # Per-epoch LR adjustment for type3 (done before training each epoch, 1-indexed)
        if lradj == "type3":
            if epoch + 1 <= 3:
                current_lr = lr
            else:
                current_lr = lr * (0.9 ** ((epoch + 1) - 3))
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        elif lradj == "constant":
            current_lr = lr
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        n_train_samples = len(indices)
        if drop_last:
            n_train_samples = (n_train_samples // batch_size) * batch_size

        for start in range(0, n_train_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_pred = batched_inference(model, X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        history["train_norm"].append(avg_train_loss)
        history["val_norm"].append(val_loss)
        history["train_raw"].append(avg_train_loss)
        history["val_raw"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        display_lr = scheduler.get_last_lr()[0] if scheduler is not None else current_lr
        postfix = {
            "train": f"{avg_train_loss:.4f}",
            "val": f"{val_loss:.4f}",
            "best": f"{best_val_loss:.4f}",
            "lr": f"{display_lr:.1e}",
        }
        if patience > 0:
            postfix["es"] = f"{no_improve_epochs}/{patience}"
        pbar.set_postfix(**postfix)

        if patience > 0 and no_improve_epochs >= patience:
            break

    if not use_last_epoch and best_weights is not None:
        model.load_state_dict(best_weights)
    return best_val_loss, best_val_loss, history, best_epoch + 1


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


def compute_patchtst_metrics(y_true, y_pred, train_mean, train_std, already_standardized=False):
    if already_standardized:
        return compute_metrics(y_true, y_pred)
    mean = torch.as_tensor(train_mean, dtype=y_true.dtype, device=y_true.device)
    std = torch.as_tensor(train_std, dtype=y_true.dtype, device=y_true.device)
    if mean.numel() != 1 or std.numel() != 1:
        raise ValueError("Raw-space PatchTST metric conversion currently supports univariate inputs only.")
    y_true_s = (y_true - mean) / std
    y_pred_s = (y_pred - mean) / std
    return compute_metrics(y_true_s, y_pred_s)



# Data prep
print(f"Loading {DATASET_LABEL} from {CSV_PATH}...")
series, feature_cols = load_time_series(CSV_PATH, features=FEATURES, target_col=TARGET_COL)
print(f"Total samples: {len(series)}")
print(f"Feature mode: {FEATURES} ({len(feature_cols)} channel(s))")

t_train_end, t_val_end, t_test_end = get_split_indices(len(series), args.dataset, WINDOW, EVAL_PROTOCOL)
train_mean, train_std = compute_train_stats(series, t_train_end)

if EVAL_PROTOCOL == "patchtst":
    series_source = (series - train_mean) / train_std
else:
    if FEATURES == "M":
        raise ValueError("Multivariate mode currently requires --eval-protocol patchtst for consistent metric scaling.")
    series_source = series

series_source = series_source[:t_test_end]
train_series = series_source[:t_train_end]
val_series = series_source[t_train_end - WINDOW : t_val_end]
test_series = series_source[t_val_end - WINDOW : t_test_end]

X_train_w, y_train_w = create_windows(train_series, WINDOW, HORIZON, stride=TRAIN_STRIDE)
X_val_w, y_val_w = create_windows(val_series, WINDOW, HORIZON, stride=1)
X_test_w, y_test_w = create_windows(test_series, WINDOW, HORIZON, stride=1)

if len(X_train_w) == 0 or len(X_val_w) == 0 or len(X_test_w) == 0:
    raise ValueError(
        f"Insufficient samples after split for dataset='{args.dataset}' with window={WINDOW}, horizon={HORIZON}. "
        "Try reducing --window and/or --horizon."
    )

N_VARS = X_train_w.shape[-1]
N_TEST_WINDOWS = len(X_test_w)
OT_CHANNEL_IDX = feature_cols.index(TARGET_COL) if TARGET_COL in feature_cols else 0
OT_CHANNEL_NAME = feature_cols[OT_CHANNEL_IDX]

X_train_mv, y_train_mv = X_train_w, y_train_w
X_val_mv, y_val_mv = X_val_w, y_val_w
X_test_mv, y_test_mv = X_test_w, y_test_w

X_train, y_train = to_channel_independent_dataset(X_train_w, y_train_w)
X_val, y_val = to_channel_independent_dataset(X_val_w, y_val_w)
X_test, y_test = to_channel_independent_dataset(X_test_w, y_test_w)

print(
    f"Windows: train={len(X_train_w)}, val={len(X_val_w)}, test={len(X_test_w)} "
    f"(window={WINDOW}, horizon={HORIZON}, channels={N_VARS})"
)
print(
    f"Channel-independent samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
)
print(f"Evaluation protocol: {EVAL_PROTOCOL}")
print(f"PatchTST metric scaler (train split): mean={format_stats(train_mean)}, std={format_stats(train_std)}")
print(f"OT channel for plotting/OT-only metrics: {OT_CHANNEL_NAME}")

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Device: {DEVICE}")

X_train_mv_t = torch.tensor(X_train_mv, dtype=torch.float32).to(DEVICE)
y_train_mv_t = torch.tensor(y_train_mv, dtype=torch.float32).to(DEVICE)
X_val_mv_t = torch.tensor(X_val_mv, dtype=torch.float32).to(DEVICE)
y_val_mv_t = torch.tensor(y_val_mv, dtype=torch.float32).to(DEVICE)
X_test_mv_t = torch.tensor(X_test_mv, dtype=torch.float32).to(DEVICE)
y_test_mv_t = torch.tensor(y_test_mv, dtype=torch.float32).to(DEVICE)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)


# Train + evaluate all models
results = {}
predictions = {}
all_histories = {}
all_run_results = {}  # {model_name: [metrics_dict_per_run]}
model_colors = {}

# Naive baseline: always predict the mean of the input window
baseline_name = "Train Mean"
baseline_color = "lightgray"

with torch.no_grad():
    y_val_baseline = X_val_t.mean(dim=-1).squeeze(1).unsqueeze(1).repeat(1, HORIZON)
    y_test_baseline = X_test_t.mean(dim=-1).squeeze(1).unsqueeze(1).repeat(1, HORIZON)

baseline_val_mse_raw, _ = compute_metrics(y_val_t, y_val_baseline)
baseline_test_mse_raw, baseline_test_mae_raw = compute_metrics(y_test_t, y_test_baseline)
baseline_val_mse_patch, _ = compute_patchtst_metrics(
    y_val_t,
    y_val_baseline,
    train_mean,
    train_std,
    already_standardized=EVAL_PROTOCOL == "patchtst",
)
baseline_test_mse_patch, baseline_test_mae_patch = compute_patchtst_metrics(
    y_test_t,
    y_test_baseline,
    train_mean,
    train_std,
    already_standardized=EVAL_PROTOCOL == "patchtst",
)
y_test_patch = to_patch_scale_tensor(y_test_t, train_mean, train_std)
y_test_baseline_patch = to_patch_scale_tensor(y_test_baseline, train_mean, train_std)
baseline_test_mse_per_ch_patch, baseline_test_mae_per_ch_patch = compute_per_channel_metrics(
    y_test_patch,
    y_test_baseline_patch,
    N_TEST_WINDOWS,
    N_VARS,
    feature_cols,
)
baseline_test_mse_ot_patch = baseline_test_mse_per_ch_patch[OT_CHANNEL_NAME]
baseline_test_mae_ot_patch = baseline_test_mae_per_ch_patch[OT_CHANNEL_NAME]

ot_mean = float(train_mean.reshape(-1)[OT_CHANNEL_IDX])
ot_std = float(train_std.reshape(-1)[OT_CHANNEL_IDX])
baseline_y_true_ot_raw = extract_channel_targets(y_test_patch, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
baseline_y_pred_ot_raw = extract_channel_targets(y_test_baseline_patch, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
baseline_test_mse_ot_raw, baseline_test_mae_ot_raw = compute_metrics(baseline_y_true_ot_raw, baseline_y_pred_ot_raw)

if USE_REVIN:
    baseline_val_mse_norm, _ = compute_normalized_metrics(X_val_t, y_val_t, y_val_baseline)
    baseline_test_mse_norm, baseline_test_mae_norm = compute_normalized_metrics(X_test_t, y_test_t, y_test_baseline)
else:
    baseline_val_mse_norm = baseline_val_mse_raw
    baseline_test_mse_norm = baseline_test_mse_raw
    baseline_test_mae_norm = baseline_test_mae_raw

_baseline_metrics = {
    "best_epoch": 0,
    "best_val_loss_norm": baseline_val_mse_norm,
    "best_val_loss_raw": baseline_val_mse_raw,
    "val_mse_patch": baseline_val_mse_patch,
    "test_mse_raw": baseline_test_mse_raw,
    "test_mae_raw": baseline_test_mae_raw,
    "test_mse_patch": baseline_test_mse_patch,
    "test_mae_patch": baseline_test_mae_patch,
    "test_mse_ot_patch": baseline_test_mse_ot_patch,
    "test_mae_ot_patch": baseline_test_mae_ot_patch,
    "test_mse_ot_raw": baseline_test_mse_ot_raw,
    "test_mae_ot_raw": baseline_test_mae_ot_raw,
    "test_mse_per_ch_patch": baseline_test_mse_per_ch_patch,
    "test_mae_per_ch_patch": baseline_test_mae_per_ch_patch,
    "test_mse_norm": baseline_test_mse_norm,
    "test_mae_norm": baseline_test_mae_norm,
}
results[baseline_name] = _baseline_metrics
all_run_results[baseline_name] = [_baseline_metrics] * REPEATS  # deterministic, duplicate
predictions[baseline_name] = (y_test_baseline, baseline_color)

print("\n" + "=" * 70)
_epoch_label = "LAST EPOCH" if USE_LAST_EPOCH else "BEST EPOCH"
model_space_label = "PatchTST standardized" if EVAL_PROTOCOL == "patchtst" else "Raw"
print(f"TRAINING ALL MODELS ON {DATASET_LABEL} "
      f"(features={FEATURES}, use_skip={USE_SKIP}, use_revin={USE_REVIN}, patch_len={PATCH_LEN}, patch_stride={PATCH_STRIDE}), "
      f"eval_protocol={EVAL_PROTOCOL}")
print(f"Band Split: {args.band_split} -> Low Frac: {SPECTRAL_LOW_FRAC:.3f}, Mid Frac (cumulative): {SPECTRAL_MID_FRAC:.3f}")
print(f"  Trend (Low):    0.000 - {SPECTRAL_LOW_FRAC:.3f}")
print(f"  Seasonal (Mid): {SPECTRAL_LOW_FRAC:.3f} - {SPECTRAL_MID_FRAC:.3f}")
print(f"  Residual (High): {SPECTRAL_MID_FRAC:.3f} - 1.000")
print(f"  Decomposition Mode: {DECOMP_MODE}, Trend Kernel: {TREND_KERNEL if TREND_KERNEL > 0 else max(3, HORIZON // 4)}, Seasonal Kernel: {SEASONAL_KERNEL if SEASONAL_KERNEL > 0 else max(3, (TREND_KERNEL if TREND_KERNEL > 0 else max(3, HORIZON // 4)) // 4)}")
print(f"  Level Loss Weight: {LEVEL_LOSS_WEIGHT:.3f}")
print(f"  Target Loss Weight: {TARGET_LOSS_WEIGHT:.3f}, Monitor Target Channel: {MONITOR_TARGET_CHANNEL}")
print(f"  Spectral Aux Warmup Epochs: {SPECTRAL_WARMUP_EPOCHS}")
if REPEATS > 1:
    print(f"  Repeats: {REPEATS}")
print("=" * 70)
for name, cfg in MODEL_CONFIGS.items():
    model_colors[name] = cfg["color"]
    all_run_results[name] = []

    for repeat_idx in range(REPEATS):
        run_seed = 12345 + repeat_idx
        if REPEATS > 1:
            print(f"\nTraining: {name}  [run {repeat_idx + 1}/{REPEATS}, seed={run_seed}]")
        else:
            print(f"\nTraining: {name}")
        print("-" * 40)

        torch.manual_seed(run_seed)

        patch_len_for_model = PATCH_LEN
        patch_stride_for_model = PATCH_STRIDE
        patchtst_epochs = EPOCHS
        patchtst_lr = LR
        patchtst_batch_size = BATCH_SIZE
        patchtst_patience = PATIENCE
        patchtst_lradj = "TST"
        patchtst_pct_start = 0.3
        patchtst_model_hparams = {}
        if cfg.get("trainer") == "patchtst_supervised" and EVAL_PROTOCOL == "patchtst":
            recipe = PATCHTST_RECIPES.get(args.dataset)
            if recipe is not None:
                if not PATCH_LEN_SET_BY_USER:
                    patch_len_for_model = recipe["patch_len"]
                if not PATCH_STRIDE_SET_BY_USER:
                    patch_stride_for_model = recipe["stride"]
                patchtst_epochs = recipe["epochs"]
                patchtst_lr = recipe["lr"]
                patchtst_batch_size = recipe["batch_size"]
                patchtst_patience = recipe["patience"]
                patchtst_lradj = recipe["lradj"]
                patchtst_pct_start = recipe.get("pct_start", 0.3)
                patchtst_model_hparams = {
                    "d_model": recipe["d_model"],
                    "n_heads": recipe["n_heads"],
                    "d_ff": recipe["d_ff"],
                    "e_layers": recipe["e_layers"],
                    "dropout": recipe["dropout"],
                    "fc_dropout": recipe["fc_dropout"],
                    "head_dropout": recipe["head_dropout"],
                }

        model_kwargs = dict(
            horizon=HORIZON,
            use_skip=USE_SKIP,
            use_revin=USE_REVIN,
            seq_len=WINDOW,
            patch_len=patch_len_for_model,
            patch_stride=patch_stride_for_model,
        )
        if cfg.get("trainer") == "patchtst_supervised":
            model_kwargs["n_vars"] = N_VARS if cfg.get("multivariate", False) else 1
            model_kwargs.update(patchtst_model_hparams)
        if cfg.get("true_multivariate"):
            model_kwargs["n_vars"] = N_VARS

        model = cfg["class"](**model_kwargs).to(DEVICE)

        use_spectral_loss = cfg.get("supports_spectral_loss", False)
        use_shape_aware_loss = use_spectral_loss and cfg.get("shape_aware", False)

        if use_spectral_loss:
            print(f"Using spectral loss with Band Split {args.band_split}")
        if use_shape_aware_loss:
            print(
                f"Using shape-aware trend+forecast penalties (slope={SHAPE_SLOPE_WEIGHT:.3f}, "
                f"curvature={SHAPE_CURVATURE_WEIGHT:.3f})"
            )

        if cfg.get("trainer") == "patchtst_supervised":
            if EVAL_PROTOCOL == "patchtst" and args.dataset in PATCHTST_RECIPES:
                print(
                    f"Applying PatchTST {args.dataset} recipe: "
                    f"epochs={patchtst_epochs}, lr={patchtst_lr}, batch={patchtst_batch_size}, "
                    f"patience={patchtst_patience}, lradj={patchtst_lradj}, "
                    f"patch_len={patch_len_for_model}, stride={patch_stride_for_model}, "
                    f"d_model={patchtst_model_hparams.get('d_model', '?')}, "
                    f"n_heads={patchtst_model_hparams.get('n_heads', '?')}, "
                    f"d_ff={patchtst_model_hparams.get('d_ff', '?')}"
                )
            best_val_loss_norm, best_val_loss_raw, history, best_epoch = train_model_patchtst(
                model,
                X_train_mv_t,
                y_train_mv_t,
                X_val_mv_t,
                y_val_mv_t,
                epochs=patchtst_epochs,
                lr=patchtst_lr,
                batch_size=patchtst_batch_size,
                use_last_epoch=USE_LAST_EPOCH,
                patience=patchtst_patience,
                lradj=patchtst_lradj,
                pct_start=patchtst_pct_start,
            )
            model.eval()
            with torch.no_grad():
                y_val_pred_mv = batched_inference(model, X_val_mv_t)
                y_pred_mv = batched_inference(model, X_test_mv_t)
            y_val_pred = y_val_pred_mv.permute(0, 2, 1).reshape(-1, HORIZON)
            y_pred = y_pred_mv.permute(0, 2, 1).reshape(-1, HORIZON)
        elif cfg.get("true_multivariate"):
            print("Using true multivariate window batches for optimization.")
            best_val_loss_norm, best_val_loss_raw, history, best_epoch = train_model_multivariate(
                model,
                X_train_mv_t,
                y_train_mv_t,
                X_val_mv_t,
                y_val_mv_t,
                use_spectral_loss=use_spectral_loss,
                spectral_low_frac=SPECTRAL_LOW_FRAC,
                spectral_mid_frac=SPECTRAL_MID_FRAC,
                spectral_trend_weight=SPECTRAL_TREND_WEIGHT,
                spectral_warmup_epochs=SPECTRAL_WARMUP_EPOCHS,
                use_shape_aware_loss=use_shape_aware_loss,
                shape_slope_weight=SHAPE_SLOPE_WEIGHT,
                shape_curvature_weight=SHAPE_CURVATURE_WEIGHT,
                decomp_mode=DECOMP_MODE,
                trend_kernel=TREND_KERNEL,
                seasonal_kernel=SEASONAL_KERNEL,
                level_loss_weight=LEVEL_LOSS_WEIGHT,
                target_channel_idx=OT_CHANNEL_IDX,
                target_loss_weight=TARGET_LOSS_WEIGHT,
                monitor_target_channel=MONITOR_TARGET_CHANNEL,
                use_last_epoch=USE_LAST_EPOCH,
            )
            model.eval()
            with torch.no_grad():
                y_val_pred_mv = batched_inference(model, X_val_mv_t)
                y_pred_mv = batched_inference(model, X_test_mv_t)
            y_val_pred = flatten_mv_predictions(y_val_pred_mv)
            y_pred = flatten_mv_predictions(y_pred_mv)
        else:
            use_patchtst_style_mv_batches = FEATURES == "M"
            X_train_fit = X_train_mv_t if use_patchtst_style_mv_batches else X_train_t
            y_train_fit = y_train_mv_t if use_patchtst_style_mv_batches else y_train_t
            X_val_fit = X_val_mv_t if use_patchtst_style_mv_batches else X_val_t
            y_val_fit = y_val_mv_t if use_patchtst_style_mv_batches else y_val_t
            if use_patchtst_style_mv_batches:
                print("Using PatchTST-style multivariate window batches for optimization.")

            if not args.no_ridge_init and hasattr(model, "direct"):
                if ridge_init_direct_path(model, X_train_fit, y_train_fit, X_val_t=X_val_fit, y_val_t=y_val_fit):
                    print("Initialized direct linear path with closed-form ridge solution.")
                    if args.freeze_direct:
                        direct_param_ids = {id(p) for p in model.direct.parameters()}
                        has_trainable_residual = any(
                            p.requires_grad and id(p) not in direct_param_ids
                            for p in model.parameters()
                        )
                        if has_trainable_residual:
                            for p in model.direct.parameters():
                                p.requires_grad_(False)
                            print("Direct linear path frozen; training deep residual only.")
                        else:
                            print("Direct path not frozen: this model has no trainable residual parameters.")

            best_val_loss_norm, best_val_loss_raw, history, best_epoch = train_model(
                model,
                X_train_fit,
                y_train_fit,
                X_val_fit,
                y_val_fit,
                use_spectral_loss=use_spectral_loss,
                spectral_low_frac=SPECTRAL_LOW_FRAC,
                spectral_mid_frac=SPECTRAL_MID_FRAC,
                spectral_trend_weight=SPECTRAL_TREND_WEIGHT,
                spectral_warmup_epochs=SPECTRAL_WARMUP_EPOCHS,
                use_shape_aware_loss=use_shape_aware_loss,
                shape_slope_weight=SHAPE_SLOPE_WEIGHT,
                shape_curvature_weight=SHAPE_CURVATURE_WEIGHT,
                decomp_mode=DECOMP_MODE,
                trend_kernel=TREND_KERNEL,
                seasonal_kernel=SEASONAL_KERNEL,
                level_loss_weight=LEVEL_LOSS_WEIGHT,
                target_channel_idx=OT_CHANNEL_IDX,
                target_loss_weight=TARGET_LOSS_WEIGHT,
                monitor_target_channel=MONITOR_TARGET_CHANNEL,
                use_last_epoch=USE_LAST_EPOCH,
            )

            model.eval()
            if hasattr(model, "deep_scale"):
                print(f"  Learned deep_scale (best epoch): {model.deep_scale.item():.4f}")
            with torch.no_grad():
                y_val_pred = batched_inference(model, X_val_t)
                y_pred = batched_inference(model, X_test_t)

        val_mse_patch, _ = compute_patchtst_metrics(
            y_val_t,
            y_val_pred,
            train_mean,
            train_std,
            already_standardized=EVAL_PROTOCOL == "patchtst",
        )
        test_mse_raw, test_mae_raw = compute_metrics(y_test_t, y_pred)
        test_mse_patch, test_mae_patch = compute_patchtst_metrics(
            y_test_t,
            y_pred,
            train_mean,
            train_std,
            already_standardized=EVAL_PROTOCOL == "patchtst",
        )
        y_test_pred_patch = to_patch_scale_tensor(y_pred, train_mean, train_std)
        test_mse_per_ch_patch, test_mae_per_ch_patch = compute_per_channel_metrics(
            y_test_patch,
            y_test_pred_patch,
            N_TEST_WINDOWS,
            N_VARS,
            feature_cols,
        )
        test_mse_ot_patch = test_mse_per_ch_patch[OT_CHANNEL_NAME]
        test_mae_ot_patch = test_mae_per_ch_patch[OT_CHANNEL_NAME]

        y_true_ot_raw = extract_channel_targets(y_test_patch, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
        y_pred_ot_raw = extract_channel_targets(y_test_pred_patch, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
        test_mse_ot_raw, test_mae_ot_raw = compute_metrics(y_true_ot_raw, y_pred_ot_raw)

        if USE_REVIN and cfg.get("trainer") != "patchtst_supervised":
            test_mse_norm, test_mae_norm = compute_normalized_metrics(X_test_t, y_test_t, y_pred)
        else:
            test_mse_norm, test_mae_norm = test_mse_raw, test_mae_raw

        run_metrics = {
            "best_epoch": best_epoch,
            "best_val_loss_norm": best_val_loss_norm,
            "best_val_loss_raw": best_val_loss_raw,
            "val_mse_patch": val_mse_patch,
            "test_mse_raw": test_mse_raw,
            "test_mae_raw": test_mae_raw,
            "test_mse_patch": test_mse_patch,
            "test_mae_patch": test_mae_patch,
            "test_mse_ot_patch": test_mse_ot_patch,
            "test_mae_ot_patch": test_mae_ot_patch,
            "test_mse_ot_raw": test_mse_ot_raw,
            "test_mae_ot_raw": test_mae_ot_raw,
            "test_mse_per_ch_patch": test_mse_per_ch_patch,
            "test_mae_per_ch_patch": test_mae_per_ch_patch,
            "test_mse_norm": test_mse_norm,
            "test_mae_norm": test_mae_norm,
        }
        all_run_results[name].append(run_metrics)

        # Keep last run's predictions/history for loss-curve + comparison plots
        predictions[name] = (y_pred, cfg["color"])
        all_histories[name] = (history, cfg["color"])

        print(f"  Best Val Loss (Instance-normalized): {best_val_loss_norm:.4f}")
        print(f"  Best Val Loss ({model_space_label} space): {best_val_loss_raw:.4f}")
        print(f"  Val MSE (PatchTST scale): {val_mse_patch:.4f}")
        print(f"  Test MSE/MAE ({model_space_label} space): {test_mse_raw:.4f} / {test_mae_raw:.4f}")
        print(f"  Test MSE/MAE (PatchTST scale): {test_mse_patch:.4f} / {test_mae_patch:.4f}")
        print(f"  Test MSE/MAE (Instance-normalized): {test_mse_norm:.4f} / {test_mae_norm:.4f}")
        print(f"  Test MSE/MAE (OT only, PatchTST scale): {test_mse_ot_patch:.4f} / {test_mae_ot_patch:.4f}")
        print(f"  Test MSE/MAE (OT only, raw units): {test_mse_ot_raw:.4f} / {test_mae_ot_raw:.4f}")
        print(
            f"  Per-channel Test MSE/MAE (PatchTST scale): "
            f"{format_channel_metric_pairs(test_mse_per_ch_patch, test_mae_per_ch_patch, feature_cols)}"
        )

    # Store aggregated (mean across runs) into top-level `results` for the summary table
    _scalar_keys = [
        "best_val_loss_norm", "best_val_loss_raw", "val_mse_patch",
        "test_mse_raw", "test_mae_raw", "test_mse_patch", "test_mae_patch",
        "test_mse_ot_patch", "test_mae_ot_patch", "test_mse_ot_raw", "test_mae_ot_raw",
        "test_mse_norm", "test_mae_norm",
    ]
    _runs = all_run_results[name]
    _agg = {}
    for k in _scalar_keys:
        vals = [r[k] for r in _runs]
        _agg[k] = float(np.mean(vals))
        _agg[k + "_std"] = float(np.std(vals))
    _agg["best_epoch"] = _runs[-1]["best_epoch"]  # last run's epoch for display
    _agg["test_mse_per_ch_patch"] = _runs[-1]["test_mse_per_ch_patch"]
    _agg["test_mae_per_ch_patch"] = _runs[-1]["test_mae_per_ch_patch"]
    results[name] = _agg

# Summary
print("\n" + "=" * 70)
_repeat_tag = f", {REPEATS} runs mean±std" if REPEATS > 1 else ""
print(f"RESULTS SUMMARY ({_epoch_label}{_repeat_tag}) (window={WINDOW} | horizon={HORIZON})")
print("=" * 70)

if REPEATS > 1:
    print(
        f"{'Model':<25} {'Val(P)':<20} {'TestMSE(P)':<20} {'TestMAE(P)':<20} "
        f"{'OT-MSE(P)':<18} {'OT-MAE(P)':<18}"
    )
    print("-" * 120)
    for name, metrics in results.items():
        def _fmt(key):
            m = metrics[key]
            s = metrics.get(key + "_std", 0.0)
            return f"{m:.4f}±{s:.4f}"
        print(
            f"{name:<25}"
            f" {_fmt('val_mse_patch'):<20}"
            f" {_fmt('test_mse_patch'):<20}"
            f" {_fmt('test_mae_patch'):<20}"
            f" {_fmt('test_mse_ot_patch'):<18}"
            f" {_fmt('test_mae_ot_patch'):<18}"
        )
else:
    print(
        f"{'Model':<25} {'Val(InstN)':>12} {'Val(P)':>12} {'TestMSE(P)':>12} {'TestMAE(P)':>12} "
        f"{'OT-MSE(P)':>10} {'OT-MAE(P)':>10}"
    )
    print("-" * 70)
    for name, metrics in results.items():
        epoch_info = f"({metrics['best_epoch']})"
        display_name = f"{name} {epoch_info}"
        print(
            f"{display_name:<25}"
            f" {metrics['best_val_loss_norm']:>12.4f}"
            f" {metrics['val_mse_patch']:>12.4f}"
            f" {metrics['test_mse_patch']:>12.4f}"
            f" {metrics['test_mae_patch']:>12.4f}"
            f" {metrics['test_mse_ot_patch']:>10.4f}"
            f" {metrics['test_mae_ot_patch']:>10.4f}"
        )

best_model = min(results, key=lambda x: results[x]["test_mse_patch"])
print("-" * (120 if REPEATS > 1 else 70))
print(f"Best (by Test MSE (PatchTST scale)): {best_model}")

print("\nPer-channel Test MSE/MAE (PatchTST scale, MSE/MAE):")
for name, metrics in results.items():
    print(
        f"  {name}: "
        f"{format_channel_metric_pairs(metrics['test_mse_per_ch_patch'], metrics['test_mae_per_ch_patch'], feature_cols)}"
    )

output_dir = os.path.join("_OUTPUTS", f"{args.dataset}_{EVAL_PROTOCOL}_{FEATURES}_w{WINDOW}_h{HORIZON}")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as _cfg_f:
    json.dump(
        {
            "source": "training_main",
            "dataset": args.dataset,
            "dataset_label": DATASET_LABEL,
            "eval_protocol": EVAL_PROTOCOL,
            "features": FEATURES,
            "window": WINDOW,
            "horizon": HORIZON,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "use_skip": USE_SKIP,
            "use_revin": USE_REVIN,
            "patch_len": PATCH_LEN,
            "patch_stride": PATCH_STRIDE,
            "repeats": REPEATS,
            "band_split": args.band_split,
            "decomp_mode": DECOMP_MODE,
            "trend_kernel": TREND_KERNEL,
            "seasonal_kernel": SEASONAL_KERNEL,
            "level_loss_weight": LEVEL_LOSS_WEIGHT,
            "target_loss_weight": TARGET_LOSS_WEIGHT,
            "monitor_target_channel": MONITOR_TARGET_CHANNEL,
            "feature_columns": feature_cols,
            "target_col": TARGET_COL,
        },
        _cfg_f,
        indent=2,
    )

_csv_rows = []
for _name, _metrics in results.items():
    _row = {"model": _name}
    for _k, _v in _metrics.items():
        if not isinstance(_v, dict):
            _row[_k] = _v
    _csv_rows.append(_row)
pd.DataFrame(_csv_rows).to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)

plot_loss_curves(all_histories, save_path=os.path.join(output_dir, f"{OUTPUT_PREFIX}_loss_curves.png"))

# Plot OT-only raw-unit forecast for interpretability.
X_test_plot = extract_channel_inputs(X_test_t, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
y_test_plot = extract_channel_targets(y_test_patch, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
predictions_plot = {}
for name, (pred, color) in predictions.items():
    pred_patch = to_patch_scale_tensor(pred, train_mean, train_std)
    pred_ot_raw = extract_channel_targets(pred_patch, N_TEST_WINDOWS, N_VARS, OT_CHANNEL_IDX) * ot_std + ot_mean
    predictions_plot[name] = (pred_ot_raw, color)

plot_comparison(
    X_test_plot,
    y_test_plot,
    predictions_plot,
    save_path=os.path.join(output_dir, f"{OUTPUT_PREFIX}_comparison.png"),
    y_label=f"{OT_CHANNEL_NAME} (raw units)",
)

# Variance plot (only when repeating)
if REPEATS > 1:
    plot_run_variance(all_run_results, model_colors, save_path=os.path.join(output_dir, f"{OUTPUT_PREFIX}_run_variance.png"))

print(f"\nSaved outputs to {output_dir}")