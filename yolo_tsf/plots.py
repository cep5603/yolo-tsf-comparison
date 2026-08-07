import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(X_test, y_test, predictions, save_path="etth1_comparison.png", max_samples=7, y_label="OT"):
    X_np = X_test.detach().cpu().numpy() if torch.is_tensor(X_test) else np.asarray(X_test)
    y_true_np = y_test.detach().cpu().numpy() if torch.is_tensor(y_test) else np.asarray(y_test)
    
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
            pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred)
            pred_line = np.concatenate([[X_np[sample_idx][-1]], pred_np[sample_idx]])
            ax.plot(t_forecast, pred_line, "--", color=color, lw=2, label=name, alpha=0.85)
        
        ax.axvline(x=input_len - 1, color="gray", ls=":", alpha=0.4)
        ax.set_title(f"Sample {sample_idx+1} (Test Index {sample_idx})", fontweight="bold")
        ax.set_ylabel(y_label)
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
    ax1.set_ylabel("Train Loss (Instance-normalized)")
    ax1.set_title("Training Loss (Instance-normalized)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss (Instance-normalized)")
    ax2.set_title("Validation Loss (Instance-normalized)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train Loss (Model Space)")
    ax3.set_title("Training Loss (Model Space)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Val Loss (Model Space)")
    ax4.set_title("Validation Loss (Model Space)", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_run_variance(all_run_results, model_colors, save_path="etth1_run_variance.png"):
    """Strip/scatter chart: one x-tick per model, dots for each run's val & test MSE."""
    model_names = [n for n in all_run_results if n != "Naive Mean Baseline"]
    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.2), 5))
    x_positions = np.arange(len(model_names))
    jitter_strength = 0.08

    val_plotted = False
    test_plotted = False

    for xi, name in enumerate(model_names):
        runs = all_run_results[name]
        val_mses = [r["val_mse_patch"] for r in runs]
        test_mses = [r["test_mse_patch"] for r in runs]
        color = model_colors.get(name, "gray")
        n = len(val_mses)
        jitter = np.random.default_rng(42).uniform(-jitter_strength, jitter_strength, n)

        ax.scatter(
            np.full(n, xi) + jitter - 0.12,
            val_mses,
            color=color,
            marker="o",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
            s=48,
            label="Val MSE" if not val_plotted else None,
            zorder=3,
        )
        val_plotted = True

        ax.scatter(
            np.full(n, xi) + jitter + 0.12,
            test_mses,
            color=color,
            marker="^",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
            s=48,
            label="Test MSE" if not test_plotted else None,
            zorder=3,
        )
        test_plotted = True

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MSE (PatchTST scale)")
    ax.set_title("Val & Test MSE per Run (variance across seeds)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")
