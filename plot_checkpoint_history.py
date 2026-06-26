import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


# 修改这里即可切换 checkpoint
CHECKPOINT_PATH = "./best_model.pth"

# 三张图会保存到这个目录
OUTPUT_DIR = "./checkpoint_history_plots"


GENERATOR_LOSS_KEYS = [
    "G_total",
    "cycle",
    "G_L",
    "G_R",
    "recon",
    "illum_prior",
    "illum_match",
    "cross_recon",
    "illum_stats",
    "illum_order",
    "temporal",
]

DISCRIMINATOR_LOSS_KEYS = [
    "D_total",
    "D_L_low",
    "D_L_high",
    "D_R",
]


def load_checkpoint_history(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loss_history = checkpoint.get("loss_history", {}) or {}
    metric_history = checkpoint.get("metric_history", {}) or {}
    best_metric_info = checkpoint.get("best_metric_info", {}) or {}
    return loss_history, metric_history, best_metric_info


def to_float_list(values):
    float_values = []
    for value in values:
        if hasattr(value, "item"):
            value = value.item()
        float_values.append(float(value))
    return float_values


def collect_history(history, keys):
    selected = {}
    for key in keys:
        values = history.get(key)
        if values is not None and len(values) > 0:
            selected[key] = to_float_list(values)
    return selected


def collect_metric_history(metric_history):
    selected = {}
    for key, values in metric_history.items():
        if values is not None and len(values) > 0:
            selected[key] = to_float_list(values)
    return selected


def plot_curves(history, title, ylabel, save_path, best_epoch=None):
    if not history:
        print(f"No data to plot: {title}")
        return

    plt.figure(figsize=(12, 6))

    for key, values in history.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, linewidth=1.8, label=key)

    if best_epoch is not None:
        plt.axvline(
            int(best_epoch),
            color="red",
            linestyle="--",
            linewidth=1.4,
            label=f"best epoch {best_epoch}",
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loss_history, metric_history, best_metric_info = load_checkpoint_history(
        CHECKPOINT_PATH
    )

    generator_losses = collect_history(loss_history, GENERATOR_LOSS_KEYS)
    discriminator_losses = collect_history(loss_history, DISCRIMINATOR_LOSS_KEYS)
    score_metrics = collect_metric_history(metric_history)

    plot_curves(
        generator_losses,
        "Generator Loss",
        "Loss",
        os.path.join(OUTPUT_DIR, "generator_loss.png"),
    )

    plot_curves(
        discriminator_losses,
        "Discriminator Loss",
        "Loss",
        os.path.join(OUTPUT_DIR, "discriminator_loss.png"),
    )

    plot_curves(
        score_metrics,
        "Evaluation Score",
        "Score / Metric",
        os.path.join(OUTPUT_DIR, "score.png"),
        best_epoch=best_metric_info.get("epoch"),
    )

    print(f"Saved 3 plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
