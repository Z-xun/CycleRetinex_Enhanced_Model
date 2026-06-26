import argparse
import csv
import json
import os
import re
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_CHECKPOINTS = [
    "./working/best_model.pth",
    "./working2/best_model_v2.pth",
    "./best_model.pth",
    "./best_model_v2.pth",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show and export all training history saved in a checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to a .pth checkpoint. If omitted, the script will use an existing default checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <checkpoint_dir>/history_view",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open matplotlib windows after saving figures.",
    )
    return parser.parse_args()


def find_default_checkpoint():
    for path in DEFAULT_CHECKPOINTS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No checkpoint path was given, and no default checkpoint was found."
    )


def to_python_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def as_float_list(values):
    result = []
    for value in values or []:
        value = to_python_value(value)
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(None)
    return result


def clean_history(history):
    cleaned = OrderedDict()
    if not isinstance(history, dict):
        return cleaned
    for key in sorted(history.keys()):
        values = as_float_list(history.get(key))
        if values:
            cleaned[key] = values
    return cleaned


def infer_epoch_from_name(path):
    match = re.search(r"checkpoint_epoch_(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def load_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    loss_history = clean_history(checkpoint.get("loss_history", {}))
    metric_history = clean_history(checkpoint.get("metric_history", {}))
    best_metric_info = checkpoint.get("best_metric_info", {}) or {}
    epoch = checkpoint.get("epoch", infer_epoch_from_name(path))
    return checkpoint, loss_history, metric_history, best_metric_info, epoch


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def max_history_len(*histories):
    max_len = 0
    for history in histories:
        for values in history.values():
            max_len = max(max_len, len(values))
    return max_len


def write_history_csv(loss_history, metric_history, out_path):
    all_keys = []
    for prefix, history in (("loss", loss_history), ("metric", metric_history)):
        for key in history:
            all_keys.append((prefix, key))

    max_len = max_history_len(loss_history, metric_history)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [f"{prefix}_{key}" for prefix, key in all_keys])
        for index in range(max_len):
            row = [index + 1]
            for prefix, key in all_keys:
                history = loss_history if prefix == "loss" else metric_history
                values = history.get(key, [])
                row.append(values[index] if index < len(values) else "")
            writer.writerow(row)


def write_summary(
    checkpoint_path,
    checkpoint,
    loss_history,
    metric_history,
    best_metric_info,
    epoch,
    out_path,
):
    model_keys = [
        key
        for key in checkpoint.keys()
        if key.endswith("state_dict") and "optimizer" not in key and "optim_" not in key
    ]
    optimizer_keys = [
        key
        for key in checkpoint.keys()
        if key.endswith("state_dict") and ("optimizer" in key or "optim_" in key)
    ]

    summary = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "saved_epoch": epoch,
        "loss_keys": list(loss_history.keys()),
        "metric_keys": list(metric_history.keys()),
        "history_epoch_count": max_history_len(loss_history, metric_history),
        "best_metric_info": {
            key: to_python_value(value) for key, value in best_metric_info.items()
        },
        "model_state_dict_keys": model_keys,
        "optimizer_state_dict_keys": optimizer_keys,
        "all_checkpoint_keys": sorted(checkpoint.keys()),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def plot_grouped_history(history, title, ylabel, out_path):
    if not history:
        return False

    plt.figure(figsize=(14, 7))
    for key, values in history.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=key, linewidth=1.8)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_single_histories(history, group_name, out_dir):
    written = []
    for key, values in history.items():
        plt.figure(figsize=(9, 5))
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f"{group_name}: {key}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{group_name}_{key}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()
        written.append(out_path)
    return written


def plot_overview(loss_history, metric_history, best_metric_info, out_path):
    rows = 2 if metric_history else 1
    plt.figure(figsize=(15, 5 * rows))

    plt.subplot(rows, 1, 1)
    for key, values in loss_history.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=key, linewidth=1.6)
    plt.title("All Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8, ncol=2)

    if metric_history:
        plt.subplot(rows, 1, 2)
        for key, values in metric_history.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=key, linewidth=1.8)

        best_epoch = best_metric_info.get("epoch")
        if best_epoch:
            plt.axvline(
                int(best_epoch),
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"best epoch {best_epoch}",
            )

        plt.title("All Metric History")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def print_console_summary(
    checkpoint_path, loss_history, metric_history, best_metric_info, epoch, out_dir
):
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved epoch: {epoch}")
    print(f"History epochs: {max_history_len(loss_history, metric_history)}")
    print(f"Loss keys: {', '.join(loss_history.keys()) or 'none'}")
    print(f"Metric keys: {', '.join(metric_history.keys()) or 'none'}")
    if best_metric_info:
        print("Best metric info:")
        for key, value in best_metric_info.items():
            print(f"  {key}: {to_python_value(value)}")
    print(f"Output directory: {out_dir}")


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or find_default_checkpoint()
    out_dir = args.out_dir
    if out_dir is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path)) or "."
        out_dir = os.path.join(checkpoint_dir, "history_view")
    ensure_out_dir(out_dir)

    checkpoint, loss_history, metric_history, best_metric_info, epoch = load_checkpoint(
        checkpoint_path
    )

    write_history_csv(
        loss_history,
        metric_history,
        os.path.join(out_dir, "history_all.csv"),
    )
    write_summary(
        checkpoint_path,
        checkpoint,
        loss_history,
        metric_history,
        best_metric_info,
        epoch,
        os.path.join(out_dir, "history_summary.json"),
    )

    plot_overview(
        loss_history,
        metric_history,
        best_metric_info,
        os.path.join(out_dir, "history_overview.png"),
    )
    plot_grouped_history(
        loss_history,
        "All Loss History",
        "Loss",
        os.path.join(out_dir, "loss_all.png"),
    )
    plot_grouped_history(
        metric_history,
        "All Metric History",
        "Metric",
        os.path.join(out_dir, "metric_all.png"),
    )
    plot_single_histories(loss_history, "loss", out_dir)
    plot_single_histories(metric_history, "metric", out_dir)

    print_console_summary(
        checkpoint_path, loss_history, metric_history, best_metric_info, epoch, out_dir
    )

    if args.show:
        overview_path = os.path.join(out_dir, "history_overview.png")
        image = plt.imread(overview_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
