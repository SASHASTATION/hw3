"""Экспорт графиков loss из TensorBoard в PNG для отчёта."""
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def values(events, tag):
    points = events.Scalars(tag)
    return [p.step for p in points], [p.value for p in points]


def plot_losses(log_dir="runs/detr", output="viz/losses.png"):
    events = EventAccumulator(log_dir)
    events.Reload()
    tags = {
        "train/loss_ce": "classification loss",
        "train/loss_bbox": "bbox L1 loss",
        "train/loss_giou": "GIoU loss",
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, label in tags.items():
        steps, losses = values(events, tag)
        ax.plot(steps, losses, label=label)
    ax.set(xlabel="step", ylabel="loss", title="DETR training losses")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print("График потерь сохранён:", output)


if __name__ == "__main__":
    plot_losses()
