"""Low-cost DETR fine-tuning without overwriting the baseline experiment.

Changes relative to train.py:
  - moderately larger input images to help box localization;
  - light train-only augmentations;
  - short fine-tuning from an existing checkpoint with a lower learning rate;
  - validation after each epoch and isolated saving of the best checkpoint.
"""
import argparse
import importlib.util
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from transformers import DetrForObjectDetection, DetrImageProcessor

from config import CLASSES, MODEL_NAME
from dataset import build_collate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = {"shortest_edge": 704, "longest_edge": 1056}
TRAIN_INPUT_SIZES = [
    {"shortest_edge": 672, "longest_edge": 1008},
    INPUT_SIZE,
    {"shortest_edge": 736, "longest_edge": 1104},
]


class AugmentedCocoDetection(torchvision.datasets.CocoDetection):
    """COCO dataset with cheap augmentations that keep boxes correct."""

    def __init__(self, img_dir, ann_file, processor, train):
        super().__init__(img_dir, ann_file)
        self.processor = processor
        self.train = train
        self.color_jitter = ColorJitter(brightness=0.1, contrast=0.1)

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        annotations = [dict(annotation) for annotation in annotations]

        if self.train:
            if random.random() < 0.5:
                width = image.width
                image = TF.hflip(image)
                for annotation in annotations:
                    x, y, box_width, box_height = annotation["bbox"]
                    annotation["bbox"] = [
                        width - x - box_width, y, box_width, box_height
                    ]
            image = self.color_jitter(image)

        target = {"image_id": self.ids[idx], "annotations": annotations}
        size = random.choice(TRAIN_INPUT_SIZES) if self.train else INPUT_SIZE
        encoding = self.processor(
            images=image, annotations=target, size=size, return_tensors="pt"
        )
        return encoding["pixel_values"][0], encoding["labels"][0]


def check_dependencies():
    if importlib.util.find_spec("timm") is None:
        raise RuntimeError(
            "Не найден обязательный пакет timm для DETR. Установите зависимости "
            f"в активное окружение:\n{sys.executable} -m pip install -r "
            "../requirements.txt"
        )


def latest_baseline_checkpoint(folder="checkpoints"):
    candidates = []
    for path in Path(folder).glob("detr_epoch*.pt"):
        match = re.fullmatch(r"detr_epoch(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"В папке {folder}/ нет baseline-чекпойнтов. "
            "Передайте путь явно через --checkpoint."
        )
    return max(candidates)[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-val-images", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_output_dir(requested):
    output_dir = requested or Path(
        "experiments", f"detr_improved_{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    return output_dir


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def make_loader(processor, split, batch_size, workers, max_samples, seed):
    train = split == "train"
    dataset = AugmentedCocoDetection(
        f"data/{split}/data", f"data/{split}/labels.json", processor, train=train
    )
    if max_samples and max_samples < len(dataset):
        dataset = Subset(dataset, range(max_samples))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=build_collate(processor),
        num_workers=workers,
        pin_memory=DEVICE == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def load_model(checkpoint):
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(DEVICE)


def configure_optimizer(model, args):
    if not args.unfreeze_backbone:
        for name, parameter in model.named_parameters():
            if "backbone" in name:
                parameter.requires_grad = False

    regular = [
        parameter
        for name, parameter in model.named_parameters()
        if "backbone" not in name and parameter.requires_grad
    ]
    groups = [{"params": regular, "lr": args.learning_rate}]

    backbone = [
        parameter
        for name, parameter in model.named_parameters()
        if "backbone" in name and parameter.requires_grad
    ]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_learning_rate})

    return torch.optim.AdamW(groups, weight_decay=1e-4)


def move_labels(labels):
    return [
        {key: value.to(DEVICE, non_blocking=True) for key, value in target.items()}
        for target in labels
    ]


def cxcywh_to_xyxy(boxes, size_hw):
    height, width = size_hw
    center_x, center_y, box_width, box_height = boxes.unbind(-1)
    return torch.stack(
        [
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        ],
        dim=-1,
    )


def train_epoch(model, loader, optimizer, scaler, amp, writer, global_step):
    model.train()
    total_loss = 0.0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(DEVICE, non_blocking=True)
        labels = move_labels(batch["labels"])

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            output = model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
            )
        scaler.scale(output.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss = output.loss.item()
        total_loss += loss
        writer.add_scalar("train/loss", loss, global_step)
        writer.add_scalar("train/loss_ce", output.loss_dict["loss_ce"].item(), global_step)
        writer.add_scalar(
            "train/loss_bbox", output.loss_dict["loss_bbox"].item(), global_step
        )
        writer.add_scalar(
            "train/loss_giou", output.loss_dict["loss_giou"].item(), global_step
        )
        global_step += 1
    return total_loss / len(loader), global_step


@torch.no_grad()
def validate(model, loader, processor, amp):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        labels = batch["labels"]
        sizes = torch.stack([target["orig_size"] for target in labels]).to(DEVICE)
        results = processor.post_process_object_detection(
            output, target_sizes=sizes, threshold=0.0
        )
        predictions = []
        targets = []
        for result, target in zip(results, labels):
            predictions.append(
                {
                    "boxes": result["boxes"].cpu(),
                    "scores": result["scores"].cpu(),
                    "labels": result["labels"].cpu(),
                }
            )
            targets.append(
                {
                    "boxes": cxcywh_to_xyxy(
                        target["boxes"], target["orig_size"].tolist()
                    ),
                    "labels": target["class_labels"],
                }
            )
        metric.update(predictions, targets)

    values = metric.compute()
    return {
        "map": float(values["map"]),
        "map_50": float(values["map_50"]),
        "map_75": float(values["map_75"]),
    }


def save_history(output_dir, history):
    with (output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)
    with (output_dir / "metrics.md").open("w") as file:
        file.write("| Этап | Train loss | mAP (0.5:0.95) | mAP50 | mAP75 |\n")
        file.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in history:
            loss = "-" if row["train_loss"] is None else f"{row['train_loss']:.3f}"
            file.write(
                f"| {row['stage']} | {loss} | {row['map']:.3f} | "
                f"{row['map_50']:.3f} | {row['map_75']:.3f} |\n"
            )


def main():
    args = parse_args()
    check_dependencies()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = args.checkpoint or latest_baseline_checkpoint()
    output_dir = make_output_dir(args.output_dir)
    amp = DEVICE == "cuda" and not args.no_amp
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=INPUT_SIZE)
    train_loader = make_loader(
        processor,
        "train",
        args.batch_size,
        args.workers,
        args.max_train_samples,
        args.seed,
    )
    val_loader = make_loader(
        processor,
        "val",
        args.batch_size,
        args.workers,
        args.max_val_images,
        args.seed,
    )
    model = load_model(checkpoint)
    optimizer = configure_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    writer = SummaryWriter(output_dir / "runs")

    print(f"device={DEVICE}, amp={amp}, checkpoint={checkpoint}")
    print(f"output={output_dir}")
    print(
        f"resize={INPUT_SIZE}, train_images={len(train_loader.dataset)}, "
        f"val_images={len(val_loader.dataset)}, epochs={args.epochs}"
    )
    print(
        "backbone="
        + ("trainable" if args.unfreeze_backbone else "frozen")
        + f", lr={args.learning_rate:g}"
    )

    started = time.monotonic()
    baseline = validate(model, val_loader, processor, amp)
    history = [{"stage": "baseline", "train_loss": None, **baseline}]
    best_map = baseline["map"]
    best_stage = "baseline"
    best_path = output_dir / "checkpoints" / "best.pt"
    torch.save(model.state_dict(), best_path)
    save_history(output_dir, history)
    print(
        f"baseline: mAP={baseline['map']:.3f}, mAP50={baseline['map_50']:.3f}, "
        f"mAP75={baseline['map_75']:.3f}"
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, amp, writer, global_step
        )
        metrics = validate(model, val_loader, processor, amp)
        stage = f"epoch_{epoch}"
        history.append({"stage": stage, "train_loss": train_loss, **metrics})
        save_history(output_dir, history)
        writer.add_scalar("val/map", metrics["map"], epoch)
        writer.add_scalar("val/map_50", metrics["map_50"], epoch)
        writer.add_scalar("val/map_75", metrics["map_75"], epoch)
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            best_stage = stage
            torch.save(model.state_dict(), best_path)
        print(
            f"{stage}: loss={train_loss:.3f}, mAP={metrics['map']:.3f}, "
            f"mAP50={metrics['map_50']:.3f}, mAP75={metrics['map_75']:.3f}, "
            f"time={time.monotonic() - epoch_started:.1f}s"
        )

    writer.close()
    print(
        f"Готово за {(time.monotonic() - started) / 60:.1f} мин. "
        f"Лучший этап: {best_stage}, mAP={best_map:.3f}"
    )
    print(f"Лучший checkpoint: {best_path}")
    print(f"Сравнение метрик: {output_dir / 'metrics.md'}")


if __name__ == "__main__":
    main()
