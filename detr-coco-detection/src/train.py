"""Шаги 2-4. Быстрый fine-tuning DETR на COCO-subset с TensorBoard и trace."""
import argparse
import glob
import os
import shutil
import time


import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import DetrForObjectDetection, DetrImageProcessor

from config import CLASSES, MODEL_NAME, PROCESSOR_SIZE
from dataset import CocoDetection, build_collate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
LR_BACKBONE = 1e-5


def make_loader(processor, batch_size, workers, max_samples):
    ds = CocoDetection("data/train/data", "data/train/labels.json", processor)
    if max_samples and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate(processor),
        num_workers=workers,
        pin_memory=DEVICE == "cuda",
        persistent_workers=workers > 0,
    )


def step_loss(model, batch, amp):
    pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
    pixel_mask = batch["pixel_mask"].to(DEVICE, non_blocking=True)
    labels = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
              for t in batch["labels"]]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    return out.loss, out.loss_dict


def optimizer_step(loss, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def run_profiler(model, dl, optimizer, scaler, amp):
    """Снимаем короткий trace для TensorBoard."""
    model.train()
    activities = [ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler("runs/detr/profiler"),
        record_shapes=True,
    ) as prof:
        for i, batch in enumerate(dl):
            loss, _ = step_loss(model, batch, amp)
            optimizer_step(loss, optimizer, scaler)
            prof.step()
            if i >= 4:
                break
    print("Trace профайлера сохранён: runs/detr/profiler")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train-samples", type=int, default=1000)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--clean-checkpoints", action="store_true")
    ap.add_argument("--fresh-logs", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    amp = DEVICE == "cuda" and not args.no_amp
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=PROCESSOR_SIZE)
    train_dl = make_loader(processor, args.batch_size, args.workers, args.max_train_samples)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
    ]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if args.fresh_logs and os.path.exists("runs/detr"):
        shutil.rmtree("runs/detr")
    if args.clean_checkpoints:
        for path in glob.glob("checkpoints/detr_epoch*.pt"):
            os.remove(path)
    writer = SummaryWriter("runs/detr")
    os.makedirs("checkpoints", exist_ok=True)

    print(f"device={DEVICE}, amp={amp}, images={len(train_dl.dataset)}, "
          f"batch={args.batch_size}, epochs={args.epochs}")
    run_profiler(model, train_dl, optimizer, scaler, amp)

    step = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        epoch_started = time.monotonic()
        last = 0.0
        for batch in train_dl:
            loss, ld = step_loss(model, batch, amp)
            optimizer_step(loss, optimizer, scaler)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/loss_ce", ld["loss_ce"].item(), step)
            writer.add_scalar("train/loss_bbox", ld["loss_bbox"].item(), step)
            writer.add_scalar("train/loss_giou", ld["loss_giou"].item(), step)
            step += 1
            last = loss.item()

        torch.save(model.state_dict(), f"checkpoints/detr_epoch{epoch}.pt")
        elapsed = time.monotonic() - epoch_started
        print(f"epoch {epoch}: loss={last:.3f}, time={elapsed:.1f}s")

    writer.close()
    print(f"Готово за {(time.monotonic() - started) / 60:.1f} мин. "
          "Логи: runs/detr | чекпойнты: checkpoints/")


if __name__ == "__main__":
    main()
