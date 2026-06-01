"""Шаг 2-4. Fine-tuning DETR на COCO-subset.

Что делает скрипт:
  - грузит предобученный facebook/detr-resnet-50 и меняет голову на 10 классов;
  - учит (бэкбон медленнее, трансформер быстрее);
  - пишет loss / loss_ce / loss_bbox / loss_giou в TensorBoard;
  - снимает trace профайлера (несколько шагов) для TensorBoard;
  - сохраняет чекпойнт после каждой эпохи.
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
from transformers import DetrForObjectDetection, DetrImageProcessor

from config import CLASSES
from dataset import CocoDetection, build_collate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH = 4
LR = 1e-4
LR_BACKBONE = 1e-5


def make_loaders(processor):
    train_ds = CocoDetection("data/train/data", "data/train/labels.json", processor)
    val_ds = CocoDetection("data/val/data", "data/val/labels.json", processor)
    collate = build_collate(processor)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2)
    return train_dl, val_dl


def step_loss(model, batch):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
    out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    return out.loss, out.loss_dict


def run_profiler(model, dl, optimizer):
    """Снимаем trace на нескольких шагах -> папка runs/detr/profiler."""
    model.train()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler("runs/detr/profiler"),
        record_shapes=True,
    ) as prof:
        for i, batch in enumerate(dl):
            loss, _ = step_loss(model, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            prof.step()
            if i >= 5:
                break
    print("Trace профайлера сохранён: runs/detr/profiler")


def main():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    train_dl, _ = make_loaders(processor)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True,   # старая голова на 91 класс -> новая на 10
    ).to(DEVICE)

    # бэкбон учим с меньшим LR, чтобы не разрушить предобученные признаки
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
    ]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    writer = SummaryWriter("runs/detr")
    os.makedirs("checkpoints", exist_ok=True)

    run_profiler(model, train_dl, optimizer)   # шаг 4: trace

    step = 0
    for epoch in range(EPOCHS):
        model.train()
        last = 0.0
        for batch in train_dl:
            loss, ld = step_loss(model, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/loss_ce", ld["loss_ce"].item(), step)       # классификация
            writer.add_scalar("train/loss_bbox", ld["loss_bbox"].item(), step)   # L1 по координатам
            writer.add_scalar("train/loss_giou", ld["loss_giou"].item(), step)   # GIoU
            step += 1
            last = loss.item()

        torch.save(model.state_dict(), f"checkpoints/detr_epoch{epoch}.pt")
        print(f"epoch {epoch}: loss={last:.3f}")

    writer.close()
    print("Готово. Логи: runs/detr | чекпойнты: checkpoints/")


if __name__ == "__main__":
    main()
