"""Шаг 5 (метрики). Считаем mAP и mAP50 на валидации.

DETR отдаёт боксы в нормализованном виде (cx,cy,w,h). torchmetrics ждёт xyxy
в пикселях. Поэтому и предсказания, и GT переводим в xyxy в координатах
исходной картинки (orig_size), затем считаем метрику.
"""
import os
import glob
import warnings
warnings.filterwarnings("ignore")  # глушим безвредные UserWarning про meta-параметры
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrForObjectDetection, DetrImageProcessor

from config import CLASSES
from dataset import CocoDetection, build_collate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def latest_checkpoint(folder="checkpoints"):
    """Берём чекпойнт с самым большим номером эпохи (не важно, сколько прогнали)."""
    files = glob.glob(os.path.join(folder, "detr_epoch*.pt"))
    if not files:
        raise FileNotFoundError(
            f"В папке {folder}/ нет чекпойнтов. Сначала запустите train.py.")
    return max(files, key=lambda f: int(f.split("epoch")[1].split(".")[0]))


def load_model(ckpt):
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=len(CLASSES), ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.to(DEVICE).eval()


def cxcywh_to_xyxy(boxes, size_hw):
    h, w = size_hw
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)


@torch.no_grad()
def evaluate(ckpt=None):
    ckpt = ckpt or latest_checkpoint()
    print("Использую чекпойнт:", ckpt)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    ds = CocoDetection("data/val/data", "data/val/labels.json", processor)
    dl = DataLoader(ds, batch_size=4, collate_fn=build_collate(processor))
    model = load_model(ckpt)
    metric = MeanAveragePrecision(box_format="xyxy")

    for batch in dl:
        pv = batch["pixel_values"].to(DEVICE)
        pm = batch["pixel_mask"].to(DEVICE)
        labels = batch["labels"]
        out = model(pixel_values=pv, pixel_mask=pm)

        sizes = torch.stack([t["orig_size"] for t in labels]).to(DEVICE)  # (B,2) = h,w
        results = processor.post_process_object_detection(out, target_sizes=sizes, threshold=0.0)

        preds, tgts = [], []
        for r, t in zip(results, labels):
            preds.append({"boxes": r["boxes"].cpu(),
                          "scores": r["scores"].cpu(),
                          "labels": r["labels"].cpu()})
            tgts.append({"boxes": cxcywh_to_xyxy(t["boxes"], t["orig_size"].tolist()),
                         "labels": t["class_labels"]})
        metric.update(preds, tgts)

    res = metric.compute()
    print(f"mAP   (0.5:0.95) = {res['map']:.3f}")
    print(f"mAP50 (0.5)      = {res['map_50']:.3f}")
    return res


if __name__ == "__main__":
    evaluate()