"""Шаг 6. Error analysis: делим ошибки на классификационные и локализационные.

Для каждой картинки сопоставляем предсказания с GT по IoU (жадно):
  IoU>=0.5 и класс совпал   -> верное детектирование (correct)
  IoU>=0.5 и класс не тот   -> ошибка классификации (бокс на месте, метка неверна)
  IoU<0.5  (нет пары)       -> ошибка локализации / ложное срабатывание
  GT без пары               -> пропуск объекта (missed)
"""
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from transformers import DetrImageProcessor

from dataset import CocoDetection, build_collate
from evaluate import load_model, cxcywh_to_xyxy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def analyze(ckpt="checkpoints/detr_epoch9.pt", thr=0.5):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    ds = CocoDetection("data/val/data", "data/val/labels.json", processor)
    dl = DataLoader(ds, batch_size=4, collate_fn=build_collate(processor))
    model = load_model(ckpt)
    stats = {"correct": 0, "cls_error": 0, "loc_error": 0, "missed": 0}

    for batch in dl:
        pv = batch["pixel_values"].to(DEVICE)
        pm = batch["pixel_mask"].to(DEVICE)
        out = model(pixel_values=pv, pixel_mask=pm)
        sizes = torch.stack([t["orig_size"] for t in batch["labels"]]).to(DEVICE)
        results = processor.post_process_object_detection(out, target_sizes=sizes, threshold=thr)

        for res, t in zip(results, batch["labels"]):
            gt_boxes = cxcywh_to_xyxy(t["boxes"], t["orig_size"].tolist())
            gt_lbls = t["class_labels"]
            pr_boxes = res["boxes"].cpu()
            pr_lbls = res["labels"].cpu()

            if len(gt_boxes) == 0 or len(pr_boxes) == 0:
                stats["missed"] += len(gt_boxes)
                stats["loc_error"] += len(pr_boxes)
                continue

            ious = box_iou(pr_boxes, gt_boxes)
            matched = set()
            for pi in range(len(pr_boxes)):
                best = int(torch.argmax(ious[pi]))
                if ious[pi, best] >= 0.5:
                    if int(pr_lbls[pi]) == int(gt_lbls[best]):
                        stats["correct"] += 1
                    else:
                        stats["cls_error"] += 1
                    matched.add(best)
                else:
                    stats["loc_error"] += 1
            stats["missed"] += len(gt_boxes) - len(matched)

    print("Error analysis:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


if __name__ == "__main__":
    analyze()
