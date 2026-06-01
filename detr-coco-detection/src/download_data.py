"""Шаг 1. Скачиваем подмножество COCO (только наши 10 классов) и
экспортируем в формат COCO (папка с картинками + labels.json).

fiftyone скачивает ТОЛЬКО картинки, где есть нужные классы, поэтому
полный COCO (~20 ГБ) качать не нужно — уйдёт пара гигабайт.
"""
import argparse
import fiftyone as fo
import fiftyone.zoo as foz

from config import CLASSES


def download(split, max_samples, out_dir):
    ds = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        classes=CLASSES,           # берём только картинки с этими классами
        only_matching=True,        # и только метки этих классов
        label_types=["detections"],
        max_samples=max_samples,
    )
    ds.export(
        export_dir=out_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        classes=CLASSES,           # фиксируем порядок -> категории получают id 0..9
    )
    print("Готово:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=1000)
    ap.add_argument("--val-samples", type=int, default=200)
    args = ap.parse_args()
    download("train", args.train_samples, "data/train")
    download("validation", args.val_samples, "data/val")
