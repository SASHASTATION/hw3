"""Датасет для DETR.

DETR хочет на вход:
  - pixel_values: картинку (нормализованную, ресайз делает processor),
  - labels: список словарей с боксами (нормализ. cx,cy,w,h) и классами.

Берём готовый torchvision.CocoDetection и прогоняем через DetrImageProcessor —
он сам переводит COCO-разметку в нужный DETR формат.
"""
import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_dir, ann_file, processor):
        super().__init__(img_dir, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": anns}
        enc = self.processor(images=img, annotations=target, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0]


def build_collate(processor):
    """Картинки в батче разного размера -> processor.pad дополняет их до общего."""
    def collate_fn(batch):
        pixel_values = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        enc = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],
            "pixel_mask": enc["pixel_mask"],
            "labels": labels,
        }
    return collate_fn
