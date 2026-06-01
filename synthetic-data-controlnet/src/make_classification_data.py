"""Шаг 1. Делаем датасет для классификации из COCO-subset (из ДЗ2):
вырезаем объекты по GT-боксам в папки по классам.

Структура на выходе:  clf_data/<класс>/<id>.jpg
В конце печатает число примеров по классам — по нему видно «редкие» классы.
"""
import os
import json
from collections import Counter
from PIL import Image

# путь к данным из ДЗ2 (поправьте при необходимости)
COCO_DIR = "../../detr-coco-detection/src/data/train"


def build(coco_dir=COCO_DIR, out="clf_data", min_size=40):
    img_dir = os.path.join(coco_dir, "data")
    anns = json.load(open(os.path.join(coco_dir, "labels.json")))
    images = {im["id"]: im["file_name"] for im in anns["images"]}
    cats = {c["id"]: c["name"] for c in anns["categories"]}
    counter = Counter()

    for a in anns["annotations"]:
        x, y, w, h = a["bbox"]
        if w < min_size or h < min_size:      # пропускаем слишком мелкие объекты
            continue
        name = cats[a["category_id"]]
        img = Image.open(os.path.join(img_dir, images[a["image_id"]])).convert("RGB")
        crop = img.crop((x, y, x + w, y + h))
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        crop.save(os.path.join(d, f"{a['id']}.jpg"))
        counter[name] += 1

    print("Примеров по классам (сверху — самые редкие):")
    for name, n in sorted(counter.items(), key=lambda kv: kv[1]):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    build()
