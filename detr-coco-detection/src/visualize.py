"""Шаг 5 (визуализация). Рисуем предсказанные боксы на нескольких картинках.
Результат -> папка viz/.
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor

from config import CLASSES, MODEL_NAME, PROCESSOR_SIZE
from evaluate import load_model, latest_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def visualize(ckpt=None, img_dir="data/val/data", n=6, thr=0.7):
    ckpt = ckpt or latest_checkpoint()
    print("Использую чекпойнт:", ckpt)
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=PROCESSOR_SIZE)
    model = load_model(ckpt)
    os.makedirs("viz", exist_ok=True)

    for f in sorted(os.listdir(img_dir))[:n]:
        img = Image.open(os.path.join(img_dir, f)).convert("RGB")
        enc = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=DEVICE == "cuda"):
            out = model(**enc)
        size = torch.tensor([img.size[::-1]]).to(DEVICE)  # (h, w)
        res = processor.post_process_object_detection(out, target_sizes=size, threshold=thr)[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        for box, lbl, sc in zip(res["boxes"], res["labels"], res["scores"]):
            x0, y0, x1, y1 = box.tolist()
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, color="red", lw=2))
            ax.text(x0, y0, f"{CLASSES[lbl]} {sc:.2f}", color="white", fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.5, pad=1))
        ax.axis("off")
        plt.savefig(f"viz/{f}", bbox_inches="tight")
        plt.close()
    print("Сохранено в viz/")


if __name__ == "__main__":
    visualize()
