"""Шаг 4. Обучение классификатора (ResNet18, предобучен на ImageNet).

Флаг --use-synthetic подмешивает синтетику в train.
Запускаем дважды (без флага и с флагом) -> сравниваем accuracy (ablation).
Удобнее запустить ablation.py, который делает оба прогона сразу.
"""
import os
import shutil
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def make_dataset(use_synthetic, real="clf_data", syn="synthetic", merged="train_merged"):
    """Собираем рабочую папку: реальные данные + (опц.) синтетика."""
    if os.path.exists(merged):
        shutil.rmtree(merged)
    shutil.copytree(real, merged)
    if use_synthetic and os.path.exists(syn):
        for cls in os.listdir(syn):
            dst = os.path.join(merged, cls)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(os.path.join(syn, cls)):
                shutil.copy(os.path.join(syn, cls, f), os.path.join(dst, f"syn_{f}"))
    return datasets.ImageFolder(merged, transform=TF)


def run(use_synthetic, epochs=8):
    ds = make_dataset(use_synthetic)
    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

    # оценка: общая accuracy + по классам (важно для редких)
    model.eval()
    correct = total = 0
    per_cls = {c: [0, 0] for c in ds.classes}
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
            for p, t in zip(pred, y):
                per_cls[ds.classes[t]][1] += 1
                if p == t:
                    per_cls[ds.classes[t]][0] += 1

    acc = correct / total
    tag = "С синтетикой" if use_synthetic else "Без синтетики"
    print(f"\n[{tag}] accuracy = {acc:.3f}")
    for c, (ok, n) in per_cls.items():
        if n:
            print(f"  {c}: {ok / n:.3f} ({n})")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-synthetic", action="store_true")
    run(ap.parse_args().use_synthetic)
