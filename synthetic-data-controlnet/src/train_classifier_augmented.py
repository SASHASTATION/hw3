"""ResNet18 classifier with train-only augmentations and deterministic validation."""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms

from train_classifier import SyntheticDataset, generated_classes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                           hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def split_indices(size):
    """Return the same deterministic 80/20 split as the original script."""
    n_val = max(1, int(0.2 * size))
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(size, generator=generator).tolist()
    return indices[:-n_val], indices[-n_val:]


def make_loaders(use_synthetic, batch_size, workers, real="clf_data",
                 syn="synthetic"):
    train_real_ds = datasets.ImageFolder(real, transform=TRAIN_TF)
    val_real_ds = datasets.ImageFolder(real, transform=VAL_TF)
    train_indices, val_indices = split_indices(len(train_real_ds))
    train_real = Subset(train_real_ds, train_indices)
    val_ds = Subset(val_real_ds, val_indices)

    train_ds = train_real
    synthetic_ds = SyntheticDataset(syn, train_real_ds.class_to_idx, TRAIN_TF)
    if use_synthetic and len(synthetic_ds):
        train_ds = ConcatDataset([train_real, synthetic_ds])

    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": DEVICE == "cuda",
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, **loader_args),
        train_real_ds.classes,
        len(synthetic_ds) if use_synthetic else 0,
    )


def run(use_synthetic, epochs=5, batch_size=128, workers=4,
        rare_classes=None):
    train_dl, val_dl, classes, synthetic_count = make_loaders(
        use_synthetic, batch_size, workers)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rare_classes = rare_classes or generated_classes(classes)

    print(f"device={DEVICE}, train={len(train_dl.dataset)}, "
          f"synthetic={synthetic_count}, epochs={epochs}, "
          f"rare_classes={','.join(rare_classes)}, train_augmentations=on")
    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    correct = total = 0
    per_cls = {c: [0, 0] for c in classes}
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp):
                pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
            for p, t in zip(pred, y):
                per_cls[classes[int(t)]][1] += 1
                if p == t:
                    per_cls[classes[int(t)]][0] += 1

    acc = correct / total
    rare_ok = sum(per_cls.get(c, [0, 0])[0] for c in rare_classes)
    rare_total = sum(per_cls.get(c, [0, 0])[1] for c in rare_classes)
    rare_acc = rare_ok / rare_total if rare_total else 0.0
    tag = "С синтетикой" if use_synthetic else "Без синтетики"
    print(f"\n[{tag}, train-only аугментации] "
          f"accuracy = {acc:.3f}, rare accuracy = {rare_acc:.3f}")
    for c, (ok, n) in per_cls.items():
        if n:
            print(f"  {c}: {ok / n:.3f} ({n})")
    return {"accuracy": acc, "rare_accuracy": rare_acc, "per_class": per_cls}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run(args.use_synthetic, args.epochs, args.batch_size, args.workers)
