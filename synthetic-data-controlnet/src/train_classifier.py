"""Шаг 4. Ablation-классификатор ResNet18 с корректным real-only validation."""
import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SyntheticDataset(Dataset):
    """Читает неполный набор папок synthetic/, сохраняя id классов real dataset."""

    def __init__(self, root, class_to_idx, transform):
        self.samples = []
        self.transform = transform
        if not os.path.isdir(root):
            return
        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for name in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, name)
                if os.path.isfile(path):
                    self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), target


def make_loaders(use_synthetic, batch_size, workers, real="clf_data", syn="synthetic"):
    real_ds = datasets.ImageFolder(real, transform=TF)
    n_val = max(1, int(0.2 * len(real_ds)))
    train_real, val_ds = random_split(
        real_ds, [len(real_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(0))
    train_ds = train_real
    synthetic_ds = SyntheticDataset(syn, real_ds.class_to_idx, TF)
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
        real_ds.classes,
        len(synthetic_ds) if use_synthetic else 0,
    )


def generated_classes(classes, syn="synthetic"):
    return [c for c in classes if os.path.isdir(os.path.join(syn, c))]


def run(use_synthetic, epochs=5, batch_size=128, workers=4, rare_classes=None):
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
          f"rare_classes={','.join(rare_classes)}")
    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
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
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
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
    print(f"\n[{tag}] accuracy = {acc:.3f}, rare accuracy = {rare_acc:.3f}")
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
