"""Compare the original classifier against train-only augmentations."""
import argparse
import json
import os

from train_classifier import run as run_baseline
from train_classifier_augmented import run as run_augmented


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-synthetic", action="store_true")
    args = ap.parse_args()

    run_args = (args.use_synthetic, args.epochs, args.batch_size, args.workers)
    baseline = run_baseline(*run_args)
    augmented = run_augmented(*run_args)
    rows = [
        ("Baseline", baseline["accuracy"], baseline["rare_accuracy"]),
        ("Train-only augmentations", augmented["accuracy"],
         augmented["rare_accuracy"]),
        ("Delta", augmented["accuracy"] - baseline["accuracy"],
         augmented["rare_accuracy"] - baseline["rare_accuracy"]),
    ]

    print("\n=== Augmentation ablation ===")
    print(f"{'Эксперимент':<26}{'Accuracy':>12}{'Rare acc.':>12}")
    for name, acc, rare_acc in rows:
        print(f"{name:<26}{acc:>12.3f}{rare_acc:>12.3f}")

    os.makedirs("results", exist_ok=True)
    suffix = "_with_synthetic" if args.use_synthetic else ""
    json_path = f"results/ablation_augmentation{suffix}.json"
    md_path = f"results/ablation_augmentation{suffix}.md"
    with open(json_path, "w") as f:
        json.dump({"baseline": baseline, "train_only_augmentations": augmented},
                  f, indent=2)
    with open(md_path, "w") as f:
        f.write("| Эксперимент | Accuracy | Acc. на редких классах |\n")
        f.write("| --- | ---: | ---: |\n")
        for name, acc, rare_acc in rows:
            f.write(f"| {name} | {acc:.3f} | {rare_acc:.3f} |\n")
    print("Таблица сохранена:", md_path)


if __name__ == "__main__":
    main()
