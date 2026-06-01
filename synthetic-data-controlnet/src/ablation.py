"""Шаг 5. Два прогона ResNet18 и сохранение таблицы ablation."""
import argparse
import json
import os

from train_classifier import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    base = run(False, args.epochs, args.batch_size, args.workers)
    syn = run(True, args.epochs, args.batch_size, args.workers)
    rows = [
        ("Без синтетики", base["accuracy"], base["rare_accuracy"]),
        ("С синтетикой", syn["accuracy"], syn["rare_accuracy"]),
        ("Delta", syn["accuracy"] - base["accuracy"],
         syn["rare_accuracy"] - base["rare_accuracy"]),
    ]

    print("\n=== Ablation ===")
    print(f"{'Эксперимент':<18}{'Accuracy':>12}{'Rare acc.':>12}")
    for name, acc, rare_acc in rows:
        print(f"{name:<18}{acc:>12.3f}{rare_acc:>12.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/ablation.json", "w") as f:
        json.dump({"without_synthetic": base, "with_synthetic": syn}, f, indent=2)
    with open("results/ablation.md", "w") as f:
        f.write("| Эксперимент | Accuracy | Acc. на редких классах |\n")
        f.write("| --- | ---: | ---: |\n")
        for name, acc, rare_acc in rows:
            f.write(f"| {name} | {acc:.3f} | {rare_acc:.3f} |\n")
    print("Таблица сохранена: results/ablation.md")


if __name__ == "__main__":
    main()
