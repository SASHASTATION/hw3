"""Шаг 5. Ablation: два прогона (без и с синтетикой) и таблица сравнения."""
from train_classifier import run

if __name__ == "__main__":
    acc_base = run(use_synthetic=False)
    acc_syn = run(use_synthetic=True)

    print("\n=== Ablation ===")
    print(f"{'Эксперимент':<18}{'Accuracy':>10}")
    print(f"{'Без синтетики':<18}{acc_base:>10.3f}")
    print(f"{'С синтетикой':<18}{acc_syn:>10.3f}")
    print(f"{'Δ':<18}{acc_syn - acc_base:>+10.3f}")
