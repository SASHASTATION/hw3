#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

ensure_data() {
  cd "$ROOT/detr-coco-detection/src"
  if [[ ! -f data/train/labels.json || ! -f data/val/labels.json ]]; then
    "$PYTHON" download_data.py --train-samples 1000 --val-samples 200
  else
    echo "COCO-subset уже скачан, пропускаю download_data.py"
  fi
}

prepare_crops() {
  cd "$ROOT/synthetic-data-controlnet/src"
  "$PYTHON" make_classification_data.py --clean
}

run_detr() {
  cd "$ROOT/detr-coco-detection/src"
  "$PYTHON" train.py --clean-checkpoints --fresh-logs
  "$PYTHON" plot_losses.py
  "$PYTHON" evaluate.py
  "$PYTHON" visualize.py
  "$PYTHON" error_analysis.py
}

run_synthetic() {
  cd "$ROOT/synthetic-data-controlnet/src"
  "$PYTHON" generate_synthetic.py --clean
  "$PYTHON" ablation.py
}

ensure_data
prepare_crops

run_detr &
detr_pid=$!
run_synthetic &
synthetic_pid=$!

set +e
wait "$detr_pid"
detr_status=$?
wait "$synthetic_pid"
synthetic_status=$?
set -e

if [[ "$detr_status" -ne 0 || "$synthetic_status" -ne 0 ]]; then
  echo "Ошибка: DETR status=$detr_status, synthetic status=$synthetic_status" >&2
  exit 1
fi

echo "Обе работы завершены."
