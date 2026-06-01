# Computer Vision HW3

Две работы запускаются параллельно на одной Tesla A100:

```bash
python -m pip install -r detr-coco-detection/requirements.txt
python -m pip install -r synthetic-data-controlnet/requirements.txt
./run_fast.sh
```

Скрипт скачивает COCO-subset только при отсутствии данных, затем создаёт кропы
и параллельно выполняет:

- fine-tuning DETR, TensorBoard-логи, trace профайлера, mAP, визуализации и
  error analysis;
- генерацию ControlNet для двух автоматически выбранных редких классов и
  ablation ResNet18 без синтетики и с ней.

При уже скачанных весах и данных ожидаемое время на A100: **20–30 минут**.
Первый запуск может занять дольше из-за скачивания COCO и весов моделей.

Основные артефакты:

- `detr-coco-detection/src/runs/detr/`
- `detr-coco-detection/src/checkpoints/`
- `detr-coco-detection/src/viz/`
- `detr-coco-detection/src/results/`
- `synthetic-data-controlnet/src/synthetic/examples.jpg`
- `synthetic-data-controlnet/src/results/ablation.md`
