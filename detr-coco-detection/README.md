# ДЗ 2 — Детектор на DETR (fine-tuning на COCO-subset)

Fine-tuning предобученного **DETR (facebook/detr-resnet-50)** на подмножестве COCO
из 10 классов. Полный цикл: данные → обучение → метрики → визуализация → error analysis.

## Классы (10)
`person, car, chair, bottle, cup, bird, dog, cat, umbrella, clock`
Список задан в `src/config.py`. Порядок задаёт id классов `0..9`.

## Установка
```bash
python -m pip install -r requirements.txt
```
Нужен GPU (подойдёт бесплатный Colab T4). Всё запускается из папки `src/`.

## Что и откуда скачивается
- **Данные COCO** — скачиваются автоматически скриптом `download_data.py`
  через библиотеку `fiftyone`. Качаются только картинки с нужными классами
  (≈2 ГБ), а не весь COCO (~20 ГБ). Ничего вручную качать не надо.
- **Веса DETR** — `facebook/detr-resnet-50` тянется автоматически с Hugging Face
  при первом запуске `train.py`.

## Запуск по шагам
```bash
cd src

# 1) данные: COCO-subset -> data/train, data/val (формат COCO)
python download_data.py

# 2-4) обучение + TensorBoard-логи + trace профайлера + чекпойнты
python train.py

# графики classification / bbox / GIoU loss -> viz/losses.png
python plot_losses.py

# 5) метрики mAP / mAP50 (по умолчанию берётся чекпойнт последней эпохи)
python evaluate.py

# 5) визуализация предсказанных боксов -> папка viz/
python visualize.py

# 6) error analysis (классификация vs локализация)
python error_analysis.py

# Просмотр логов и trace профайлера
tensorboard --logdir runs
```

По умолчанию используется быстрый профиль для A100. Обе работы можно запустить
параллельно из корня репозитория:
```bash
./run_fast.sh
```

## Гиперпараметры
| Параметр            | Значение            |
| ------------------- | ------------------- |
| Модель              | facebook/detr-resnet-50 |
| Эпохи               | 5                   |
| Batch size          | 16                  |
| LR (трансформер)    | 1e-4                |
| LR (бэкбон)         | 1e-5                |
| Optimizer           | AdamW (wd=1e-4)     |
| Размер train / val  | 1000 / 200          |
| Вход                 | shortest edge 640, longest edge 960 |
| Ускорение            | AMP fp16, TF32      |

Бэкбон учим меньшим LR, чтобы не «сломать» предобученные признаки ResNet —
стандартный приём из оригинальной статьи DETR.

## Loss (что логируется в TensorBoard)
- `loss_ce` — классификация (cross-entropy по классам объектов и «нет объекта»);
- `loss_bbox` — L1 по координатам боксов;
- `loss_giou` — Generalized IoU (форма/перекрытие боксов);
- `loss` — их взвешенная сумма (то, что оптимизируется).

DETR сопоставляет предсказания и GT венгерским алгоритмом (один-к-одному),
поэтому NMS не нужен.

## Метрики (заполнить после запуска `evaluate.py`)
| Метрика          | Значение |
| ---------------- | -------- |
| mAP (0.5:0.95)   | …        |
| mAP50            | …        |

## Структура
```
src/
  config.py           # список классов
  download_data.py    # шаг 1: скачать COCO-subset
  dataset.py          # COCO -> формат DETR
  train.py            # шаги 2-4: обучение, TensorBoard, профайлер, чекпойнты
  plot_losses.py      # графики loss для отчёта
  evaluate.py         # шаг 5: mAP / mAP50
  visualize.py        # шаг 5: картинки с боксами
  error_analysis.py   # шаг 6: разбор ошибок
runs/                 # логи TensorBoard + trace профайлера (появятся после train.py)
checkpoints/          # веса по эпохам
viz/                  # визуализации
results/              # таблицы метрик и error analysis
```

## Наблюдения (заполнить своими)
- loss и его компоненты падают; `loss_giou`/`loss_bbox` обычно сходятся медленнее `loss_ce`.
- error analysis: какого типа ошибок больше — классификации (бокс верный, метка нет)
  или локализации (объект найден, но бокс смещён / ложные срабатывания).
- редкие классы (`umbrella`, `clock`, `cat`) детектируются хуже частых (`person`, `car`).
