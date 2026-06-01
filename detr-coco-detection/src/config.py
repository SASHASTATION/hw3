# 10 классов из COCO. Порядок важен: он задаёт id 0..9 для модели.
CLASSES = [
    "person", "car", "chair", "bottle", "cup",
    "bird", "dog", "cat", "umbrella", "clock",
]

MODEL_NAME = "facebook/detr-resnet-50"

# Уменьшенный размер заметно ускоряет fine-tuning, оставаясь достаточным
# для учебного COCO-subset. Один и тот же размер используется train/eval.
PROCESSOR_SIZE = {"shortest_edge": 640, "longest_edge": 960}
