"""Шаг 2. Генерируем синтетику для редких классов: Stable Diffusion 1.5 + ControlNet (Canny).

Идея: берём реальный кроп объекта -> строим по нему карту контуров (Canny) ->
ControlNet заставляет SD генерировать новый объект ТОЙ ЖЕ формы, но с другой
текстурой/цветом/фоном. Так получаем разнообразные, но валидные примеры класса.
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                       UniPCMultistepScheduler)

# Старый "runwayml/stable-diffusion-v1-5" удалён с Hugging Face (август 2024).
# Используем официальное зеркало. Запасной вариант: "sd-legacy/stable-diffusion-v1-5".
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN = "lllyasviel/sd-controlnet-canny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_canny(img):
    a = np.array(img.resize((512, 512)))
    edges = cv2.Canny(a, 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def load_pipe():
    controlnet = ControlNetModel.from_pretrained(CN, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()   # экономит VRAM
    return pipe


def generate(pipe, class_name, n_per_image=3, src="clf_data", out="synthetic"):
    src_dir = os.path.join(src, class_name)
    out_dir = os.path.join(out, class_name)
    os.makedirs(out_dir, exist_ok=True)
    prompt = f"a photo of a {class_name}, realistic, detailed, natural lighting"
    neg = "blurry, low quality, distorted, cartoon, deformed"

    k = 0
    for f in os.listdir(src_dir):
        control = to_canny(Image.open(os.path.join(src_dir, f)).convert("RGB"))
        for _ in range(n_per_image):
            img = pipe(prompt, image=control, negative_prompt=neg,
                       num_inference_steps=25).images[0]
            img.save(os.path.join(out_dir, f"syn_{k}.jpg"))
            k += 1
    print(f"{class_name}: сгенерировано {k} изображений -> {out_dir}")


if __name__ == "__main__":
    pipe = load_pipe()
    # редкие классы: посмотрите вывод make_classification_data.py и поправьте список
    for c in ["umbrella", "clock"]:
        generate(pipe, c, n_per_image=3)
