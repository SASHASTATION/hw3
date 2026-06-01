"""Шаг 2. Быстрая генерация синтетики через SD 1.5 + ControlNet Canny."""
import argparse
import os
import shutil
import time

import cv2
import numpy as np
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image, ImageDraw, ImageOps

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN = "lllyasviel/sd-controlnet-canny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_canny(img, size):
    a = np.array(img.resize((size, size)))
    edges = cv2.Canny(a, 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def load_pipe(cpu_offload=False):
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(CN, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD, controlnet=controlnet, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if DEVICE == "cuda" and cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)
    return pipe


def discover_rare_classes(src="clf_data", count=2):
    classes = []
    for class_name in os.listdir(src):
        class_dir = os.path.join(src, class_name)
        if os.path.isdir(class_dir):
            n = sum(os.path.isfile(os.path.join(class_dir, f))
                    for f in os.listdir(class_dir))
            if n:
                classes.append((n, class_name))
    return [class_name for _, class_name in sorted(classes)[:count]]


def generate(pipe, class_name, n_per_image, max_source_images, steps, size,
             seed, src="clf_data", out="synthetic", clean=False):
    src_dir = os.path.join(src, class_name)
    out_dir = os.path.join(out, class_name)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Нет кропов класса {class_name}: {src_dir}")
    if clean and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    prompt = f"a photo of a {class_name}, realistic, detailed, natural lighting"
    neg = "blurry, low quality, distorted, cartoon, deformed"
    files = sorted(f for f in os.listdir(src_dir)
                   if os.path.isfile(os.path.join(src_dir, f)))[:max_source_images]
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    paths = []
    for f in files:
        control = to_canny(Image.open(os.path.join(src_dir, f)).convert("RGB"), size)
        for _ in range(n_per_image):
            img = pipe(prompt, image=control, negative_prompt=neg,
                       num_inference_steps=steps, generator=generator).images[0]
            path = os.path.join(out_dir, f"syn_{len(paths)}.jpg")
            img.save(path)
            paths.append(path)
    print(f"{class_name}: сгенерировано {len(paths)} изображений -> {out_dir}")
    return paths


def save_preview(paths, output="synthetic/examples.jpg", max_images=8):
    paths = paths[:max_images]
    if not paths:
        return
    thumb_size = 192
    canvas = Image.new("RGB", (thumb_size * 4, (thumb_size + 24) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        img = ImageOps.fit(Image.open(path).convert("RGB"), (thumb_size, thumb_size))
        x, y = (i % 4) * thumb_size, (i // 4) * (thumb_size + 24)
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + thumb_size + 4), os.path.basename(os.path.dirname(path)),
                  fill="black")
    canvas.save(output)
    print("Превью сохранено:", output)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+",
                    help="По умолчанию выбираются два самых редких непустых класса")
    ap.add_argument("--n-per-image", type=int, default=1)
    ap.add_argument("--max-source-images", type=int, default=24)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--clean", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.monotonic()
    classes = args.classes or discover_rare_classes()
    if len(classes) < 2:
        raise RuntimeError("Нужно минимум два непустых класса в clf_data/")
    print("Редкие классы для генерации:", ", ".join(classes))
    if args.clean and os.path.exists("synthetic"):
        shutil.rmtree("synthetic")
    pipe = load_pipe(args.cpu_offload)
    paths = []
    for i, class_name in enumerate(classes):
        paths += generate(pipe, class_name, args.n_per_image, args.max_source_images,
                          args.steps, args.size, args.seed + i)
    save_preview(paths)
    print(f"Генерация завершена за {(time.monotonic() - started) / 60:.1f} мин.")
