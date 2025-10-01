import os, random
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLASSES = ["rock", "paper", "scissors"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def _list_images(folder):
    return [f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in IMG_EXTS]

def count_images_per_class():
    counts = {c: len(_list_images(os.path.join(DATA_DIR, c))) for c in CLASSES}
    total = sum(counts.values())
    print("Images per class:", counts, "| Total:", total)

def show_random_grid(n=9, img_size=128):
    files = []
    for c in CLASSES:
        folder = os.path.join(DATA_DIR, c)
        files += [os.path.join(folder, f) for f in _list_images(folder)]
    random.shuffle(files)
    files = files[:n]

    cols = max(1, int(n**0.5))
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, path in enumerate(files, 1):
        img = Image.open(path).convert("RGB")
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(os.path.dirname(path)))
    plt.tight_layout(); plt.show()

def inspect_image_shapes(sample_per_class=5):
    for c in CLASSES:
        folder = os.path.join(DATA_DIR, c)
        imgs = _list_images(folder)
        if not imgs:
            print(f"{c}: no images found")
            continue
        k = min(sample_per_class, len(imgs))
        paths = random.sample(imgs, k)
        shapes = []
        for p in paths:
            w,h = Image.open(os.path.join(folder, p)).size
            shapes.append((w,h))
        print(f"{c}: example shapes ->", shapes)

if __name__ == "__main__":
    count_images_per_class()
    inspect_image_shapes()
    show_random_grid()
