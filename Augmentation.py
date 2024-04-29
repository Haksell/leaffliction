import os
import random
import sys
from PIL import Image, ImageEnhance, ImageFilter

FLIPS = list(Image.Transpose)


def flip(img):
    return img.transpose(random.choice(FLIPS))


def rotate(img):
    return img.rotate(random.randint(10, 350))


def zoom(img):
    percentage = random.randint(5, 25)
    horizontal_crop = img.width * percentage // 100
    vertical_crop = img.height * percentage // 100
    return img.crop(
        (
            horizontal_crop,
            vertical_crop,
            img.width - horizontal_crop,
            img.height - vertical_crop,
        )
    ).resize(img.size)


def random_dissimilar():
    return (
        random.uniform(0.4, 0.8) if random.random() < 0.5 else random.uniform(1.25, 2.5)
    )


def contrast(img):
    return ImageEnhance.Contrast(img).enhance(random_dissimilar())


def brightness(img):
    return ImageEnhance.Brightness(img).enhance(random_dissimilar())


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 3)))


AUGMENTATIONS = [flip, rotate, zoom, contrast, brightness, blur]


def save(img, path, type):
    name, ext = os.path.splitext(path)
    img.save(f"{name}_{type}{ext}")


def augment(img, path):
    for augmentation in AUGMENTATIONS:
        save(augmentation(img), path, augmentation.__name__)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            augment(img, path)
    except (IOError, SyntaxError) as e:
        print(f"Error loading image {path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
