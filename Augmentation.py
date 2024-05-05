import argparse
import os
import random
import shutil
import sys
from PIL import Image, ImageEnhance, ImageFilter


DEBUG_PERCENTAGE = 1
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
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.5)))


AUGMENTATIONS = [flip, rotate, zoom, contrast, brightness, blur]
AUGMENTED_PREFIX = "augmented_"


def save(img, path, augmentation, *, top_level):
    name, ext = os.path.splitext(path)
    path = f"{name}_{augmentation or 'original'}{ext}"
    if not top_level:
        path = AUGMENTED_PREFIX + path
    img.save(path)
    print(f'Transformation saved to "{path}"')


def augment_file(path, *, top_level):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            for augmentation in AUGMENTATIONS:
                save(
                    augmentation(img), path, augmentation.__name__, top_level=top_level
                )
            if not top_level:
                save(img, path, None, top_level=top_level)
    except (IOError, SyntaxError) as e:
        print(f"Error loading image {path}: {e}")
        sys.exit(1)


def augment_directory(path, *, debug_mode, top_level):
    if os.path.isfile(path):
        augment_file(path, top_level=top_level)
        return
    augmented_path = AUGMENTED_PREFIX + path
    shutil.rmtree(augmented_path, ignore_errors=True)
    os.mkdir(augmented_path)
    files = [os.path.join(path, file) for file in os.listdir(path)]
    random.shuffle(files)
    num_files = sum(map(os.path.isfile, files))
    remaining_files = round(num_files * DEBUG_PERCENTAGE / 100)
    for file in files:
        if debug_mode and os.path.isfile(file):
            if remaining_files == 0:
                continue
            remaining_files -= 1
        augment_directory(file, debug_mode=debug_mode, top_level=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path of the file or directory to augment"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Take only {DEBUG_PERCENTAGE}%% of images in each directory",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Create directories with the same image count",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.path):
        augment_directory(args.path, debug_mode=args.debug, top_level=True)
    else:
        print(f"File not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
